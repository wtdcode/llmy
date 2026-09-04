//! Process execution for the harness: a bash tool that never blocks the agent
//! loop forever (a command still running after the foreground window moves to
//! the background), an explicit background mode, task inspection tools, and
//! condition monitors that push a notification into the loop when they fire.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use llmy_types::error::LLMYError;
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};

use crate::state::HarnessStateDB;

/// Limits and defaults for command execution and monitors.
#[derive(Debug, Clone)]
pub struct TaskConfig {
    /// How long a bash command may run in the foreground before it is moved
    /// to the background instead of being killed.
    pub foreground_timeout_ms: u64,
    /// Cap on captured output kept per task (bytes). Output beyond the cap is
    /// dropped and counted.
    pub max_output_bytes: usize,
    /// Characters of output tail included in results and notifications.
    pub tail_chars: usize,
    /// Default polling interval for monitors.
    pub monitor_interval_secs: u64,
    /// Default lifetime of a monitor before it expires.
    pub monitor_timeout_secs: u64,
    /// Hard cap on a single monitor condition check.
    pub monitor_check_timeout_ms: u64,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            foreground_timeout_ms: 180_000,
            max_output_bytes: 4 * 1024 * 1024,
            tail_chars: 2_000,
            monitor_interval_secs: 30,
            monitor_timeout_secs: 1_800,
            monitor_check_timeout_ms: 60_000,
        }
    }
}

/// An event produced outside the agent loop (a finished background task or a
/// fired monitor), delivered to the model as an injected message before its
/// next step.
#[derive(Debug, Clone)]
pub enum HarnessNotification {
    TaskFinished {
        task_id: i64,
        command: String,
        exit_code: Option<i32>,
        killed: bool,
        tail: String,
    },
    MonitorFired {
        monitor_id: i64,
        description: String,
        output: String,
    },
    MonitorExpired {
        monitor_id: i64,
        description: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TaskStatus {
    Running,
    Exited(Option<i32>),
    Killed,
}

impl TaskStatus {
    fn render(&self) -> String {
        match self {
            Self::Running => "running".to_string(),
            Self::Exited(Some(code)) => format!("exited with code {code}"),
            Self::Exited(None) => "exited (terminated by signal)".to_string(),
            Self::Killed => "killed".to_string(),
        }
    }
}

/// Captured, size-capped command output shared between reader tasks and the
/// inspection tools.
#[derive(Debug)]
struct TaskBuffer {
    data: String,
    dropped_bytes: usize,
    cap: usize,
}

impl TaskBuffer {
    fn new(cap: usize) -> Self {
        Self {
            data: String::new(),
            dropped_bytes: 0,
            cap,
        }
    }

    fn append(&mut self, chunk: &[u8]) {
        let remaining = self.cap.saturating_sub(self.data.len());
        if remaining == 0 {
            self.dropped_bytes += chunk.len();
            return;
        }
        let kept = chunk.len().min(remaining);
        self.data.push_str(&String::from_utf8_lossy(&chunk[..kept]));
        if kept < chunk.len() {
            self.dropped_bytes += chunk.len() - kept;
        }
    }

    fn tail(&self, chars: usize) -> String {
        let total = self.data.chars().count();
        let skip = total.saturating_sub(chars);
        let mut out: String = self.data.chars().skip(skip).collect();
        if skip > 0 {
            out = format!("[...{skip} earlier characters omitted]\n{out}");
        }
        if self.dropped_bytes > 0 {
            out.push_str(&format!(
                "\n[{} bytes beyond the capture cap were dropped]",
                self.dropped_bytes
            ));
        }
        out
    }

    fn rendered_len(&self) -> usize {
        self.data.chars().count()
    }

    fn slice(&self, offset: usize, limit: usize) -> String {
        self.data.chars().skip(offset).take(limit).collect()
    }
}

struct TaskEntry {
    command: String,
    working_directory: PathBuf,
    buffer: Arc<StdMutex<TaskBuffer>>,
    status: TaskStatus,
    pid: Option<u32>,
    kill_requested: Arc<AtomicBool>,
    waiter: Option<tokio::task::JoinHandle<()>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MonitorStatus {
    Watching,
    Fired,
    Expired,
    Cancelled,
}

struct MonitorEntry {
    description: String,
    command: String,
    interval_secs: u64,
    status: MonitorStatus,
    abort: tokio::task::AbortHandle,
}

struct TaskRegistryInner {
    db: HarnessStateDB,
    run_id: i64,
    config: TaskConfig,
    tasks: StdMutex<BTreeMap<i64, TaskEntry>>,
    monitors: StdMutex<BTreeMap<i64, MonitorEntry>>,
    monitor_seq: AtomicI64,
    notifications: StdMutex<VecDeque<HarnessNotification>>,
}

/// Shared registry of background tasks and monitors for one harness run.
#[derive(Clone)]
pub struct TaskRegistry {
    inner: Arc<TaskRegistryInner>,
}

impl std::fmt::Debug for TaskRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskRegistry")
            .field("run_id", &self.inner.run_id)
            .finish()
    }
}

struct SpawnedCommand {
    child: Child,
    buffer: Arc<StdMutex<TaskBuffer>>,
}

impl TaskRegistry {
    pub fn new(db: HarnessStateDB, run_id: i64, config: TaskConfig) -> Self {
        Self {
            inner: Arc::new(TaskRegistryInner {
                db,
                run_id,
                config,
                tasks: StdMutex::new(BTreeMap::new()),
                monitors: StdMutex::new(BTreeMap::new()),
                monitor_seq: AtomicI64::new(1),
                notifications: StdMutex::new(VecDeque::new()),
            }),
        }
    }

    pub fn config(&self) -> &TaskConfig {
        &self.inner.config
    }

    /// Take every pending notification, oldest first.
    pub fn drain_notifications(&self) -> Vec<HarnessNotification> {
        match self.inner.notifications.lock() {
            Ok(mut queue) => queue.drain(..).collect(),
            Err(poisoned) => poisoned.into_inner().drain(..).collect(),
        }
    }

    /// Whether any background task is still running or any monitor is still
    /// watching.
    pub fn has_live_work(&self) -> bool {
        let tasks_alive = self
            .lock_tasks()
            .values()
            .any(|entry| entry.status == TaskStatus::Running);
        let monitors_alive = self
            .lock_monitors()
            .values()
            .any(|entry| entry.status == MonitorStatus::Watching);
        tasks_alive || monitors_alive
    }

    fn lock_tasks(&self) -> std::sync::MutexGuard<'_, BTreeMap<i64, TaskEntry>> {
        match self.inner.tasks.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn lock_monitors(&self) -> std::sync::MutexGuard<'_, BTreeMap<i64, MonitorEntry>> {
        match self.inner.monitors.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn push_notification(&self, notification: HarnessNotification) {
        match self.inner.notifications.lock() {
            Ok(mut queue) => queue.push_back(notification),
            Err(poisoned) => poisoned.into_inner().push_back(notification),
        }
    }

    fn spawn_shell(
        &self,
        command: &str,
        working_directory: &Path,
    ) -> std::io::Result<SpawnedCommand> {
        let shell_path = std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_string());
        let mut shell = Command::new(shell_path);
        shell.arg("-lc").arg(command);
        shell.current_dir(working_directory);
        shell.stdin(Stdio::null());
        shell.stdout(Stdio::piped());
        shell.stderr(Stdio::piped());
        shell.kill_on_drop(true);
        #[cfg(unix)]
        shell.process_group(0);

        let mut child = shell.spawn()?;
        let buffer = Arc::new(StdMutex::new(TaskBuffer::new(
            self.inner.config.max_output_bytes,
        )));

        if let Some(stdout) = child.stdout.take() {
            Self::spawn_reader(stdout, buffer.clone());
        }
        if let Some(stderr) = child.stderr.take() {
            Self::spawn_reader(stderr, buffer.clone());
        }

        Ok(SpawnedCommand { child, buffer })
    }

    fn spawn_reader<R>(mut stream: R, buffer: Arc<StdMutex<TaskBuffer>>)
    where
        R: AsyncRead + Unpin + Send + 'static,
    {
        tokio::spawn(async move {
            let mut chunk = [0_u8; 4096];
            loop {
                match stream.read(&mut chunk).await {
                    Ok(0) => break,
                    Ok(read) => match buffer.lock() {
                        Ok(mut guard) => guard.append(&chunk[..read]),
                        Err(poisoned) => poisoned.into_inner().append(&chunk[..read]),
                    },
                    Err(error) => {
                        tracing::debug!("task output reader stopped: {}", error);
                        break;
                    }
                }
            }
        });
    }

    fn kill_process_group(pid: u32) {
        #[cfg(unix)]
        {
            // The child was spawned as its own process group leader, so a
            // negative pid takes the whole tree down with it.
            unsafe {
                libc::kill(-(pid as i32), libc::SIGKILL);
            }
        }
        #[cfg(not(unix))]
        {
            let _ = pid;
        }
    }

    /// Move a still-running command into the background: persist a task row,
    /// register the entry and spawn a waiter that finalizes the row and pushes
    /// a notification once the process exits.
    async fn register_background(
        &self,
        command: String,
        working_directory: PathBuf,
        spawned: SpawnedCommand,
    ) -> Result<i64, LLMYError> {
        let task_id = self
            .inner
            .db
            .insert_background_task(
                self.inner.run_id,
                &command,
                &working_directory.display().to_string(),
            )
            .await?;

        let SpawnedCommand { mut child, buffer } = spawned;
        let pid = child.id();
        let kill_requested = Arc::new(AtomicBool::new(false));

        let registry = self.clone();
        let waiter_kill_flag = kill_requested.clone();
        let waiter_buffer = buffer.clone();
        let waiter_command = command.clone();
        let waiter = tokio::spawn(async move {
            let exit_code = match child.wait().await {
                Ok(status) => status.code(),
                Err(error) => {
                    tracing::warn!("background task {} wait failed: {}", task_id, error);
                    None
                }
            };
            let killed = waiter_kill_flag.load(Ordering::SeqCst);
            let status = if killed {
                TaskStatus::Killed
            } else {
                TaskStatus::Exited(exit_code)
            };

            let (full_output, tail) = match waiter_buffer.lock() {
                Ok(guard) => (
                    guard.data.clone(),
                    guard.tail(registry.inner.config.tail_chars),
                ),
                Err(poisoned) => {
                    let guard = poisoned.into_inner();
                    (
                        guard.data.clone(),
                        guard.tail(registry.inner.config.tail_chars),
                    )
                }
            };

            if let Some(entry) = registry.lock_tasks().get_mut(&task_id) {
                entry.status = status.clone();
            }
            let db_status = if killed { "killed" } else { "exited" };
            if let Err(error) = registry
                .inner
                .db
                .finish_background_task(task_id, db_status, exit_code, &full_output)
                .await
            {
                tracing::warn!("failed to persist background task {}: {}", task_id, error);
            }
            registry.push_notification(HarnessNotification::TaskFinished {
                task_id,
                command: waiter_command,
                exit_code,
                killed,
                tail,
            });
        });

        self.lock_tasks().insert(
            task_id,
            TaskEntry {
                command,
                working_directory,
                buffer,
                status: TaskStatus::Running,
                pid,
                kill_requested,
                waiter: Some(waiter),
            },
        );

        Ok(task_id)
    }

    /// Kill every running task and cancel every monitor, then make sure their
    /// rows are finalized. Called once when the run ends.
    pub async fn shutdown(&self) {
        let waiters: Vec<(i64, tokio::task::JoinHandle<()>)> = {
            let mut tasks = self.lock_tasks();
            let mut waiters = vec![];
            for (id, entry) in tasks.iter_mut() {
                if entry.status == TaskStatus::Running {
                    entry.kill_requested.store(true, Ordering::SeqCst);
                    if let Some(pid) = entry.pid {
                        Self::kill_process_group(pid);
                    }
                }
                if let Some(waiter) = entry.waiter.take() {
                    waiters.push((*id, waiter));
                }
            }
            waiters
        };

        for (task_id, waiter) in waiters {
            if tokio::time::timeout(Duration::from_secs(5), waiter)
                .await
                .is_err()
            {
                tracing::warn!("background task {} did not finalize in time", task_id);
                if let Err(error) = self
                    .inner
                    .db
                    .finish_background_task(task_id, "killed", None, "")
                    .await
                {
                    tracing::warn!("failed to force-finalize task {}: {}", task_id, error);
                }
            }
        }

        let mut monitors = self.lock_monitors();
        for (_, entry) in monitors.iter_mut() {
            if entry.status == MonitorStatus::Watching {
                entry.status = MonitorStatus::Cancelled;
                entry.abort.abort();
            }
        }
    }

    fn resolve_working_directory(&self, root: &Path, path: Option<&Path>) -> PathBuf {
        match path {
            Some(path) if path.is_absolute() => path.to_path_buf(),
            Some(path) => root.join(path),
            None => root.to_path_buf(),
        }
    }

    fn start_monitor(
        &self,
        command: String,
        description: String,
        interval_secs: u64,
        timeout_secs: u64,
        working_directory: PathBuf,
    ) -> i64 {
        let monitor_id = self.inner.monitor_seq.fetch_add(1, Ordering::SeqCst);
        let registry = self.clone();
        let poll_command = command.clone();
        let poll_description = description.clone();
        let handle = tokio::spawn(async move {
            let started = tokio::time::Instant::now();
            let check_timeout =
                Duration::from_millis(registry.inner.config.monitor_check_timeout_ms);
            loop {
                if started.elapsed() >= Duration::from_secs(timeout_secs) {
                    if let Some(entry) = registry.lock_monitors().get_mut(&monitor_id) {
                        entry.status = MonitorStatus::Expired;
                    }
                    registry.push_notification(HarnessNotification::MonitorExpired {
                        monitor_id,
                        description: poll_description,
                    });
                    return;
                }

                match registry.spawn_shell(&poll_command, &working_directory) {
                    Ok(mut spawned) => {
                        let waited =
                            tokio::time::timeout(check_timeout, spawned.child.wait()).await;
                        match waited {
                            Ok(Ok(status)) if status.success() => {
                                // Give the readers a moment to flush the pipes.
                                tokio::time::sleep(Duration::from_millis(50)).await;
                                let output = match spawned.buffer.lock() {
                                    Ok(guard) => guard.tail(registry.inner.config.tail_chars),
                                    Err(poisoned) => {
                                        poisoned.into_inner().tail(registry.inner.config.tail_chars)
                                    }
                                };
                                if let Some(entry) = registry.lock_monitors().get_mut(&monitor_id) {
                                    entry.status = MonitorStatus::Fired;
                                }
                                registry.push_notification(HarnessNotification::MonitorFired {
                                    monitor_id,
                                    description: poll_description,
                                    output,
                                });
                                return;
                            }
                            Ok(Ok(_)) => {}
                            Ok(Err(error)) => {
                                tracing::warn!(
                                    "monitor {} condition wait failed: {}",
                                    monitor_id,
                                    error
                                );
                            }
                            Err(_) => {
                                tracing::warn!("monitor {} condition check timed out", monitor_id);
                            }
                        }
                    }
                    Err(error) => {
                        tracing::warn!(
                            "monitor {} failed to spawn condition: {}",
                            monitor_id,
                            error
                        );
                    }
                }

                tokio::time::sleep(Duration::from_secs(interval_secs)).await;
            }
        });

        self.lock_monitors().insert(
            monitor_id,
            MonitorEntry {
                description,
                command,
                interval_secs,
                status: MonitorStatus::Watching,
                abort: handle.abort_handle(),
            },
        );

        monitor_id
    }
}

/// Arguments accepted by [`HarnessBashTool`].
#[derive(Deserialize, JsonSchema)]
pub struct HarnessBashArgs {
    /// The shell command to execute.
    pub command: String,
    /// Short description of what the command does.
    #[serde(default)]
    pub description: Option<String>,
    /// Optional working directory. Relative paths resolve against the
    /// harness root.
    #[serde(default)]
    pub working_directory: Option<PathBuf>,
    /// Run the command in the background immediately and return a task id.
    #[serde(default)]
    pub run_in_background: Option<bool>,
    /// How long to wait in the foreground (milliseconds) before the command
    /// is moved to the background. Capped by the harness limit.
    #[serde(default)]
    pub foreground_wait_ms: Option<u64>,
}

/// Executes shell commands. Commands that outlive the foreground window keep
/// running in the background instead of being killed.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = HarnessBashArgs,
    invoke = bash,
    name = "bash",
    description = "Execute a shell command from the harness root (or `working_directory`). Commands still running after the foreground window are NOT killed: they move to the background and you receive a task id plus a notification when they finish. Set `run_in_background` to background a command immediately. Use `check_task` / `read_task_output` / `kill_task` to manage background tasks.",
)]
pub struct HarnessBashTool {
    registry: TaskRegistry,
    root: PathBuf,
}

impl HarnessBashTool {
    pub fn new(registry: TaskRegistry, root: PathBuf) -> Self {
        Self { registry, root }
    }

    async fn bash(&self, args: HarnessBashArgs) -> Result<String, LLMYError> {
        if args.command.trim().is_empty() {
            return Ok("command must not be empty".to_string());
        }

        let working_directory = self
            .registry
            .resolve_working_directory(&self.root, args.working_directory.as_deref());
        let mut spawned = match self.registry.spawn_shell(&args.command, &working_directory) {
            Ok(spawned) => spawned,
            Err(error) => {
                return Ok(format!(
                    "Failed to execute command {:?} in {}: {}",
                    args.command,
                    working_directory.display(),
                    error
                ));
            }
        };

        if args.run_in_background.unwrap_or(false) {
            let task_id = self
                .registry
                .register_background(args.command.clone(), working_directory, spawned)
                .await?;
            return Ok(format!(
                "Command started in the background as task #{task_id}. You will be notified when it finishes; use check_task/read_task_output with task_id={task_id} to inspect it meanwhile."
            ));
        }

        let config_cap = self.registry.config().foreground_timeout_ms;
        let wait_ms = args
            .foreground_wait_ms
            .unwrap_or(config_cap)
            .min(config_cap)
            .max(100);

        let waited =
            tokio::time::timeout(Duration::from_millis(wait_ms), spawned.child.wait()).await;
        match waited {
            Ok(Ok(status)) => {
                // Give the pipe readers a moment to flush remaining output.
                tokio::time::sleep(Duration::from_millis(50)).await;
                let tail_chars = self.registry.config().max_output_bytes;
                let output = match spawned.buffer.lock() {
                    Ok(guard) => guard.tail(tail_chars),
                    Err(poisoned) => poisoned.into_inner().tail(tail_chars),
                };
                let exit = match status.code() {
                    Some(code) => format!("Exit code: {code}"),
                    None => "Exit code: terminated by signal".to_string(),
                };
                let output_section = if output.is_empty() {
                    "(no output)".to_string()
                } else {
                    output
                };
                Ok(format!(
                    "Command: {}\nWorking directory: {}\n{}\nOutput:\n{}",
                    args.command,
                    working_directory.display(),
                    exit,
                    output_section
                ))
            }
            Ok(Err(error)) => Ok(format!(
                "Failed while waiting for command {:?}: {}",
                args.command, error
            )),
            Err(_) => {
                let tail = match spawned.buffer.lock() {
                    Ok(guard) => guard.tail(self.registry.config().tail_chars),
                    Err(poisoned) => poisoned
                        .into_inner()
                        .tail(self.registry.config().tail_chars),
                };
                let task_id = self
                    .registry
                    .register_background(args.command.clone(), working_directory, spawned)
                    .await?;
                Ok(format!(
                    "Command did not finish within {wait_ms} ms and now continues in the background as task #{task_id}. It was NOT killed. You will be notified when it finishes; use check_task/read_task_output with task_id={task_id} to inspect it, or kill_task to stop it.\nOutput so far:\n{tail}"
                ))
            }
        }
    }
}

/// Arguments accepted by [`CheckTaskTool`].
#[derive(Deserialize, JsonSchema)]
pub struct CheckTaskArgs {
    /// Task id to inspect. Omit to list every background task of this run.
    #[serde(default)]
    pub task_id: Option<i64>,
}

/// Reports background task status (single task, or all of them).
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = CheckTaskArgs,
    invoke = check,
    name = "check_task",
    description = "Check the status of one background task (by task_id) or list all background tasks and monitors of this run.",
)]
pub struct CheckTaskTool {
    registry: TaskRegistry,
}

impl CheckTaskTool {
    pub fn new(registry: TaskRegistry) -> Self {
        Self { registry }
    }

    async fn check(&self, args: CheckTaskArgs) -> Result<String, LLMYError> {
        let tail_chars = self.registry.config().tail_chars;
        if let Some(task_id) = args.task_id {
            let tasks = self.registry.lock_tasks();
            let Some(entry) = tasks.get(&task_id) else {
                return Ok(format!("No background task with id {task_id}"));
            };
            let tail = match entry.buffer.lock() {
                Ok(guard) => guard.tail(tail_chars),
                Err(poisoned) => poisoned.into_inner().tail(tail_chars),
            };
            return Ok(format!(
                "Task #{task_id}: {}\nCommand: {}\nWorking directory: {}\nOutput tail:\n{}",
                entry.status.render(),
                entry.command,
                entry.working_directory.display(),
                if tail.is_empty() {
                    "(no output yet)"
                } else {
                    tail.as_str()
                }
            ));
        }

        let mut lines = vec![];
        {
            let tasks = self.registry.lock_tasks();
            if tasks.is_empty() {
                lines.push("No background tasks.".to_string());
            } else {
                lines.push("Background tasks:".to_string());
                for (id, entry) in tasks.iter() {
                    lines.push(format!(
                        "- #{}: {} — {}",
                        id,
                        entry.status.render(),
                        entry.command
                    ));
                }
            }
        }
        {
            let monitors = self.registry.lock_monitors();
            if !monitors.is_empty() {
                lines.push("Monitors:".to_string());
                for (id, entry) in monitors.iter() {
                    lines.push(format!(
                        "- monitor #{}: {:?} (every {}s) — {}: {}",
                        id, entry.status, entry.interval_secs, entry.description, entry.command
                    ));
                }
            }
        }
        Ok(lines.join("\n"))
    }
}

/// Arguments accepted by [`ReadTaskOutputTool`].
#[derive(Deserialize, JsonSchema)]
pub struct ReadTaskOutputArgs {
    /// Id of the background task to read output from.
    pub task_id: i64,
    /// 0-based character offset to start reading from. Defaults to 0.
    #[serde(default)]
    pub offset: Option<usize>,
    /// Maximum number of characters to return.
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Pages through the captured output of a background task.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = ReadTaskOutputArgs,
    invoke = read,
    name = "read_task_output",
    description = "Read the captured output of a background task, with optional character offset/limit paging. Works both while the task is running and after it finished.",
)]
pub struct ReadTaskOutputTool {
    registry: TaskRegistry,
    chunk_chars: usize,
}

impl ReadTaskOutputTool {
    pub fn new(registry: TaskRegistry, chunk_chars: usize) -> Self {
        Self {
            registry,
            chunk_chars,
        }
    }

    async fn read(&self, args: ReadTaskOutputArgs) -> Result<String, LLMYError> {
        let tasks = self.registry.lock_tasks();
        let Some(entry) = tasks.get(&args.task_id) else {
            return Ok(format!("No background task with id {}", args.task_id));
        };
        let offset = args.offset.unwrap_or(0);
        let limit = args.limit.unwrap_or(self.chunk_chars).min(self.chunk_chars);
        let (total, chunk, dropped) = match entry.buffer.lock() {
            Ok(guard) => (
                guard.rendered_len(),
                guard.slice(offset, limit),
                guard.dropped_bytes,
            ),
            Err(poisoned) => {
                let guard = poisoned.into_inner();
                (
                    guard.rendered_len(),
                    guard.slice(offset, limit),
                    guard.dropped_bytes,
                )
            }
        };
        let end = offset + chunk.chars().count();
        let mut header = format!(
            "Task #{} ({}), captured {} characters, showing [{}..{})",
            args.task_id,
            entry.status.render(),
            total,
            offset,
            end
        );
        if end < total {
            header.push_str(&format!("; continue with offset={end}"));
        }
        if dropped > 0 {
            header.push_str(&format!("; {dropped} bytes beyond the cap were dropped"));
        }
        Ok(format!("{header}\n{chunk}"))
    }
}

/// Arguments accepted by [`KillTaskTool`].
#[derive(Deserialize, JsonSchema)]
pub struct KillTaskArgs {
    /// Id of the background task to kill.
    pub task_id: i64,
}

/// Kills a background task (its whole process group).
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = KillTaskArgs,
    invoke = kill,
    name = "kill_task",
    description = "Kill a running background task by task_id. The whole process group of the task is terminated.",
)]
pub struct KillTaskTool {
    registry: TaskRegistry,
}

impl KillTaskTool {
    pub fn new(registry: TaskRegistry) -> Self {
        Self { registry }
    }

    async fn kill(&self, args: KillTaskArgs) -> Result<String, LLMYError> {
        let mut tasks = self.registry.lock_tasks();
        let Some(entry) = tasks.get_mut(&args.task_id) else {
            return Ok(format!("No background task with id {}", args.task_id));
        };
        if entry.status != TaskStatus::Running {
            return Ok(format!(
                "Task #{} is not running (status: {})",
                args.task_id,
                entry.status.render()
            ));
        }
        entry.kill_requested.store(true, Ordering::SeqCst);
        match entry.pid {
            Some(pid) => {
                TaskRegistry::kill_process_group(pid);
                Ok(format!(
                    "Kill signal sent to task #{} (pid {}). A completion notification will follow.",
                    args.task_id, pid
                ))
            }
            None => Ok(format!(
                "Task #{} has no pid recorded; it may have already exited.",
                args.task_id
            )),
        }
    }
}

/// Arguments accepted by [`MonitorTool`].
#[derive(Deserialize, JsonSchema)]
pub struct MonitorArgs {
    /// Shell command used as the condition. Exit code 0 means the condition
    /// is satisfied and the monitor fires.
    pub command: String,
    /// What this monitor is waiting for, echoed back in the notification.
    pub description: String,
    /// Polling interval in seconds.
    #[serde(default)]
    pub interval_secs: Option<u64>,
    /// Monitor lifetime in seconds; after this it expires with a notification.
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    /// Optional working directory for the condition command.
    #[serde(default)]
    pub working_directory: Option<PathBuf>,
}

/// Registers a polling condition that injects a notification when satisfied.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = MonitorArgs,
    invoke = monitor,
    name = "monitor",
    description = "Register a condition monitor: the given shell command is run periodically, and when it exits with code 0 you receive a notification carrying its output. Useful to wait for a background task's effect (a file appearing, a port opening, a log line) without polling yourself. Monitors expire after timeout_secs.",
)]
pub struct MonitorTool {
    registry: TaskRegistry,
    root: PathBuf,
}

impl MonitorTool {
    pub fn new(registry: TaskRegistry, root: PathBuf) -> Self {
        Self { registry, root }
    }

    async fn monitor(&self, args: MonitorArgs) -> Result<String, LLMYError> {
        if args.command.trim().is_empty() {
            return Ok("monitor command must not be empty".to_string());
        }
        let config = self.registry.config();
        let interval = args
            .interval_secs
            .unwrap_or(config.monitor_interval_secs)
            .max(1);
        let timeout = args
            .timeout_secs
            .unwrap_or(config.monitor_timeout_secs)
            .max(interval);
        let working_directory = self
            .registry
            .resolve_working_directory(&self.root, args.working_directory.as_deref());
        let monitor_id = self.registry.start_monitor(
            args.command,
            args.description,
            interval,
            timeout,
            working_directory,
        );
        Ok(format!(
            "Monitor #{monitor_id} registered (checked every {interval}s, expires after {timeout}s). You will be notified when it fires or expires; cancel it with cancel_monitor."
        ))
    }
}

/// Arguments accepted by [`CancelMonitorTool`].
#[derive(Deserialize, JsonSchema)]
pub struct CancelMonitorArgs {
    /// Id of the monitor to cancel.
    pub monitor_id: i64,
}

/// Cancels a running monitor.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = CancelMonitorArgs,
    invoke = cancel,
    name = "cancel_monitor",
    description = "Cancel a previously registered monitor by monitor_id.",
)]
pub struct CancelMonitorTool {
    registry: TaskRegistry,
}

impl CancelMonitorTool {
    pub fn new(registry: TaskRegistry) -> Self {
        Self { registry }
    }

    async fn cancel(&self, args: CancelMonitorArgs) -> Result<String, LLMYError> {
        let mut monitors = self.registry.lock_monitors();
        let Some(entry) = monitors.get_mut(&args.monitor_id) else {
            return Ok(format!("No monitor with id {}", args.monitor_id));
        };
        if entry.status != MonitorStatus::Watching {
            return Ok(format!(
                "Monitor #{} is not watching (status: {:?})",
                args.monitor_id, entry.status
            ));
        }
        entry.status = MonitorStatus::Cancelled;
        entry.abort.abort();
        Ok(format!("Monitor #{} cancelled.", args.monitor_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::HarnessStateDB;

    async fn registry_with(config: TaskConfig) -> (tempfile::TempDir, TaskRegistry, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("state.sqlite3");
        let db = HarnessStateDB::open(&db_path.display().to_string())
            .await
            .expect("open state db");
        let run_id = db.begin_run("test-model", "prompt").await.expect("run row");
        let root = dir.path().to_path_buf();
        (dir, TaskRegistry::new(db, run_id, config), root)
    }

    #[tokio::test]
    async fn a_slow_command_moves_to_the_background_and_notifies() {
        let (_dir, registry, root) = registry_with(TaskConfig::default()).await;
        let tool = HarnessBashTool::new(registry.clone(), root);

        let result = tool
            .bash(HarnessBashArgs {
                command: "sleep 0.5; echo done-marker".to_string(),
                description: None,
                working_directory: None,
                run_in_background: None,
                foreground_wait_ms: Some(100),
            })
            .await
            .expect("bash ran");
        assert!(result.contains("continues in the background"), "{result}");
        assert!(result.contains("task #1"), "{result}");
        assert!(registry.has_live_work());

        tokio::time::sleep(Duration::from_millis(900)).await;
        let notifications = registry.drain_notifications();
        assert_eq!(notifications.len(), 1);
        match &notifications[0] {
            HarnessNotification::TaskFinished {
                task_id,
                exit_code,
                killed,
                tail,
                ..
            } => {
                assert_eq!(*task_id, 1);
                assert_eq!(*exit_code, Some(0));
                assert!(!killed);
                assert!(tail.contains("done-marker"), "{tail}");
            }
            other => panic!("expected TaskFinished, got {other:?}"),
        }
        assert!(!registry.has_live_work());
    }

    #[tokio::test]
    async fn fast_commands_finish_in_the_foreground() {
        let (_dir, registry, root) = registry_with(TaskConfig::default()).await;
        let tool = HarnessBashTool::new(registry.clone(), root);

        let result = tool
            .bash(HarnessBashArgs {
                command: "echo fast-marker".to_string(),
                description: None,
                working_directory: None,
                run_in_background: None,
                foreground_wait_ms: None,
            })
            .await
            .expect("bash ran");
        assert!(result.contains("Exit code: 0"), "{result}");
        assert!(result.contains("fast-marker"), "{result}");
        assert!(registry.drain_notifications().is_empty());
    }

    #[tokio::test]
    async fn explicit_background_returns_immediately_and_can_be_killed() {
        let (_dir, registry, root) = registry_with(TaskConfig::default()).await;
        let tool = HarnessBashTool::new(registry.clone(), root);

        let result = tool
            .bash(HarnessBashArgs {
                command: "sleep 30".to_string(),
                description: None,
                working_directory: None,
                run_in_background: Some(true),
                foreground_wait_ms: None,
            })
            .await
            .expect("bash ran");
        assert!(result.contains("background as task #1"), "{result}");

        let killed = KillTaskTool::new(registry.clone())
            .kill(KillTaskArgs { task_id: 1 })
            .await
            .expect("kill ran");
        assert!(killed.contains("Kill signal sent"), "{killed}");

        tokio::time::sleep(Duration::from_millis(500)).await;
        let notifications = registry.drain_notifications();
        assert_eq!(notifications.len(), 1);
        match &notifications[0] {
            HarnessNotification::TaskFinished { killed, .. } => assert!(*killed),
            other => panic!("expected TaskFinished, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_monitor_fires_once_its_condition_holds() {
        let (dir, registry, root) = registry_with(TaskConfig {
            monitor_interval_secs: 1,
            ..TaskConfig::default()
        })
        .await;
        let tool = MonitorTool::new(registry.clone(), root.clone());

        let result = tool
            .monitor(MonitorArgs {
                command: "test -f flag && cat flag".to_string(),
                description: "flag file appears".to_string(),
                interval_secs: Some(1),
                timeout_secs: Some(30),
                working_directory: None,
            })
            .await
            .expect("monitor registered");
        assert!(result.contains("Monitor #1 registered"), "{result}");

        tokio::time::sleep(Duration::from_millis(300)).await;
        std::fs::write(dir.path().join("flag"), "flag-content").expect("write flag");
        tokio::time::sleep(Duration::from_millis(1_600)).await;

        let notifications = registry.drain_notifications();
        assert_eq!(notifications.len(), 1, "{notifications:?}");
        match &notifications[0] {
            HarnessNotification::MonitorFired {
                monitor_id, output, ..
            } => {
                assert_eq!(*monitor_id, 1);
                assert!(output.contains("flag-content"), "{output}");
            }
            other => panic!("expected MonitorFired, got {other:?}"),
        }
        assert!(!registry.has_live_work());
    }

    #[tokio::test]
    async fn shutdown_kills_running_tasks() {
        let (_dir, registry, root) = registry_with(TaskConfig::default()).await;
        let tool = HarnessBashTool::new(registry.clone(), root);
        tool.bash(HarnessBashArgs {
            command: "sleep 30".to_string(),
            description: None,
            working_directory: None,
            run_in_background: Some(true),
            foreground_wait_ms: None,
        })
        .await
        .expect("bash ran");
        assert!(registry.has_live_work());
        registry.shutdown().await;
        assert!(!registry.has_live_work());
    }
}
