use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use llmy_agent::LLMYError;
use llmy_agent_derive::tool;
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::{
    io::{AsyncRead, AsyncReadExt},
    process::Command,
};

const DEFAULT_TIMEOUT_MS: u64 = 30_000;
const MAX_TIMEOUT_MS: u64 = 600_000;
const MAX_CAPTURE_BYTES: usize = 32 * 1024;

#[derive(Debug, Clone)]
pub struct BashToolConfig {
    pub default_timeout_ms: u64,
    pub max_timeout_ms: u64,
    pub max_capture_bytes: usize,
}

impl Default for BashToolConfig {
    fn default() -> Self {
        Self {
            default_timeout_ms: DEFAULT_TIMEOUT_MS,
            max_timeout_ms: MAX_TIMEOUT_MS,
            max_capture_bytes: MAX_CAPTURE_BYTES,
        }
    }
}

/// Arguments accepted by [`BashTool`].
#[derive(Deserialize, JsonSchema, Default)]
pub struct BashToolArgs {
    /// The shell command to execute.
    pub command: String,
    /// Optional user-facing explanation of what the command does.
    #[serde(default)]
    pub description: Option<String>,
    /// Optional working directory. Relative paths are resolved against the tool root.
    #[serde(default)]
    pub working_directory: Option<PathBuf>,
    /// Optional timeout in milliseconds. Defaults to 30000 and cannot exceed 600000.
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

#[derive(Debug)]
struct CapturedCommandOutput {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    stdout_truncated: bool,
    stderr_truncated: bool,
    exit_code: Option<i32>,
    timed_out: bool,
}

/// Executes shell commands from a configurable working directory.
#[derive(Debug, Clone)]
#[tool(
	arguments = BashToolArgs,
	invoke = bash,
	name = "bash",
	description = "Execute a shell command from the local environment. This tool is dangerous because it can modify files, install packages, or access the network. Commands run from the tool working directory unless `working_directory` is provided.",
)]
pub struct BashTool {
    /// Default working directory used when `working_directory` is not provided.
    pub cwd: PathBuf,
    /// Execution and capture limits for this tool instance.
    pub config: BashToolConfig,
}

impl BashTool {
    /// Creates a bash tool rooted at `cwd`.
    pub fn new(cwd: PathBuf, config: BashToolConfig) -> Self {
        Self { cwd, config }
    }

    fn resolve_timeout(&self, timeout_ms: Option<u64>) -> Result<Duration, String> {
        match timeout_ms {
            Some(0) => Err("timeout_ms must be greater than 0".to_string()),
            Some(ms) if ms > self.config.max_timeout_ms => Err(format!(
                "timeout_ms {} exceeds the maximum of {}",
                ms, self.config.max_timeout_ms
            )),
            Some(ms) => Ok(Duration::from_millis(ms)),
            None => Ok(Duration::from_millis(self.config.default_timeout_ms)),
        }
    }

    fn resolve_working_directory(&self, path: Option<&Path>) -> PathBuf {
        match path {
            Some(path) if path.is_absolute() => path.to_path_buf(),
            Some(path) => self.cwd.join(path),
            None => self.cwd.clone(),
        }
    }

    /// Executes `command` through the system shell and captures stdout/stderr.
    pub async fn bash(&self, args: BashToolArgs) -> Result<String, LLMYError> {
        if args.command.trim().is_empty() {
            return Ok("command must not be empty".to_string());
        }

        let timeout = match self.resolve_timeout(args.timeout_ms) {
            Ok(timeout) => timeout,
            Err(error) => return Ok(error),
        };
        let working_directory = self.resolve_working_directory(args.working_directory.as_deref());

        let output = match run_command(
            &args.command,
            &working_directory,
            timeout,
            self.config.max_capture_bytes,
        )
        .await
        {
            Ok(output) => output,
            Err(error) => {
                return Ok(format!(
                    "Failed to execute command {:?} in {} due to {}",
                    args.command,
                    working_directory.display(),
                    error
                ));
            }
        };

        Ok(render_command_result(
            &args,
            &working_directory,
            timeout,
            self.config.max_capture_bytes,
            output,
        ))
    }
}

fn build_shell_command(command: &str) -> Command {
    #[cfg(windows)]
    {
        let mut shell = Command::new("cmd");
        shell.arg("/C").arg(command);
        shell
    }

    #[cfg(not(windows))]
    {
        let shell_path = std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_string());
        let mut shell = Command::new(shell_path);
        shell.arg("-lc").arg(command);
        shell
    }
}

async fn read_stream_limited<R>(
    mut stream: R,
    max_capture_bytes: usize,
) -> std::io::Result<(Vec<u8>, bool)>
where
    R: AsyncRead + Unpin,
{
    let mut out = Vec::new();
    let mut truncated = false;
    let mut chunk = [0_u8; 4096];

    loop {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            break;
        }

        if out.len() < max_capture_bytes {
            let remaining = max_capture_bytes - out.len();
            let kept = read.min(remaining);
            out.extend_from_slice(&chunk[..kept]);
            if kept < read {
                truncated = true;
            }
        } else {
            truncated = true;
        }
    }

    Ok((out, truncated))
}

async fn run_command(
    command: &str,
    working_directory: &Path,
    timeout: Duration,
    max_capture_bytes: usize,
) -> std::io::Result<CapturedCommandOutput> {
    let mut shell = build_shell_command(command);
    shell.kill_on_drop(true);
    shell.current_dir(working_directory);
    shell.stdin(Stdio::null());
    shell.stdout(Stdio::piped());
    shell.stderr(Stdio::piped());

    let mut child = shell.spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("stdout was not captured"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| std::io::Error::other("stderr was not captured"))?;

    let stdout_task =
        tokio::spawn(async move { read_stream_limited(stdout, max_capture_bytes).await });
    let stderr_task =
        tokio::spawn(async move { read_stream_limited(stderr, max_capture_bytes).await });

    let (status, timed_out) = match tokio::time::timeout(timeout, child.wait()).await {
        Ok(status) => (status?, false),
        Err(_) => {
            let _ = child.kill().await;
            (child.wait().await?, true)
        }
    };

    let (stdout, stdout_truncated) = stdout_task.await.map_err(std::io::Error::other)??;
    let (stderr, stderr_truncated) = stderr_task.await.map_err(std::io::Error::other)??;

    Ok(CapturedCommandOutput {
        stdout,
        stderr,
        stdout_truncated,
        stderr_truncated,
        exit_code: status.code(),
        timed_out,
    })
}

fn format_captured_output(bytes: &[u8], truncated: bool, max_capture_bytes: usize) -> String {
    let text = String::from_utf8_lossy(bytes);
    if text.is_empty() {
        return "(empty)".to_string();
    }

    if truncated {
        format!(
            "{}\n\n[output truncated to the first {} bytes]",
            text, max_capture_bytes
        )
    } else {
        text.into_owned()
    }
}

fn render_command_result(
    args: &BashToolArgs,
    working_directory: &Path,
    timeout: Duration,
    max_capture_bytes: usize,
    output: CapturedCommandOutput,
) -> String {
    let mut sections = vec![];

    if let Some(description) = args.description.as_deref() {
        sections.push(format!("Description: {}", description));
    }

    sections.push(format!("Command: {}", args.command));
    sections.push(format!(
        "Working directory: {}",
        working_directory.display()
    ));
    sections.push(format!("Timeout: {} ms", timeout.as_millis()));
    if output.timed_out {
        sections.push(format!(
            "Status: timed out after {} ms",
            timeout.as_millis()
        ));
    } else if let Some(code) = output.exit_code {
        sections.push(format!("Exit code: {}", code));
    } else {
        sections.push("Exit code: terminated by signal".to_string());
    }

    sections.push(format!(
        "stdout:\n{}",
        format_captured_output(&output.stdout, output.stdout_truncated, max_capture_bytes,)
    ));
    sections.push(format!(
        "stderr:\n{}",
        format_captured_output(&output.stderr, output.stderr_truncated, max_capture_bytes,)
    ));

    sections.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmy_agent::Tool;
    use tempfile::tempdir;

    #[test]
    fn derive_preserves_tool_metadata() {
        assert_eq!(<BashTool as Tool>::NAME, "bash");
    }

    #[tokio::test]
    async fn runs_simple_command() {
        let dir = tempdir().unwrap();
        let tool = BashTool::new(dir.path().to_path_buf(), BashToolConfig::default());

        let result = tool
            .bash(BashToolArgs {
                command: "printf 'hello'".to_string(),
                description: None,
                working_directory: None,
                timeout_ms: Some(1_000),
            })
            .await
            .unwrap();

        assert!(result.contains("Exit code: 0"));
        assert!(result.contains("stdout:\nhello"));
    }

    #[tokio::test]
    async fn resolves_relative_working_directory_against_tool_root() {
        let dir = tempdir().unwrap();
        let nested = dir.path().join("nested");
        tokio::fs::create_dir_all(&nested).await.unwrap();
        let tool = BashTool::new(dir.path().to_path_buf(), BashToolConfig::default());

        let result = tool
            .bash(BashToolArgs {
                command: "pwd".to_string(),
                description: None,
                working_directory: Some(PathBuf::from("nested")),
                timeout_ms: Some(1_000),
            })
            .await
            .unwrap();

        assert!(result.contains(&format!("Working directory: {}", nested.display())));
        assert!(result.contains(&nested.display().to_string()));
    }

    #[tokio::test]
    async fn reports_timeout() {
        let dir = tempdir().unwrap();
        let tool = BashTool::new(dir.path().to_path_buf(), BashToolConfig::default());

        let result = tool
            .bash(BashToolArgs {
                command: "sleep 1".to_string(),
                description: None,
                working_directory: None,
                timeout_ms: Some(10),
            })
            .await
            .unwrap();

        assert!(result.contains("Status: timed out"));
    }

    #[test]
    fn config_defaults_match_previous_constants() {
        let config = BashToolConfig::default();
        assert_eq!(config.default_timeout_ms, DEFAULT_TIMEOUT_MS);
        assert_eq!(config.max_timeout_ms, MAX_TIMEOUT_MS);
        assert_eq!(config.max_capture_bytes, MAX_CAPTURE_BYTES);
    }
}
