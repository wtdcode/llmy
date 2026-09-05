//! The harness runner: builds the toolbox and system prompt from the
//! configured features, drives the agent loop to completion, injects
//! background notifications between steps, and enforces the finish gates
//! (structured output, forced memory) before the run may end.

use std::path::PathBuf;

use color_eyre::eyre::eyre;
use llmy_agent::tool::ToolBox;
use llmy_client::client::LLM;
use llmy_client::settings::LLMSettings;
use llmy_harness::{Agent, AgentConfig};
use llmy_types::error::LLMYError;
use serde_json::Value;

use crate::files::{FileToolContext, GrepTool, ReadTool, SearchTool, StatTool};
use crate::kg::{KgMemoryContext, KgMemoryDB};
use crate::output::{StructuredOutputState, SubmitResultTool};
use crate::prompts::{
    DEFAULT_MEMORY_INSTRUCTION, SystemPromptSpec, SystemPromptTemplate, render_initial_prompt,
    render_memory_nudge, render_notifications, render_output_nudge,
};
use crate::scratchpad::{ReadScratchpadTool, Scratchpad, UpdateJsonFieldTool};
use crate::state::{HarnessStateDB, ReadToolOutputTool, ToolResultPolicy};
use crate::tasks::{
    CancelMonitorTool, CheckTaskTool, HarnessBashTool, KillTaskTool, MonitorTool,
    ReadTaskOutputTool, TaskConfig, TaskRegistry,
};

/// Memory feature configuration.
#[derive(Debug, Clone)]
pub struct MemoryOptions {
    /// Path of the (cross-run) memory sqlite database.
    pub db_path: String,
    /// Memory instruction defining the level hierarchy and when to write.
    /// `None` uses the built-in default.
    pub instruction: Option<String>,
    /// Refuse to finish the run until at least one memory write happened.
    pub force: bool,
    /// How many times the finish gate nudges the model before giving up.
    pub force_attempts: u64,
    /// Cap on the rendered memory index in the system prompt.
    pub index_max_chars: usize,
}

/// Everything a single harness run needs besides the LLM connection.
#[derive(Debug, Clone)]
pub struct HarnessOptions {
    /// Root directory the file tools and bash commands anchor to.
    pub root: PathBuf,
    /// Path of the run-state sqlite database.
    pub state_db_path: String,
    /// The task prompt (already read from file/arg by the caller).
    pub prompt: String,
    /// Hard cap on agent steps.
    pub max_steps: u64,
    /// Tool results beyond this many characters are truncated in context.
    pub tool_result_chars: usize,
    /// Bash / background-task / monitor family. `None` disables it.
    pub bash: Option<TaskConfig>,
    /// Scratchpad JSON file. `None` disables scratchpad mode.
    pub scratchpad: Option<PathBuf>,
    /// Output JSON schema. `None` means the final message is the output.
    pub output_schema: Option<Value>,
    /// Nudges after a non-conforming stop before the run fails.
    pub output_attempts: u64,
    /// Memory feature. `None` disables memory tools entirely.
    pub memory: Option<MemoryOptions>,
    /// Fixed prompt cache key.
    pub cache_key: Option<String>,
    /// Run tool calls sequentially instead of concurrently.
    pub sequential_tools: bool,
    /// Extra text appended to the system prompt.
    pub append_system_prompt: Option<String>,
    /// Which base system prompt to render.
    pub system_prompt_template: SystemPromptTemplate,
}

impl Default for HarnessOptions {
    fn default() -> Self {
        Self {
            root: PathBuf::from("."),
            state_db_path: "llmy-harness.sqlite3".to_string(),
            prompt: String::new(),
            max_steps: 300,
            tool_result_chars: 20_000,
            bash: None,
            scratchpad: None,
            output_schema: None,
            output_attempts: 3,
            memory: None,
            cache_key: None,
            sequential_tools: false,
            append_system_prompt: None,
            system_prompt_template: SystemPromptTemplate::default(),
        }
    }
}

/// How a run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarnessRunStatus {
    /// All gates passed.
    Completed,
    /// A schema was required but no conforming output was accepted within
    /// the attempt budget.
    OutputRejected,
    /// The step budget ran out.
    MaxSteps,
}

impl HarnessRunStatus {
    pub fn render(&self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::OutputRejected => "output_rejected",
            Self::MaxSteps => "max_steps",
        }
    }
}

/// The result of one harness run.
#[derive(Debug, Clone)]
pub struct HarnessOutcome {
    pub run_id: i64,
    pub steps: u64,
    pub status: HarnessRunStatus,
    /// The model's final assistant message, when it produced one.
    pub final_text: Option<String>,
    /// The accepted structured output, when a schema was configured.
    pub structured: Option<Value>,
}

/// One prepared harness run. Built by [`HarnessRunner::prepare`], consumed by
/// [`HarnessRunner::run`].
pub struct HarnessRunner {
    options: HarnessOptions,
    db: HarnessStateDB,
    run_id: i64,
    agent: Agent,
    registry: Option<TaskRegistry>,
    structured: Option<StructuredOutputState>,
    memory: Option<KgMemoryContext>,
    initial_prompt: String,
}

impl HarnessRunner {
    /// Open the databases, build the toolbox and system prompt, and insert
    /// the run row. `extra_tools` and `extra_sections` let the caller attach
    /// feature bundles the runner does not know about (e.g. codegraph).
    pub async fn prepare(
        options: HarnessOptions,
        llm: &LLM,
        extra_tools: ToolBox,
        extra_sections: Vec<String>,
    ) -> Result<Self, LLMYError> {
        if options.prompt.trim().is_empty() {
            return Err(eyre!("the task prompt is empty").into());
        }

        let db = HarnessStateDB::open(&options.state_db_path).await?;
        let run_id = db
            .begin_run(&llm.model.to_string(), &options.prompt)
            .await?;

        let mut policy = ToolResultPolicy {
            max_result_chars: options.tool_result_chars,
            ..Default::default()
        };
        policy.exempt_tools.insert("read_tool_output".to_string());
        policy.exempt_tools.insert("read_task_output".to_string());

        let mut tools = ToolBox::new();
        let file_context = FileToolContext::new(options.root.clone());
        tools.add_tool(ReadTool::new(file_context.clone()));
        tools.add_tool(GrepTool::new(file_context.clone()));
        tools.add_tool(StatTool::new(file_context.clone()));
        tools.add_tool(SearchTool::new(file_context));
        tools.add_tool(ReadToolOutputTool::new(db.clone(), policy.clone()));

        let registry = match &options.bash {
            Some(config) => {
                let registry = TaskRegistry::new(db.clone(), run_id, config.clone());
                tools.add_tool(HarnessBashTool::new(registry.clone(), options.root.clone()));
                tools.add_tool(CheckTaskTool::new(registry.clone()));
                tools.add_tool(ReadTaskOutputTool::new(
                    registry.clone(),
                    policy.reader_chunk_chars(),
                ));
                tools.add_tool(KillTaskTool::new(registry.clone()));
                tools.add_tool(MonitorTool::new(registry.clone(), options.root.clone()));
                tools.add_tool(CancelMonitorTool::new(registry.clone()));
                Some(registry)
            }
            None => None,
        };

        let scratchpad = match &options.scratchpad {
            Some(path) => {
                let scratchpad = Scratchpad::load(path.clone()).await?;
                tools.add_tool(UpdateJsonFieldTool::new(scratchpad.clone()));
                tools.add_tool(ReadScratchpadTool::new(scratchpad.clone()));
                Some(scratchpad)
            }
            None => None,
        };

        let structured = match &options.output_schema {
            Some(schema) => {
                let state = StructuredOutputState::new(schema.clone())?;
                tools.add_dyn_tool(Box::new(SubmitResultTool::new(state.clone())))?;
                Some(state)
            }
            None => None,
        };

        let (memory, memory_instruction, memory_index) = match &options.memory {
            Some(memory_options) => {
                let context =
                    KgMemoryContext::new(KgMemoryDB::open(&memory_options.db_path).await?);
                tools.extend(context.tool_box());
                let instruction = memory_options
                    .instruction
                    .clone()
                    .unwrap_or_else(|| DEFAULT_MEMORY_INSTRUCTION.to_string());
                let index = context.render_index(memory_options.index_max_chars).await?;
                (Some(context), Some(instruction), Some(index))
            }
            None => (None, None, None),
        };

        tools.extend(extra_tools);
        let recorded = db.record_toolbox(&tools, run_id, &policy)?;

        let spec = SystemPromptSpec {
            template: options.system_prompt_template,
            root: options.root.display().to_string(),
            platform: format!("{} ({})", std::env::consts::OS, std::env::consts::ARCH),
            date: chrono::Utc::now().format("%Y-%m-%d").to_string(),
            bash_enabled: options.bash.is_some(),
            scratchpad_enabled: options.scratchpad.is_some(),
            structured_output: options.output_schema.is_some(),
            memory_instruction,
            memory_index,
            extra_sections,
            appendix: options.append_system_prompt.clone(),
        };

        let mut agent_config = AgentConfig::default();
        if options.sequential_tools {
            agent_config = agent_config.sequential_toolcall();
        }
        let agent = Agent::new_with_config(
            spec.render(),
            recorded,
            options.cache_key.clone(),
            agent_config,
        );

        let initial_prompt = render_initial_prompt(
            &options.prompt,
            scratchpad.as_ref().map(|s| s.render()).as_deref(),
        );

        Ok(Self {
            options,
            db,
            run_id,
            agent,
            registry,
            structured,
            memory,
            initial_prompt,
        })
    }

    pub fn run_id(&self) -> i64 {
        self.run_id
    }

    fn memory_gate_pending(&self) -> bool {
        match (&self.options.memory, &self.memory) {
            (Some(options), Some(context)) => options.force && context.write_count() == 0,
            _ => false,
        }
    }

    fn accepted_output(&self) -> Option<Value> {
        self.structured.as_ref().and_then(|state| state.accepted())
    }

    /// Drive the agent loop until every finish gate passes (or a budget runs
    /// out), then finalize databases and background work.
    pub async fn run(
        mut self,
        llm: &LLM,
        settings: &LLMSettings,
        debug_prefix: Option<&str>,
    ) -> Result<HarnessOutcome, LLMYError> {
        self.agent.push_user_message(self.initial_prompt.clone());

        let mut steps: u64 = 0;
        let mut output_attempts_left = self.options.output_attempts;
        let mut memory_attempts_left = self
            .options
            .memory
            .as_ref()
            .map(|m| m.force_attempts)
            .unwrap_or(0);
        let mut final_text: Option<String> = None;

        let status = loop {
            if steps >= self.options.max_steps {
                tracing::warn!(
                    "run {} hit the step budget of {}",
                    self.run_id,
                    self.options.max_steps
                );
                break HarnessRunStatus::MaxSteps;
            }
            steps += 1;

            let step = self
                .agent
                .step(llm, debug_prefix, Some(settings.clone()))
                .await;
            let step = match step {
                Ok(step) => step,
                Err(error) => {
                    self.finalize("errored", None).await;
                    return Err(error);
                }
            };

            if let Some(registry) = &self.registry {
                let notifications = registry.drain_notifications();
                if !notifications.is_empty() {
                    self.agent
                        .push_user_message(render_notifications(&notifications));
                }
            }

            if step.did_tool_call() {
                // A mid-loop submit_result acceptance ends the run as soon as
                // no other gate is pending.
                if self.accepted_output().is_some() && !self.memory_gate_pending() {
                    break HarnessRunStatus::Completed;
                }
                continue;
            }

            let text = match &step {
                llmy_agent::StepResult::Stop(text) => text.clone(),
                llmy_agent::StepResult::Toolcalled(_) => continue,
            };
            final_text = Some(text.clone());

            if let Some(structured) = &self.structured
                && structured.accepted().is_none()
            {
                if let Err(errors) = structured.try_accept_text(&text) {
                    if output_attempts_left > 0 {
                        output_attempts_left -= 1;
                        tracing::info!(
                            "run {}: final message failed schema validation, nudging ({} attempts left)",
                            self.run_id,
                            output_attempts_left
                        );
                        self.agent
                            .push_user_message(render_output_nudge(&errors, output_attempts_left));
                        continue;
                    }
                    break HarnessRunStatus::OutputRejected;
                }
            }

            if self.memory_gate_pending() {
                if memory_attempts_left > 0 {
                    memory_attempts_left -= 1;
                    tracing::info!(
                        "run {}: no memory written yet, nudging ({} attempts left)",
                        self.run_id,
                        memory_attempts_left
                    );
                    self.agent
                        .push_user_message(render_memory_nudge(memory_attempts_left));
                    continue;
                }
                tracing::warn!(
                    "run {}: memory was required but the model never wrote any; finishing anyway",
                    self.run_id
                );
            }

            break HarnessRunStatus::Completed;
        };

        let structured = self.accepted_output();
        let persisted_output = structured
            .as_ref()
            .map(|value| value.to_string())
            .or_else(|| final_text.clone());
        self.finalize(status.render(), persisted_output.as_deref())
            .await;

        Ok(HarnessOutcome {
            run_id: self.run_id,
            steps,
            status,
            final_text,
            structured,
        })
    }

    async fn finalize(&self, status: &str, final_output: Option<&str>) {
        if let Some(registry) = &self.registry {
            registry.shutdown().await;
        }
        if let Err(error) = self.db.finish_run(self.run_id, status, final_output).await {
            tracing::warn!("failed to finalize run {}: {}", self.run_id, error);
        }
    }
}
