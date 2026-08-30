use color_eyre::eyre::eyre;
use llmy_agent::{LLMYError, StepResult, Tool, tool::ToolBox};
use llmy_agent_tools::memory::{
    AgentMemory, AgentMemoryContent, AgentMemoryContext, UpdateMemoryTool, WriteMemoryTool,
};
use llmy_client::debug::completion_to_string;
use llmy_client::model::OpenAIModel;
use llmy_client::req::{ChatCompletionRequestMessageRaw, PromptCacheMode, PromptCacheOptionsRaw};
use llmy_client::resp::{ChatChoice, FinishReason};
use llmy_client::{
    client::{LLM, LLMRequest, Message, MessagePart, RawExtensibleChatCompletionRequest},
    model::ModelConfig,
    settings::LLMSettings,
};
use llmy_types::error::GeneralToolCall;

use crate::{
    memory::AgentMemorySystemPromptCriteria,
    prompt::{
        render_compact_system_prompt, render_compact_user_prompt, render_compacted_context_message,
    },
    utils::chat_choice_to_toolcalls,
};

#[derive(Clone)]
struct AgentMemoryRuntime {
    context: AgentMemoryContext,
    criteria: AgentMemorySystemPromptCriteria,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentConfig {
    pub sequential_tool_call: bool,
    pub allow_empty_tool_calls: bool,
    /// How many times a step re-asks the model after a turn is discarded by a
    /// tool rejection ([`LLMYError::ToolCallRejected`]) before giving up and
    /// surfacing the rejection to the caller.
    pub tool_reject_retries: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            sequential_tool_call: false,
            allow_empty_tool_calls: false,
            tool_reject_retries: 32,
        }
    }
}

impl AgentConfig {
    pub fn from_model(&self, model: &OpenAIModel) -> Self {
        let mut config = self.clone();
        if model.is_mimo() {
            config.allow_empty_tool_calls = true;
        }
        config
    }

    pub fn sequential_toolcall(mut self) -> Self {
        self.sequential_tool_call = true;
        self
    }
}

/// Agent implementation backed by an in-memory conversation context and
/// toolbox. The context is held in protocol-neutral [`Message`] form, so the
/// same agent runs natively on every backend protocol without loss.
#[derive(Clone)]
pub struct Agent {
    base_system_prompt: String,
    system_prompt: String,
    tools: ToolBox,
    context: Vec<Message>,
    checkpoints: Vec<(Option<StepResult>, Vec<Message>)>,
    last_step: Option<StepResult>,
    /// Fixed `prompt_cache_key` for this conversation. `None` leaves the choice
    /// to the client, which picks one per prompt prefix.
    cache_key: Option<String>,
    memory: Option<AgentMemoryRuntime>,
    config: AgentConfig,
    /// Request-wide prompt cache mode. `None` (the default) omits
    /// `prompt_cache_options` entirely, leaving the provider's own default.
    cache_mode: Option<PromptCacheMode>,
    /// Whether the synthesized system message carries a cache breakpoint. It is
    /// not stored in `context`, so it needs its own flag.
    system_breakpoint: bool,
}

impl Agent {
    pub fn new(system_prompt: String, tools: ToolBox, cache_key: Option<String>) -> Self {
        Self::new_with_config(system_prompt, tools, cache_key, AgentConfig::default())
    }

    pub fn new_with_config(
        system_prompt: String,
        tools: ToolBox,
        cache_key: Option<String>,
        config: AgentConfig,
    ) -> Self {
        Self {
            base_system_prompt: system_prompt.clone(),
            system_prompt,
            tools,
            checkpoints: vec![],
            context: vec![],
            last_step: None,
            cache_key,
            memory: None,
            config,
            cache_mode: None,
            system_breakpoint: false,
        }
    }

    pub async fn with_memory(
        system_prompt: String,
        tools: ToolBox,
        cache_key: Option<String>,
        memory: &AgentMemoryContext,
        criteria: &AgentMemorySystemPromptCriteria,
    ) -> Self {
        Self::with_memory_config(
            system_prompt,
            tools,
            cache_key,
            memory,
            criteria,
            AgentConfig::default(),
        )
        .await
    }

    pub async fn with_memory_config(
        system_prompt: String,
        mut tools: ToolBox,
        cache_key: Option<String>,
        memory: &AgentMemoryContext,
        criteria: &AgentMemorySystemPromptCriteria,
        config: AgentConfig,
    ) -> Self {
        tools.extend(memory.tool_box());
        let guard = memory.memory.read().await;
        let rendered_system_prompt = criteria.render_system_prompt(&system_prompt, &guard);
        Self {
            base_system_prompt: system_prompt,
            system_prompt: rendered_system_prompt,
            tools,
            context: vec![],
            checkpoints: vec![],
            last_step: None,
            cache_key,
            memory: Some(AgentMemoryRuntime {
                context: memory.clone(),
                criteria: criteria.clone(),
            }),
            config,
            cache_mode: None,
            system_breakpoint: false,
        }
    }

    pub fn conversation_context(&self) -> Vec<Message> {
        let mut system = Message::system(self.system_prompt.clone());
        if self.system_breakpoint {
            system.breakpoint();
        }
        std::iter::once(system)
            .chain(self.context.iter().cloned())
            .collect()
    }

    /// Mutable access to the stored conversation, so callers can mark a cache
    /// breakpoint on a specific message:
    ///
    /// ```no_run
    /// # fn mark(agent: &mut llmy_harness::Agent) {
    /// if let Some(msg) = agent.context_mut().last_mut() {
    ///     msg.breakpoint();
    /// }
    /// # }
    /// ```
    ///
    /// This is the conversation only. The system-prompt message is synthesized
    /// per request by [`Agent::conversation_context`], so it gets its own switch:
    /// [`Agent::toggle_system_breakpoint`].
    pub fn context_mut(&mut self) -> &mut Vec<Message> {
        &mut self.context
    }

    /// Toggle a cache breakpoint on the system-prompt message, which is
    /// synthesized per request and so cannot be reached via
    /// [`Agent::context_mut`]. Off by default.
    ///
    /// This is usually the breakpoint worth having: the system prompt is the
    /// stable head of every request.
    pub fn toggle_system_breakpoint(&mut self, enabled: bool) {
        self.system_breakpoint = enabled;
    }

    /// Whether the system-prompt message currently carries a cache breakpoint.
    pub fn system_breakpoint(&self) -> bool {
        self.system_breakpoint
    }

    /// Drop every explicit cache breakpoint, in the conversation and on the
    /// system prompt.
    pub fn clear_breakpoints(&mut self) {
        for message in self.context.iter_mut() {
            message.toggle_cache_breakpoint(false);
        }
        self.system_breakpoint = false;
    }

    /// Choose how the provider places cache breakpoints for this agent.
    ///
    /// `explicit = true` suppresses the implicit breakpoint the API would
    /// otherwise put on the latest message, so only the ones marked through
    /// [`Agent::context_mut`] are used for cache reads and writes.
    /// `explicit = false` restores the provider default (implicit breakpoint
    /// *plus* the explicit ones).
    ///
    /// Until this is called, `prompt_cache_options` is left off the request
    /// entirely. See [`Agent::breakpoint_mode`].
    pub fn toggle_breakpoint_implicit(&mut self, explicit: bool) {
        self.cache_mode = Some(if explicit {
            PromptCacheMode::Explicit
        } else {
            PromptCacheMode::Implicit
        });
    }

    /// The prompt cache mode set by [`Agent::toggle_breakpoint_implicit`], or
    /// `None` while the provider default is in force.
    pub fn breakpoint_mode(&self) -> Option<PromptCacheMode> {
        self.cache_mode
    }

    /// Attach this agent's `prompt_cache_options` to an outgoing request.
    ///
    /// Skipped for models that don't address their cache by breakpoint: the
    /// field is unknown to them, and strict providers reject unknown request
    /// fields outright.
    fn apply_cache_options(&self, llm: &LLM, req: &mut RawExtensibleChatCompletionRequest) {
        let Some(mode) = self.cache_mode else {
            return;
        };
        let policy = llm.model.cache_policy();
        if !policy.needs_breakpoints() {
            tracing::warn!(
                "dropping prompt cache mode {:?}: {} caches by {}, not by breakpoint",
                mode,
                llm.model,
                policy
            );
            return;
        }
        req.prompt_cache_options = Some(PromptCacheOptionsRaw::with_mode(mode));
    }

    pub fn render_context(&self) -> String {
        Message::many_to_chat(&self.conversation_context())
            .iter()
            .map(|m| completion_to_string(&m.0))
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn render_tools(&self, details: bool) -> String {
        let tools = self.tools.render_tools(details);

        if tools.is_empty() {
            return "No tools are enabled for this chat.".to_string();
        }

        format!("Enabled tools ({}):\n- {}", tools.len(), tools.join("\n- "))
    }

    /// Mutable access to the agent's toolbox, for attaching or detaching
    /// tools at runtime (e.g. `agent.tools_mut().extend(bundle)` or
    /// `agent.tools_mut().remove_tool("read_file")`). Only the toolbox
    /// changes; the conversation context is untouched, so the next step sees
    /// the new tool set with history intact.
    pub fn tools_mut(&mut self) -> &mut ToolBox {
        &mut self.tools
    }

    pub fn approx_context_tokens(&self, model: &ModelConfig) -> Option<usize> {
        model.count_tokens(&self.render_context())
    }

    pub async fn render_memory(&self) -> Option<String> {
        let memory = self.memory.as_ref()?;
        let guard = memory.context.memory.read().await;
        Some(render_memory_snapshot(&guard))
    }

    pub fn last_step(&self) -> &Option<StepResult> {
        &self.last_step
    }

    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    pub fn system_prompt(&self) -> String {
        self.system_prompt.clone()
    }

    pub fn push_user_message(&mut self, user_prompt: String) {
        self.context.push(Message::user(user_prompt));
    }

    pub async fn step(
        &mut self,
        llm: &LLM,
        debug_prefix: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<StepResult, LLMYError> {
        let config = self.config.from_model(&llm.model);

        let current_context = self.context.clone();
        let conversation = self.conversation_context();
        let cache_key = self.cache_key.as_deref();
        let settings = settings.unwrap_or_else(|| llm.default_settings.clone());
        let timeout = settings.timeout();
        let retry = settings.llm_retry;

        // A turn discarded by a tool rejection is retried from an unchanged
        // context: same conversation, same request — nothing about the
        // rejected attempt (assistant turn, tool outputs, rejection reason)
        // leaks into what the model sees next.
        let mut reject_attempts: u64 = 0;
        let (step_result, extra_messages, assistant_message) = loop {
            let tools = (self.tools.len() != 0).then(|| self.tools.openai_objects());
            // The context is protocol-neutral; the client builds the backend's
            // native request from it directly.
            let mut req =
                llm.build_conversation_request(&conversation, cache_key, &settings, tools)?;
            if let LLMRequest::Chat(chat) = &mut req {
                self.apply_cache_options(llm, chat);
            }
            let (mut resp, assistant_native) = llm
                .complete_request_message_once_with_retry(
                    req,
                    debug_prefix,
                    Some(timeout),
                    Some(retry),
                )
                .await?;

            if resp.choices.is_empty() {
                return Err(LLMYError::EmptyChoice);
            }

            if resp.choices.len() != 1 {
                tracing::warn!(
                    "We expect exactly one choice per call but get {} choices",
                    resp.choices.len()
                );
            }

            let choice: ChatChoice = resp.choices.pop().unwrap();

            if let Some(refused) = choice.message.refusal.clone() {
                return Err(LLMYError::Filtered(refused));
            }

            let reason = choice
                .finish_reason
                .ok_or_else(|| eyre!("no finish reason?!"))?;

            let mut assistant_content = choice.message.content.clone();
            let mut content_override: Option<String> = None;

            match reason {
                FinishReason::ToolCalls | FinishReason::FunctionCall => {
                    let calls = chat_choice_to_toolcalls(&choice);
                    let mut out = vec![];
                    if calls.is_empty() {
                        if config.allow_empty_tool_calls {
                            tracing::warn!("no tool calls but give tool call reason");
                            let nudge = "Your previous tool call format is incorrect, please stick to json format and retry. Nothing is executed for your tool call request.".to_string();
                            assistant_content = Some(nudge.clone());
                            content_override = Some(nudge);
                        } else {
                            return Err(eyre!("no tool calls but give tool call reason").into());
                        }
                    } else {
                        // Phase one: gate the whole batch before anything
                        // runs, so a rejection here costs zero side effects.
                        if let Err(rejected) = self.tools.validate_calls(&calls).await {
                            reject_attempts += 1;
                            tracing::warn!(
                                "tool call rejected in validation ({}/{}), discarding the turn: {}",
                                reject_attempts,
                                config.tool_reject_retries,
                                rejected
                            );
                            if reject_attempts > config.tool_reject_retries {
                                return Err(rejected);
                            }
                            continue;
                        }

                        // Phase two: execute. A rejection surfacing here
                        // means execution already started (side effects may
                        // exist), so re-asking would make the model repeat
                        // them — it degrades to a soft tool result instead;
                        // only the validate phase may discard the turn.
                        let calls = self.invoke_tool_calls(calls, &config).await;
                        for (call, tool_out) in calls.into_iter() {
                            if let Some(tool_out) = tool_out {
                                match tool_out {
                                    Ok(tool_out) => {
                                        out.push(Message::from_chat_request(&tool_out));
                                    }
                                    Err(LLMYError::IncorrectToolCall(_, _, e)) => {
                                        tracing::warn!(
                                            "Incorrect tool call detected for {}, schema is {:?}, we will ask LLM to retry.",
                                            call,
                                            &e
                                        );
                                        out.push(Message::tool_result(
                                            call.tool_id.clone(),
                                            format!(
                                                "Tool call to {} does not conform to schema {:?}",
                                                call, e
                                            ),
                                        ));
                                    }
                                    Err(LLMYError::ToolCallRejected(_, reject_reason)) => {
                                        tracing::warn!(
                                            "Tool call {} rejected mid-run (reason: {}); returning a soft rejection result.",
                                            call,
                                            reject_reason
                                        );
                                        out.push(Message::tool_result(
                                            call.tool_id.clone(),
                                            format!(
                                                "Tool call to {} was rejected by the tool.",
                                                call
                                            ),
                                        ));
                                    }
                                    Err(e) => return Err(e),
                                }
                            } else {
                                tracing::warn!("Tool call {} is not defined", call);
                                out.push(Message::tool_result(
                                    call.tool_id.clone(),
                                    format!("The tool of {} is not defined", call),
                                ));
                            }
                        }
                    }

                    // The protocol-faithful assistant turn goes into the
                    // context — typed parts (signed thinking, encrypted
                    // reasoning, tool-call extras) included; only the
                    // empty-tool-call retry nudge replaces it with synthetic
                    // content.
                    let assistant_message = match content_override {
                        Some(content) => Message::assistant(content),
                        None => assistant_native,
                    };
                    break (
                        StepResult::Toolcalled(assistant_content.clone()),
                        out,
                        assistant_message,
                    );
                }
                FinishReason::ContentFilter => {
                    tracing::warn!("Our response is filtered?! {:?}", &resp);
                    return Err(LLMYError::Filtered(
                        choice.message.content.clone().unwrap_or_default(),
                    ));
                }
                FinishReason::Stop => {
                    break (
                        StepResult::Stop(choice.message.content.clone().unwrap_or_default()),
                        vec![],
                        assistant_native,
                    );
                }
                FinishReason::Length => return Err(LLMYError::OutputLength),
            }
        };

        self.checkpoints
            .push((self.last_step().clone(), current_context));
        self.context.push(assistant_message);
        self.context.extend(extra_messages);
        self.last_step = Some(step_result.clone());
        Ok(step_result)
    }

    async fn invoke_tool_calls(
        &self,
        calls: Vec<GeneralToolCall>,
        config: &AgentConfig,
    ) -> Vec<(
        GeneralToolCall,
        Option<Result<ChatCompletionRequestMessageRaw, LLMYError>>,
    )> {
        if config.sequential_tool_call {
            self.tools.agent_invoke_many_sequential(calls).await
        } else {
            self.tools.agent_invoke_many(calls).await
        }
    }

    pub async fn step_with_user(
        &mut self,
        user_prompt: String,
        llm: &LLM,
        debug_prefix: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<StepResult, LLMYError> {
        self.push_user_message(user_prompt);
        self.step(llm, debug_prefix, settings).await
    }

    pub async fn loop_step_user(
        &mut self,
        user_prompt: String,
        llm: &LLM,
        debug_prefix: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<StepResult, LLMYError> {
        let mut step_result = self
            .step_with_user(user_prompt, llm, debug_prefix, settings.clone())
            .await?;

        while step_result.did_tool_call() {
            step_result = self.step(llm, debug_prefix, settings.clone()).await?;
        }

        Ok(step_result)
    }

    pub async fn revert_step(&mut self) -> Result<(), LLMYError> {
        let (previous_last_step, previous_context) = self
            .checkpoints
            .pop()
            .ok_or_else(|| eyre!("no checkpoints to revert"))?;

        self.last_step = previous_last_step;
        self.context = previous_context;
        Ok(())
    }

    pub async fn compact(
        &self,
        llm: &LLM,
        debug_prefix: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<Self, LLMYError> {
        if self.context.is_empty() {
            return Ok(self.fresh_agent(None).await);
        }

        let memory_enabled = self.memory.is_some();
        let history_text = self.compact_history_text();
        let compact_system_prompt = render_compact_system_prompt(memory_enabled);
        let compact_user_prompt = render_compact_user_prompt(&history_text);
        let compact_debug_prefix = debug_prefix.map(|prefix| format!("{prefix}-compact"));
        let compact_settings = compact_settings(settings, memory_enabled);
        // Compaction is a different prompt shape, so it gets its own key when
        // one was pinned at all.
        let compact_cache_key = self.cache_key.as_ref().map(|key| format!("{key}-compact"));

        let mut compact_agent = match &self.memory {
            Some(memory) => {
                Self::with_memory_config(
                    compact_system_prompt,
                    ToolBox::new(),
                    compact_cache_key,
                    &memory.context,
                    &memory.criteria,
                    self.config.clone(),
                )
                .await
            }
            None => Self::new_with_config(
                compact_system_prompt,
                ToolBox::new(),
                compact_cache_key,
                self.config.clone(),
            ),
        };

        let step_result = compact_agent
            .loop_step_user(
                compact_user_prompt,
                llm,
                compact_debug_prefix.as_deref(),
                compact_settings.clone(),
            )
            .await?;

        if memory_enabled && !compact_agent.did_write_memory() {
            return Err(eyre!("compaction finished without writing or updating memory").into());
        }

        let summary = match step_result {
            StepResult::Stop(summary) => normalize_compact_summary(&summary),
            StepResult::Toolcalled(_) => unreachable!("tool-call loop should exit only after stop"),
        };

        if summary.is_empty() {
            return Err(eyre!("compaction produced an empty summary").into());
        }

        Ok(self.fresh_agent(Some(summary)).await)
    }

    async fn fresh_agent(&self, compact_summary: Option<String>) -> Self {
        let mut agent = self.clone();
        agent.context.clear();
        agent.checkpoints.clear();
        agent.last_step = None;

        if let Some(memory) = &agent.memory {
            let guard = memory.context.memory.read().await;
            agent.system_prompt = memory
                .criteria
                .render_system_prompt(&agent.base_system_prompt, &guard);
        }

        if let Some(summary) = compact_summary {
            agent.push_user_message(render_compacted_context_message(&summary));
        }

        agent
    }

    fn compact_history_text(&self) -> String {
        self.render_context()
    }

    fn did_write_memory(&self) -> bool {
        self.context.iter().any(|message| {
            message.parts.iter().any(|part| {
                matches!(part, MessagePart::ToolCall { name, .. } if is_memory_write_tool(name))
            })
        })
    }
}

fn is_memory_write_tool(tool_name: &str) -> bool {
    tool_name == <WriteMemoryTool as Tool>::NAME || tool_name == <UpdateMemoryTool as Tool>::NAME
}

fn normalize_compact_summary(summary: &str) -> String {
    summary.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn compact_settings(settings: Option<LLMSettings>, memory_enabled: bool) -> Option<LLMSettings> {
    settings.map(|mut settings| {
        settings.llm_tool_choice = Some(if memory_enabled {
            "auto".parse().unwrap()
        } else {
            "none".parse().unwrap()
        });
        settings
    })
}

fn render_memory_snapshot(memory: &AgentMemory) -> String {
    [
        render_memory_section("Long-term memory entries", memory.long_term.values()),
        render_memory_section("Short-term memory entries", memory.short_term.values()),
    ]
    .join("\n\n")
}

fn render_memory_section<'a>(
    title: &str,
    memories: impl IntoIterator<Item = &'a AgentMemoryContent>,
) -> String {
    let rendered = memories
        .into_iter()
        .map(render_memory_entry)
        .collect::<Vec<_>>();

    if rendered.is_empty() {
        format!("{title}:\n<none/>")
    } else {
        format!("{title}:\n\n{}", rendered.join("\n\n---\n\n"))
    }
}

fn render_memory_entry(memory: &AgentMemoryContent) -> String {
    let mut rendered = memory.render_full();

    if let Some(raw_content) = memory.raw_content.as_deref() {
        rendered.push_str("\nraw_content:\n");
        rendered.push_str(raw_content);
    }

    rendered
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::Mutex;

    #[derive(Debug, Clone)]
    struct ZebraTool;

    #[derive(Debug, Clone)]
    struct AlphaTool;

    #[derive(Debug, Clone)]
    struct SlowRecordingTool {
        events: Arc<Mutex<Vec<String>>>,
    }

    #[derive(Debug, Clone)]
    struct FastRecordingTool {
        events: Arc<Mutex<Vec<String>>>,
    }

    impl Tool for ZebraTool {
        type ARGUMENTS = ();
        const NAME: &str = "zebra_tool";
        const DESCRIPTION: Option<&str> = Some("test tool");

        async fn invoke(&self, _arguments: Self::ARGUMENTS) -> Result<String, LLMYError> {
            Ok("ok".to_string())
        }
    }

    impl Tool for AlphaTool {
        type ARGUMENTS = ();
        const NAME: &str = "alpha_tool";
        const DESCRIPTION: Option<&str> = Some("test tool");

        async fn invoke(&self, _arguments: Self::ARGUMENTS) -> Result<String, LLMYError> {
            Ok("ok".to_string())
        }
    }

    impl Tool for SlowRecordingTool {
        type ARGUMENTS = ();
        const NAME: &str = "slow_recording_tool";
        const DESCRIPTION: Option<&str> = Some("test tool");

        async fn invoke(&self, _arguments: Self::ARGUMENTS) -> Result<String, LLMYError> {
            self.events.lock().await.push("slow_start".to_string());
            tokio::time::sleep(Duration::from_millis(20)).await;
            self.events.lock().await.push("slow_end".to_string());
            Ok("slow".to_string())
        }
    }

    impl Tool for FastRecordingTool {
        type ARGUMENTS = ();
        const NAME: &str = "fast_recording_tool";
        const DESCRIPTION: Option<&str> = Some("test tool");

        async fn invoke(&self, _arguments: Self::ARGUMENTS) -> Result<String, LLMYError> {
            self.events.lock().await.push("fast".to_string());
            Ok("fast".to_string())
        }
    }

    fn tool_call(tool_name: &str, tool_id: &str) -> GeneralToolCall {
        GeneralToolCall {
            tool_id: tool_id.to_string(),
            tool_name: tool_name.to_string(),
            tool_args: "null".to_string(),
        }
    }

    /// A tool whose `validate` rejects a marker payload; `invoke` records an
    /// event so tests can prove a rejection cost zero executions.
    #[derive(Debug, Clone)]
    struct PickyTool {
        events: Arc<Mutex<Vec<String>>>,
    }

    impl Tool for PickyTool {
        type ARGUMENTS = String;
        const NAME: &str = "picky_tool";
        const DESCRIPTION: Option<&str> = Some("test tool");

        async fn invoke(&self, _arguments: Self::ARGUMENTS) -> Result<String, LLMYError> {
            self.events.lock().await.push("picky_ran".to_string());
            Ok("ok".to_string())
        }

        async fn validate(&self, arguments: Self::ARGUMENTS) -> Result<(), String> {
            if arguments == "bad" {
                return Err("the payload is marked bad".to_string());
            }
            Ok(())
        }
    }

    /// A tool that only discovers its rejection during execution.
    #[derive(Debug, Clone)]
    struct MidRunRejectTool;

    impl Tool for MidRunRejectTool {
        type ARGUMENTS = ();
        const NAME: &str = "mid_run_reject_tool";
        const DESCRIPTION: Option<&str> = Some("test tool");

        async fn invoke(&self, _arguments: Self::ARGUMENTS) -> Result<String, LLMYError> {
            Err(LLMYError::tool_call_rejected(
                Self::NAME,
                "null",
                "refused during execution",
            ))
        }
    }

    #[tokio::test]
    async fn a_validation_rejection_gates_the_whole_batch_before_any_execution() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut tools = ToolBox::new();
        tools.add_tool(PickyTool {
            events: events.clone(),
        });
        tools.add_tool(FastRecordingTool {
            events: events.clone(),
        });

        let mut bad = tool_call("picky_tool", "id-1");
        bad.tool_args = "\"bad\"".to_string();
        let calls = vec![bad, tool_call("fast_recording_tool", "id-2")];

        let rejected = tools.validate_calls(&calls).await.unwrap_err();
        match rejected {
            LLMYError::ToolCallRejected(call, reject_reason) => {
                // Bound to the real wire call, reason preserved.
                assert_eq!(call.tool_id, "id-1");
                assert_eq!(call.tool_name, "picky_tool");
                assert_eq!(reject_reason, "the payload is marked bad");
            }
            other => panic!("expected a rejection, got {other:?}"),
        }
        // Phase one ran no tool at all.
        assert!(events.lock().await.is_empty());
    }

    #[tokio::test]
    async fn schema_mismatch_and_unknown_tools_are_not_rejections() {
        let mut tools = ToolBox::new();
        tools.add_tool(ZebraTool);

        // Unparseable args keep their soft `IncorrectToolCall` path...
        let mut garbled = tool_call("zebra_tool", "id-1");
        garbled.tool_args = "not json at all".to_string();
        // ...valid JSON of the wrong shape does too...
        let mut wrong_shape = tool_call("zebra_tool", "id-2");
        wrong_shape.tool_args = "123".to_string();
        // ...and unknown tools keep their "not defined" tool_result path.
        let unknown = tool_call("no_such_tool", "id-3");

        tools
            .validate_calls(&[garbled, wrong_shape, unknown])
            .await
            .expect("none of these is a rejection");
    }

    #[tokio::test]
    async fn a_mid_run_rejection_is_soft_and_does_not_stop_the_batch() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut tools = ToolBox::new();
        tools.add_tool(MidRunRejectTool);
        tools.add_tool(FastRecordingTool {
            events: events.clone(),
        });
        let config = AgentConfig {
            sequential_tool_call: true,
            allow_empty_tool_calls: false,
            tool_reject_retries: 3,
        };
        let agent = Agent::new_with_config(
            "base system prompt".to_string(),
            tools,
            Some("cache".to_string()),
            config.clone(),
        );

        let results = agent
            .invoke_tool_calls(
                vec![
                    tool_call("mid_run_reject_tool", "id-1"),
                    tool_call("fast_recording_tool", "id-2"),
                ],
                &config,
            )
            .await;

        // Execution already started, so the rejection is just one call's
        // result; the rest of the batch still runs.
        assert_eq!(results.len(), 2);
        assert!(matches!(
            results[0].1,
            Some(Err(LLMYError::ToolCallRejected(..)))
        ));
        assert_eq!(events.lock().await.clone(), vec!["fast".to_string()]);
    }

    #[test]
    fn tool_reject_retries_defaults_to_thirty_two() {
        assert_eq!(AgentConfig::default().tool_reject_retries, 32);
    }

    #[test]
    fn agent_config_defaults_to_parallel_tool_calls() {
        assert!(!AgentConfig::default().sequential_tool_call);

        let agent = Agent::new(
            "base system prompt".to_string(),
            ToolBox::new(),
            Some("cache".to_string()),
        );

        assert_eq!(agent.config(), &AgentConfig::default());
    }

    #[tokio::test]
    async fn configured_agent_invokes_tool_calls_sequentially() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut tools = ToolBox::new();
        tools.add_tool(SlowRecordingTool {
            events: events.clone(),
        });
        tools.add_tool(FastRecordingTool {
            events: events.clone(),
        });
        let config = AgentConfig {
            sequential_tool_call: true,
            allow_empty_tool_calls: false,
            tool_reject_retries: 3,
        };
        let agent = Agent::new_with_config(
            "base system prompt".to_string(),
            tools,
            Some("cache".to_string()),
            config.clone(),
        );

        let results = agent
            .invoke_tool_calls(
                vec![
                    tool_call("slow_recording_tool", "slow"),
                    tool_call("fast_recording_tool", "fast"),
                ],
                &config,
            )
            .await;

        assert_eq!(results.len(), 2);
        assert_eq!(
            events.lock().await.clone(),
            vec![
                "slow_start".to_string(),
                "slow_end".to_string(),
                "fast".to_string()
            ]
        );
    }

    #[test]
    fn compact_history_text_includes_system_and_user_messages() {
        let mut agent = Agent::new(
            "base system prompt".to_string(),
            ToolBox::new(),
            Some("cache".to_string()),
        );
        agent.push_user_message("implement compaction".to_string());

        let rendered = agent.render_context();

        assert!(rendered.contains("<SYSTEM>\nbase system prompt\n</SYSTEM>"));
        assert!(rendered.contains("<USER>\nimplement compaction\n</USER>"));
    }

    #[test]
    fn render_tools_lists_sorted_tool_names() {
        let mut tools = ToolBox::new();
        tools.add_tool(ZebraTool);
        tools.add_tool(AlphaTool);

        let agent = Agent::new(
            "base system prompt".to_string(),
            tools,
            Some("cache".to_string()),
        );

        assert_eq!(
            agent.render_tools(false),
            "Enabled tools (2):\n- alpha_tool\n- zebra_tool"
        );
    }

    #[test]
    fn render_tools_with_details_includes_description() {
        let mut tools = ToolBox::new();
        tools.add_tool(ZebraTool);
        tools.add_tool(AlphaTool);

        let agent = Agent::new(
            "base system prompt".to_string(),
            tools,
            Some("cache".to_string()),
        );

        assert_eq!(
            agent.render_tools(true),
            "Enabled tools (2):\n- `alpha_tool`: \"test tool\"\n- `zebra_tool`: \"test tool\""
        );
    }

    #[test]
    fn render_tools_reports_when_no_tools_are_enabled() {
        let agent = Agent::new(
            "base system prompt".to_string(),
            ToolBox::new(),
            Some("cache".to_string()),
        );

        assert_eq!(
            agent.render_tools(false),
            "No tools are enabled for this chat."
        );
    }

    #[test]
    fn tools_mut_changes_toolbox_without_touching_context() {
        let mut tools = ToolBox::new();
        tools.add_tool(ZebraTool);
        tools.add_tool(AlphaTool);
        let mut agent = Agent::new(
            "base system prompt".to_string(),
            tools,
            Some("cache".to_string()),
        );
        agent.push_user_message("use the tools".to_string());

        let context_before = agent.render_context();

        assert!(agent.tools_mut().remove_tool("zebra_tool"));
        assert!(!agent.tools_mut().remove_tool("zebra_tool"));

        assert_eq!(agent.render_context(), context_before);
        assert_eq!(
            agent.render_tools(false),
            "Enabled tools (1):\n- alpha_tool"
        );
    }

    #[test]
    fn approx_context_tokens_uses_model_tokenizer() {
        let mut agent = Agent::new(
            "base system prompt".to_string(),
            ToolBox::new(),
            Some("cache".to_string()),
        );
        agent.push_user_message("implement compaction".to_string());

        let model =
            llmy_client::model::OpenAIModel::from_str("o1").expect("failed to load built-in model");
        let token_count = agent
            .approx_context_tokens(&model.config)
            .expect("expected tokenizer-backed token count");

        assert!(token_count > 0);
    }

    #[tokio::test]
    async fn render_memory_returns_none_without_shared_memory() {
        let agent = Agent::new(
            "base system prompt".to_string(),
            ToolBox::new(),
            Some("cache".to_string()),
        );

        assert_eq!(agent.render_memory().await, None);
    }

    #[test]
    fn render_memory_snapshot_includes_long_short_and_raw_content() {
        let mut memory = AgentMemory::default();
        memory.long_term.insert(
            "long".to_string(),
            AgentMemoryContent {
                title: "long".to_string(),
                related_context: "repo".to_string(),
                trigger_scenario: "planning".to_string(),
                content: "long-term detail".to_string(),
                raw_content: Some("full long-term transcript".to_string()),
            },
        );
        memory.short_term.insert(
            "short".to_string(),
            AgentMemoryContent {
                title: "short".to_string(),
                related_context: "task".to_string(),
                trigger_scenario: "active work".to_string(),
                content: "short-term detail".to_string(),
                raw_content: None,
            },
        );

        let rendered = render_memory_snapshot(&memory);

        assert!(rendered.contains("Long-term memory entries:"));
        assert!(rendered.contains("Short-term memory entries:"));
        assert!(rendered.contains("content:\nlong-term detail"));
        assert!(rendered.contains("content:\nshort-term detail"));
        assert!(rendered.contains("raw_content:\nfull long-term transcript"));
    }

    #[test]
    fn normalize_compact_summary_flattens_whitespace() {
        let normalized =
            normalize_compact_summary("current task\n\nfix compaction   path\tand preserve memory");

        assert_eq!(
            normalized,
            "current task fix compaction path and preserve memory"
        );
    }

    #[tokio::test]
    async fn fresh_agent_replaces_history_with_compacted_context_message() {
        let mut agent = Agent::new(
            "base system prompt".to_string(),
            ToolBox::new(),
            Some("cache".to_string()),
        );
        agent.push_user_message("first message".to_string());

        let compacted = agent
            .fresh_agent(Some("single paragraph summary".to_string()))
            .await;

        assert_eq!(compacted.context.len(), 1);
        assert_eq!(
            compacted.context[0].role,
            llmy_client::client::MessageRole::User
        );
        assert_eq!(
            compacted.context[0].text(),
            "Compacted context: single paragraph summary"
        );
        assert!(compacted.last_step.is_none());
        assert!(compacted.checkpoints.is_empty());
    }

    #[cfg(feature = "memory-embed-search")]
    #[tokio::test]
    async fn with_memory_snapshots_prompt_until_fresh_agent() {
        use llmy_agent_tools::memory::{
            AgentMemory, AgentMemoryContent,
            embed::{SimilarityModel, SimilarityModelConfig},
        };

        let cache_dir = tempfile::tempdir().unwrap();
        let memory = AgentMemoryContext::new(
            AgentMemory::default(),
            SimilarityModel::new(SimilarityModelConfig {
                cache_dir: Some(cache_dir.path().to_path_buf()),
                ..Default::default()
            })
            .await
            .unwrap(),
        );
        {
            let mut guard = memory.memory.write().await;
            guard.long_term.insert(
                "before compact".to_string(),
                AgentMemoryContent {
                    title: "before compact".to_string(),
                    related_context: "initial".to_string(),
                    trigger_scenario: "bootstrap".to_string(),
                    content: "existing memory".to_string(),
                    raw_content: None,
                },
            );
        }

        let criteria = AgentMemorySystemPromptCriteria::default();
        let agent = Agent::with_memory(
            "base system prompt".to_string(),
            ToolBox::new(),
            Some("cache".to_string()),
            &memory,
            &criteria,
        )
        .await;

        {
            let mut guard = memory.memory.write().await;
            guard.long_term.insert(
                "after snapshot".to_string(),
                AgentMemoryContent {
                    title: "after snapshot".to_string(),
                    related_context: "later".to_string(),
                    trigger_scenario: "after write".to_string(),
                    content: "new memory".to_string(),
                    raw_content: None,
                },
            );
        }

        let conversation = agent.conversation_context();
        let compacted = agent.fresh_agent(None).await;
        let refreshed_conversation = compacted.conversation_context();

        let snapshot_prompt = conversation[0].text();
        let refreshed_prompt = refreshed_conversation[0].text();

        assert!(snapshot_prompt.contains("title: before compact"));
        assert!(!snapshot_prompt.contains("title: after snapshot"));
        assert!(refreshed_prompt.contains("title: after snapshot"));
    }

    // --- prompt cache breakpoints ------------------------------------------

    fn cache_test_settings() -> LLMSettings {
        LLMSettings {
            llm_temperature: None,
            llm_presence_penalty: None,
            llm_prompt_timeout: 1,
            llm_retry: 0,
            llm_max_completion_tokens: None,
            llm_tool_choice: None,
            llm_stream: false,
            top_p: None,
            reasoning_effort: None,
            auto_strip: false,
            auto_cache_key: true,
            cache_key_ttl: llmy_client::cache_key::DEFAULT_TTL_SECS,
            cache_key_rpm: llmy_client::cache_key::DEFAULT_MAX_RPM,
            billing_log_tokens: 100_000,
            token_estimate_pct: 10.0,
            allow_implicit_convert: false,
            llm_concurrent: 0,
        }
    }

    /// An LLM pointed at a dead address — these tests never issue a request.
    fn cache_test_llm(model: &str) -> LLM {
        LLM::new(
            llmy_client::client::SupportedConfig::new("http://127.0.0.1:1", "key"),
            OpenAIModel::from_str(model).expect("built-in model"),
            llmy_client::rust_decimal::Decimal::ONE,
            cache_test_settings(),
            None,
        )
    }

    fn cache_test_agent() -> Agent {
        let mut agent = Agent::new("system".to_string(), ToolBox::new(), Some("k".to_string()));
        agent.push_user_message("first".to_string());
        agent.push_user_message("second".to_string());
        agent
    }

    fn breakpoint_marked(msg: &Message) -> bool {
        msg.cache_breakpoint
    }

    #[test]
    fn context_mut_marks_a_breakpoint_on_one_message() {
        let mut agent = cache_test_agent();
        agent.context_mut()[0].breakpoint();

        let ctx = agent.conversation_context();
        // System prompt is synthesized and unmarked; only message 0 is marked.
        assert!(!breakpoint_marked(&ctx[0]));
        assert!(breakpoint_marked(&ctx[1]));
        assert!(!breakpoint_marked(&ctx[2]));

        // The marker lowers into the chat wire form too.
        let chat = Message::many_to_chat(&ctx);
        assert!(
            serde_json::to_string(&chat[1])
                .unwrap()
                .contains("prompt_cache_breakpoint")
        );
    }

    #[test]
    fn toggle_system_breakpoint_marks_the_synthesized_system_message() {
        let mut agent = cache_test_agent();
        assert!(!agent.system_breakpoint());
        agent.toggle_system_breakpoint(true);
        assert!(agent.system_breakpoint());

        let ctx = agent.conversation_context();
        // Only the system message — the conversation is untouched.
        assert!(breakpoint_marked(&ctx[0]));
        assert!(!breakpoint_marked(&ctx[1]));
        assert!(!breakpoint_marked(&ctx[2]));

        agent.toggle_system_breakpoint(false);
        assert!(!agent.conversation_context().iter().any(breakpoint_marked));
    }

    #[test]
    fn clear_breakpoints_removes_every_marker() {
        let mut agent = cache_test_agent();
        agent.toggle_system_breakpoint(true);
        agent.context_mut()[0].breakpoint();
        agent.context_mut()[1].breakpoint();

        agent.clear_breakpoints();
        assert!(!agent.system_breakpoint());
        assert!(!agent.conversation_context().iter().any(breakpoint_marked));
        // Idempotent.
        agent.clear_breakpoints();
        assert!(!agent.conversation_context().iter().any(breakpoint_marked));
    }

    #[test]
    fn cache_options_are_absent_until_toggled() {
        let agent = cache_test_agent();
        let llm = cache_test_llm("openai/gpt-5.6-sol");
        assert_eq!(agent.breakpoint_mode(), None);

        let mut req = llm
            .build_chat_request(
                Message::many_to_chat(&agent.conversation_context()),
                None,
                &cache_test_settings(),
                None,
            )
            .unwrap();
        agent.apply_cache_options(&llm, &mut req);
        assert!(req.prompt_cache_options.is_none());
    }

    #[test]
    fn toggle_breakpoint_implicit_sets_the_request_mode() {
        let llm = cache_test_llm("openai/gpt-5.6-sol");
        let settings = cache_test_settings();

        for (explicit, expected) in [
            (true, PromptCacheMode::Explicit),
            (false, PromptCacheMode::Implicit),
        ] {
            let mut agent = cache_test_agent();
            agent.toggle_breakpoint_implicit(explicit);
            assert_eq!(agent.breakpoint_mode(), Some(expected));

            let mut req = llm
                .build_chat_request(
                    Message::many_to_chat(&agent.conversation_context()),
                    None,
                    &settings,
                    None,
                )
                .unwrap();
            agent.apply_cache_options(&llm, &mut req);
            assert_eq!(
                req.prompt_cache_options.as_ref().map(|o| o.inner.mode),
                Some(Some(expected))
            );
        }
    }

    #[test]
    fn cache_options_are_dropped_for_prefix_cached_models() {
        // gpt-5.5 caches by matching prefix; `prompt_cache_options` would be an
        // unknown request field there.
        let llm = cache_test_llm("openai/gpt-5.5");
        let mut agent = cache_test_agent();
        agent.toggle_breakpoint_implicit(true);

        let mut req = llm
            .build_chat_request(
                Message::many_to_chat(&agent.conversation_context()),
                None,
                &cache_test_settings(),
                None,
            )
            .unwrap();
        agent.apply_cache_options(&llm, &mut req);
        assert!(req.prompt_cache_options.is_none());
        // The agent still remembers what was asked for.
        assert_eq!(agent.breakpoint_mode(), Some(PromptCacheMode::Explicit));
    }
}
