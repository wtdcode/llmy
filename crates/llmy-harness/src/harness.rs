use color_eyre::eyre::eyre;
use llmy_agent::{LLMYError, StepResult, Tool, tool::ToolBox};
use llmy_agent_tools::memory::{
    AgentMemory, AgentMemoryContent, AgentMemoryContext, UpdateMemoryTool, WriteMemoryTool,
};
use llmy_client::debug::completion_to_string;
use llmy_client::model::OpenAIModel;
use llmy_client::req::{
    ChatCompletionMessageToolCallsRaw, ChatCompletionRequestMessageRaw,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestSystemMessageRaw,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestToolMessageRaw,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageRaw,
};
use llmy_client::resp::{ChatChoice, FinishReason};
use llmy_client::{
    client::{LLM, RawExtensibleChatRequestMessage},
    model::ModelConfig,
    settings::LLMSettings,
};
use llmy_types::error::GeneralToolCall;
use llmy_types::other::WithOtherFields;

use crate::{
    memory::AgentMemorySystemPromptCriteria,
    prompt::{
        render_compact_system_prompt, render_compact_user_prompt, render_compacted_context_message,
    },
    utils::{chat_choice_to_assistant_with_content, chat_choice_to_toolcalls},
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
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            sequential_tool_call: false,
            allow_empty_tool_calls: false,
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

/// Agent implementation backed by an in-memory conversation context and toolbox.
#[derive(Clone)]
pub struct Agent {
    base_system_prompt: String,
    system_prompt: String,
    tools: ToolBox,
    context: Vec<RawExtensibleChatRequestMessage>,
    checkpoints: Vec<(Option<StepResult>, Vec<RawExtensibleChatRequestMessage>)>,
    last_step: Option<StepResult>,
    cache_key: String,
    memory: Option<AgentMemoryRuntime>,
    config: AgentConfig,
}

impl Agent {
    pub fn new(system_prompt: String, tools: ToolBox, cache_key: String) -> Self {
        Self::new_with_config(system_prompt, tools, cache_key, AgentConfig::default())
    }

    pub fn new_with_config(
        system_prompt: String,
        tools: ToolBox,
        cache_key: String,
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
        }
    }

    pub async fn with_memory(
        system_prompt: String,
        tools: ToolBox,
        cache_key: String,
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
        cache_key: String,
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
        }
    }

    fn system_message(system_prompt: String) -> ChatCompletionRequestMessageRaw {
        let raw = ChatCompletionRequestSystemMessageRaw {
            content: ChatCompletionRequestSystemMessageContent::Text(system_prompt),
            name: None,
        };
        ChatCompletionRequestMessageRaw::System(WithOtherFields::new(raw))
    }

    pub fn conversation_context(&self) -> Vec<RawExtensibleChatRequestMessage> {
        std::iter::once(RawExtensibleChatRequestMessage::new(Self::system_message(
            self.system_prompt.clone(),
        )))
        .chain(self.context.clone())
        .collect()
    }

    pub fn render_context(&self) -> String {
        self.conversation_context()
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
        let user = ChatCompletionRequestUserMessageRaw {
            content: ChatCompletionRequestUserMessageContent::Text(user_prompt),
            name: None,
        };
        self.context.push(RawExtensibleChatRequestMessage::new(
            ChatCompletionRequestMessageRaw::User(WithOtherFields::new(user)),
        ));
    }

    pub async fn step(
        &mut self,
        llm: &LLM,
        debug_prefix: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<StepResult, LLMYError> {
        let config = self.config.from_model(&llm.model);

        let current_context = self.context.clone();
        let messages = self.conversation_context();
        let tools = (self.tools.len() != 0).then(|| self.tools.openai_objects());
        let cache_key = (!self.cache_key.is_empty()).then_some(self.cache_key.as_str());
        let settings = settings.unwrap_or_else(|| llm.default_settings.clone());
        let timeout = settings.timeout();
        let retry = settings.llm_retry;
        let req = llm.build_chat_request(messages, cache_key, &settings, tools)?;
        let mut resp = llm
            .complete_extensible_once_with_retry(&req, debug_prefix, Some(timeout), Some(retry))
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
        let propagated_reasoning = propagated_reasoning_content(&choice);

        if let Some(refused) = choice.message.refusal.clone() {
            return Err(LLMYError::Filtered(refused));
        }

        let reason = choice
            .finish_reason
            .ok_or_else(|| eyre!("no finish reason?!"))?;

        let mut assistant_content = choice.message.content.clone();

        let (step_result, extra_messages): (StepResult, Vec<ChatCompletionRequestMessageRaw>) =
            match reason {
                FinishReason::ToolCalls | FinishReason::FunctionCall => {
                    let calls = chat_choice_to_toolcalls(&choice);
                    let mut out = vec![];
                    if calls.is_empty() {
                        if config.allow_empty_tool_calls {
                            tracing::warn!("no tool calls but give tool call reason");
                            assistant_content = Some("Your previous tool call format is incorrect, please stick to json format and retry. Nothing is executed for your tool call request.".to_string());
                        } else {
                            return Err(eyre!("no tool calls but give tool call reason").into());
                        }
                    } else {
                        let calls = self.invoke_tool_calls(calls, &config).await;

                        for (call, tool_out) in calls.into_iter() {
                            if let Some(tool_out) = tool_out {
                                match tool_out {
                                    Ok(tool_out) => {
                                        out.push(tool_out);
                                    }
                                    Err(LLMYError::IncorrectToolCall(_, _, e)) => {
                                        tracing::warn!(
                                            "Incorrect tool call detected for {}, schema is {:?}, we will ask LLM to retry.",
                                            call,
                                            &e
                                        );
                                        let tool_msg = ChatCompletionRequestToolMessageRaw {
                                            content: ChatCompletionRequestToolMessageContent::Text(
                                                format!(
                                                    "Tool call to {} does not conform to schema {:?}",
                                                    call, e
                                                ),
                                            ),
                                            tool_call_id: call.tool_id.clone(),
                                        };
                                        out.push(ChatCompletionRequestMessageRaw::Tool(
                                            WithOtherFields::new(tool_msg),
                                        ));
                                    }
                                    Err(e) => return Err(e),
                                }
                            } else {
                                tracing::warn!("Tool call {} is not defined", call);
                                let tool_msg = ChatCompletionRequestToolMessageRaw {
                                    content: ChatCompletionRequestToolMessageContent::Text(
                                        format!("The tool of {} is not defined", call),
                                    ),
                                    tool_call_id: call.tool_id.clone(),
                                };
                                out.push(ChatCompletionRequestMessageRaw::Tool(
                                    WithOtherFields::new(tool_msg),
                                ));
                            }
                        }
                    }

                    (StepResult::Toolcalled(assistant_content.clone()), out)
                }
                FinishReason::ContentFilter => {
                    tracing::warn!("Our response is filtered?! {:?}", &resp);
                    return Err(LLMYError::Filtered(
                        choice.message.content.clone().unwrap_or_default(),
                    ));
                }
                FinishReason::Stop => (
                    StepResult::Stop(choice.message.content.clone().unwrap_or_default()),
                    vec![],
                ),
                FinishReason::Length => return Err(LLMYError::OutputLength),
            };

        let assistant = chat_choice_to_assistant_with_content(&choice, assistant_content)?;

        self.checkpoints
            .push((self.last_step().clone(), current_context));
        let mut assistant_message = RawExtensibleChatRequestMessage::new(
            ChatCompletionRequestMessageRaw::Assistant(assistant),
        );
        if let Some(reasoning_content) = propagated_reasoning {
            assistant_message.insert_extra_string("reasoning_content", reasoning_content);
        }
        self.context.push(assistant_message);
        self.context.extend(
            extra_messages
                .into_iter()
                .map(RawExtensibleChatRequestMessage::new),
        );
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
        let compact_cache_key = if self.cache_key.is_empty() {
            String::new()
        } else {
            format!("{}-compact", self.cache_key)
        };

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
        self.context.iter().any(|message| match &message.inner {
            ChatCompletionRequestMessageRaw::Assistant(assistant) => {
                assistant.tool_calls.as_ref().is_some_and(|tool_calls| {
                    tool_calls.iter().any(|tool_call| match &tool_call.inner {
                        ChatCompletionMessageToolCallsRaw::Function(function) => {
                            is_memory_write_tool(function.function.name.as_str())
                        }
                        ChatCompletionMessageToolCallsRaw::Custom(custom) => {
                            is_memory_write_tool(custom.custom_tool.name.as_str())
                        }
                    })
                })
            }
            _ => false,
        })
    }
}

fn propagated_reasoning_content(choice: &ChatChoice) -> Option<String> {
    choice
        .inner
        .message
        .other
        .get("reasoning_content")?
        .as_str()
        .filter(|reasoning_content| !reasoning_content.trim().is_empty())
        .map(|reasoning_content| reasoning_content.to_string())
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

    #[test]
    fn agent_config_defaults_to_parallel_tool_calls() {
        assert!(!AgentConfig::default().sequential_tool_call);

        let agent = Agent::new(
            "base system prompt".to_string(),
            ToolBox::new(),
            "cache".to_string(),
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
        };
        let agent = Agent::new_with_config(
            "base system prompt".to_string(),
            tools,
            "cache".to_string(),
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
            "cache".to_string(),
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

        let agent = Agent::new("base system prompt".to_string(), tools, "cache".to_string());

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

        let agent = Agent::new("base system prompt".to_string(), tools, "cache".to_string());

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
            "cache".to_string(),
        );

        assert_eq!(
            agent.render_tools(false),
            "No tools are enabled for this chat."
        );
    }

    #[test]
    fn approx_context_tokens_uses_model_tokenizer() {
        let mut agent = Agent::new(
            "base system prompt".to_string(),
            ToolBox::new(),
            "cache".to_string(),
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
            "cache".to_string(),
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
    fn reasoning_content_extra_carries_reasoning_for_mimo() {
        let choice: llmy_client::client::RawExtensibleChatChoice =
            serde_json::from_value(serde_json::json!({
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "call the tool",
                    "reasoning_content": "I need the tool."
                },
                "finish_reason": "tool_calls"
            }))
            .unwrap();

        assert_eq!(
            propagated_reasoning_content(&choice),
            Some("I need the tool.".to_string())
        );
    }

    #[test]
    fn reasoning_content_extra_skips_blank_reasoning_for_mimo() {
        let model = OpenAIModel::from_str("mimo-v2.5-pro").unwrap();
        let choice: llmy_client::client::RawExtensibleChatChoice =
            serde_json::from_value(serde_json::json!({
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "reasoning_content": "   "
                },
                "finish_reason": "stop"
            }))
            .unwrap();

        assert_eq!(propagated_reasoning_content(&choice), None);
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
            "cache".to_string(),
        );
        agent.push_user_message("first message".to_string());

        let compacted = agent
            .fresh_agent(Some("single paragraph summary".to_string()))
            .await;

        assert_eq!(compacted.context.len(), 1);
        match &compacted.context[0].inner {
            ChatCompletionRequestMessageRaw::User(user) => {
                assert_eq!(
                    user.content,
                    ChatCompletionRequestUserMessageContent::Text(
                        "Compacted context: single paragraph summary".to_string(),
                    )
                );
            }
            other => panic!("expected compacted user message, got {:?}", other),
        }
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
            "cache".to_string(),
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

        let snapshot_prompt = completion_to_string(&conversation[0]);
        let refreshed_prompt = completion_to_string(&refreshed_conversation[0]);

        assert!(snapshot_prompt.contains("title: before compact"));
        assert!(!snapshot_prompt.contains("title: after snapshot"));
        assert!(refreshed_prompt.contains("title: after snapshot"));
    }
}
