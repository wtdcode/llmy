use async_openai::types::chat::{
    ChatCompletionMessageToolCalls, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent, FinishReason,
};
use color_eyre::eyre::eyre;
use llmy_agent::{LLMYError, StepResult, Tool, tool::ToolBox};
use llmy_agent_tools::memory::{AgentMemoryContext, UpdateMemoryTool, WriteMemoryTool};
use llmy_client::debug::completion_to_string;
use llmy_client::{client::LLM, settings::LLMSettings};

use crate::{
    memory::AgentMemorySystemPromptCriteria,
    prompt::{
        render_compact_system_prompt, render_compact_user_prompt, render_compacted_context_message,
    },
    utils::{chat_choice_to_assistant, chat_choice_to_toolcalls},
};

#[derive(Clone)]
struct AgentMemoryRuntime {
    context: AgentMemoryContext,
    criteria: AgentMemorySystemPromptCriteria,
}

/// Agent implementation backed by an in-memory conversation context and toolbox.
#[derive(Clone)]
pub struct Agent {
    base_system_prompt: String,
    system_prompt: String,
    tools: ToolBox,
    context: Vec<ChatCompletionRequestMessage>,
    checkpoints: Vec<(Option<StepResult>, Vec<ChatCompletionRequestMessage>)>,
    last_step: Option<StepResult>,
    cache_key: String,
    memory: Option<AgentMemoryRuntime>,
}

impl Agent {
    pub fn new(system_prompt: String, tools: ToolBox, cache_key: String) -> Self {
        Self {
            base_system_prompt: system_prompt.clone(),
            system_prompt,
            tools,
            checkpoints: vec![],
            context: vec![],
            last_step: None,
            cache_key,
            memory: None,
        }
    }

    pub async fn with_memory(
        system_prompt: String,
        mut tools: ToolBox,
        cache_key: String,
        memory: &AgentMemoryContext,
        criteria: &AgentMemorySystemPromptCriteria,
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
        }
    }

    fn system_message(system_prompt: String) -> ChatCompletionRequestMessage {
        ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
            content: ChatCompletionRequestSystemMessageContent::Text(system_prompt),
            name: None,
        })
    }

    pub fn conversation_context(&self) -> Vec<ChatCompletionRequestMessage> {
        std::iter::once(Self::system_message(self.system_prompt.clone()))
            .chain(self.context.clone())
            .collect()
    }

    pub fn last_step(&self) -> &Option<StepResult> {
        &self.last_step
    }

    pub fn system_prompt(&self) -> String {
        self.system_prompt.clone()
    }

    pub fn push_user_message(&mut self, user_prompt: String) {
        self.context.push(ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(user_prompt),
                name: None,
            },
        ));
    }

    pub async fn step(
        &mut self,
        llm: &LLM,
        debug_prefix: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<StepResult, LLMYError> {
        let current_context = self.context.clone();
        let messages = self.conversation_context();
        let tools = (self.tools.len() != 0).then(|| self.tools.openai_objects());
        let mut resp = llm
            .prompt_messages_once(
                messages,
                debug_prefix,
                (!self.cache_key.is_empty()).then_some(self.cache_key.as_str()),
                settings,
                tools,
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

        let choice = resp.choices.pop().unwrap();

        if let Some(refused) = choice.message.refusal {
            return Err(LLMYError::Filtered(refused));
        }

        let reason = choice
            .finish_reason
            .ok_or_else(|| eyre!("no finish reason?!"))?;

        let assistant = chat_choice_to_assistant(&choice)?;

        let (step_result, extra_messages): (StepResult, Vec<ChatCompletionRequestMessage>) =
            match reason {
                FinishReason::ToolCalls | FinishReason::FunctionCall => {
                    let calls = chat_choice_to_toolcalls(&choice);
                    if calls.is_empty() {
                        return Err(eyre!("no tool calls but give tool call reason").into());
                    }

                    for call in &calls {
                        if !self.tools.has_tool(&call.tool_name) {
                            return Err(LLMYError::NonExistingToolCall(call.clone()));
                        }
                    }

                    let out = self
                        .tools
                        .agent_invoke_many(calls)
                        .await?
                        .into_iter()
                        .map(|v| v.1)
                        .collect();
                    (StepResult::Toolcalled(choice.message.content.clone()), out)
                }
                FinishReason::ContentFilter => {
                    return Err(LLMYError::Filtered(
                        choice.message.content.unwrap_or_default(),
                    ));
                }
                FinishReason::Stop => (
                    StepResult::Stop(choice.message.content.unwrap_or_default()),
                    vec![],
                ),
                FinishReason::Length => return Err(LLMYError::OutputLength),
            };

        self.checkpoints
            .push((self.last_step().clone(), current_context));
        self.context.push(assistant.into());
        self.context.extend(extra_messages);
        self.last_step = Some(step_result.clone());
        Ok(step_result)
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
                Self::with_memory(
                    compact_system_prompt,
                    ToolBox::new(),
                    compact_cache_key,
                    &memory.context,
                    &memory.criteria,
                )
                .await
            }
            None => Self::new(compact_system_prompt, ToolBox::new(), compact_cache_key),
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
        let mut messages = Vec::with_capacity(self.context.len() + 1);
        messages.push(completion_to_string(&Self::system_message(
            self.system_prompt.clone(),
        )));
        messages.extend(self.context.iter().map(completion_to_string));
        messages.join("\n")
    }

    fn did_write_memory(&self) -> bool {
        self.context.iter().any(|message| match message {
            ChatCompletionRequestMessage::Assistant(assistant) => {
                assistant.tool_calls.as_ref().is_some_and(|tool_calls| {
                    tool_calls.iter().any(|tool_call| match tool_call {
                        ChatCompletionMessageToolCalls::Function(function) => {
                            is_memory_write_tool(function.function.name.as_str())
                        }
                        ChatCompletionMessageToolCalls::Custom(custom) => {
                            is_memory_write_tool(custom.custom_tool.name.as_str())
                        }
                    })
                })
            }
            _ => false,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_history_text_includes_system_and_user_messages() {
        let mut agent = Agent::new(
            "base system prompt".to_string(),
            ToolBox::new(),
            "cache".to_string(),
        );
        agent.push_user_message("implement compaction".to_string());

        let rendered = agent.compact_history_text();

        assert!(rendered.contains("<SYSTEM>\nbase system prompt\n</SYSTEM>"));
        assert!(rendered.contains("<USER>\nimplement compaction\n</USER>"));
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
        match &compacted.context[0] {
            ChatCompletionRequestMessage::User(user) => {
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

    #[tokio::test]
    async fn with_memory_snapshots_prompt_until_fresh_agent() {
        use llmy_agent_tools::memory::{
            AgentMemory, AgentMemoryContent,
            embed::{SimilarityModel, SimilarityModelConfig},
        };

        let memory = AgentMemoryContext::new(
            AgentMemory::default(),
            SimilarityModel::new(SimilarityModelConfig::default())
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
