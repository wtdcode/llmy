use async_openai::types::chat::{
    ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessageArgs,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, FinishReason,
};
use color_eyre::eyre::eyre;
use llmy_client::{client::LLM, settings::LLMSettings};
use llmy_types::error::{GeneralToolCall, LLMYError};
use tokio::task::JoinSet;

use crate::{memory::AgentMemory, tool::ToolBox};

#[derive(Debug, Clone)]
pub enum StepResult {
    Stop(String),
    Toolcalled(Option<String>),
}

impl StepResult {
    pub fn assistant_message(&self) -> Option<&String> {
        match self {
            Self::Stop(v) => Some(v),
            Self::Toolcalled(v) => v.as_ref(),
        }
    }

    pub fn did_tool_call(&self) -> bool {
        matches!(self, Self::Toolcalled(_))
    }

    pub fn did_stop(&self) -> bool {
        matches!(self, Self::Stop(_))
    }
}

/// Agent object is simply a collection of agent related resources
pub struct Agent {
    pub system_prompt: String,
    pub memory: AgentMemory,
    pub tools: ToolBox,
    pub context: Vec<ChatCompletionRequestMessage>,
    pub checkpoints: Vec<(Option<StepResult>, Vec<ChatCompletionRequestMessage>)>,
    pub last_step: Option<StepResult>,
    pub cache_key: String,
}

impl Agent {
    pub fn new(system_prompt: String, tools: ToolBox, cache_key: String) -> Self {
        Self {
            system_prompt,
            memory: AgentMemory::default(),
            tools,
            checkpoints: vec![],
            context: vec![],
            last_step: None,
            cache_key,
        }
    }

    pub fn conversation_context(&self) -> Vec<ChatCompletionRequestMessage> {
        std::iter::once(ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessage {
                content: ChatCompletionRequestSystemMessageContent::Text(
                    self.system_prompt.clone(),
                ),
                name: None,
            },
        ))
        .chain(self.context.clone().into_iter())
        .collect()
    }

    pub fn last_step(&self) -> &Option<StepResult> {
        &self.last_step
    }

    pub async fn step_by_user(
        &mut self,
        user_prompt: String,
        llm: &LLM,
        debug_prefix: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<StepResult, LLMYError> {
        self.context.push(ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(user_prompt),
                name: None,
            },
        ));
        self.step(llm, debug_prefix, settings).await
    }

    pub async fn step(
        &mut self,
        llm: &LLM,
        debug_prefix: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<StepResult, LLMYError> {
        let current_context = self.context.clone();
        let messages = self.conversation_context();
        let tools = (!self.tools.tools.is_empty()).then(|| self.tools.openai_objects());
        let mut resp = llm
            .prompt_messages_once(
                messages,
                debug_prefix,
                (!self.cache_key.is_empty()).then_some(self.cache_key.as_str()),
                settings,
                tools,
            )
            .await?;

        if resp.choices.len() == 0 {
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

        // This shall not happen in any case
        let reason = choice
            .finish_reason
            .ok_or_else(|| eyre!("no finish reason?!"))?;

        let mut builder = ChatCompletionRequestAssistantMessageArgs::default();

        if let Some(content) = &choice.message.content {
            builder.content(content.clone());
        }
        if let Some(tool_calls) = &choice.message.tool_calls {
            builder.tool_calls(tool_calls.clone());
        }
        #[allow(deprecated)]
        if let Some(function_call) = &choice.message.function_call {
            builder.function_call(function_call.clone());
        }

        let (step_result, extra_messages): (StepResult, Vec<ChatCompletionRequestMessage>) =
            match reason {
                FinishReason::ToolCalls | FinishReason::FunctionCall => {
                    let mut calls = vec![];

                    for tool in choice.message.tool_calls.iter().flatten() {
                        let (id, tool_name, args) = match tool {
                            ChatCompletionMessageToolCalls::Function(func) => (
                                func.id.clone(),
                                func.function.name.clone(),
                                func.function.arguments.clone(),
                            ),
                            ChatCompletionMessageToolCalls::Custom(custom) => (
                                custom.id.clone(),
                                custom.custom_tool.name.clone(),
                                custom.custom_tool.name.clone(),
                            ),
                        };
                        calls.push(GeneralToolCall {
                            tool_id: id,
                            tool_name,
                            tool_args: args,
                        });
                    }

                    if calls.len() == 0 {
                        return Err(eyre!("no tool calls but give tool call reason").into());
                    }

                    for call in calls.iter() {
                        if !self.tools.has_tool(&call.tool_name) {
                            return Err(LLMYError::NonExistingToolCall(call.clone()));
                        }
                    }

                    let mut js = JoinSet::new();
                    for call in calls {
                        let tb = self.tools.clone();
                        js.spawn(async move {
                            let tc: GeneralToolCall = call.clone();
                            tracing::info!("Calling {}", &tc);
                            (tc, tb.invoke(call.tool_name, call.tool_args).await)
                        });
                    }

                    let results = js.join_all().await;
                    let mut out = vec![];
                    for (call, result) in results {
                        let result = result
                            .unwrap()
                            .map_err(|e| LLMYError::ToolCallError(call.clone(), Box::new(e)))?;
                        out.push(
                            ChatCompletionRequestToolMessage {
                                tool_call_id: call.tool_id,
                                content: ChatCompletionRequestToolMessageContent::Text(result),
                            }
                            .into(),
                        );
                    }
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
        let assistant = builder.build()?;
        self.context.push(assistant.into());
        self.context.extend(extra_messages.into_iter());
        self.last_step = Some(step_result.clone());
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
}
