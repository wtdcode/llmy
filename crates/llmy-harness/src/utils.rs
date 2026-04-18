use async_openai::types::chat::{
    ChatChoice, ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestAssistantMessageArgs,
};
use llmy_agent::LLMYError;
use llmy_types::error::GeneralToolCall;

pub fn chat_choice_to_assistant(
    choice: &ChatChoice,
) -> Result<ChatCompletionRequestAssistantMessage, LLMYError> {
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
    let assistant = builder.build()?;
    Ok(assistant)
}

pub fn chat_choice_to_toolcalls(choice: &ChatChoice) -> Vec<GeneralToolCall> {
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

    #[allow(deprecated)]
    if let Some(fcall) = &choice.message.function_call {
        calls.push(GeneralToolCall {
            tool_id: "function call".to_string(),
            tool_name: fcall.name.clone(),
            tool_args: fcall.arguments.clone(),
        });
    }

    calls
}
