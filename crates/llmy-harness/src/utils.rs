use async_openai::types::chat::{
    ChatChoice, ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestAssistantMessageArgs,
};
use llmy_agent::LLMYError;
use llmy_types::error::GeneralToolCall;

pub fn chat_choice_to_assistant(
    choice: &ChatChoice,
) -> Result<ChatCompletionRequestAssistantMessage, LLMYError> {
    chat_choice_to_assistant_with_content(choice, choice.message.content.clone())
}

pub fn chat_choice_to_assistant_with_content(
    choice: &ChatChoice,
    content: Option<String>,
) -> Result<ChatCompletionRequestAssistantMessage, LLMYError> {
    let mut builder = ChatCompletionRequestAssistantMessageArgs::default();

    if let Some(content) = content {
        builder.content(content);
    } else if let Some(content) = choice.message.content.as_ref() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::chat::{
        ChatCompletionRequestAssistantMessageContent, ChatCompletionResponseMessage, Role,
    };

    fn choice_with_content(content: &str) -> ChatChoice {
        ChatChoice {
            index: 0,
            message: ChatCompletionResponseMessage {
                content: Some(content.to_string()),
                refusal: None,
                tool_calls: None,
                annotations: None,
                role: Role::Assistant,
                function_call: None,
                audio: None,
            },
            finish_reason: None,
            logprobs: None,
        }
    }

    #[test]
    fn chat_choice_to_assistant_preserves_choice_content() {
        let assistant = chat_choice_to_assistant(&choice_with_content("original")).unwrap();

        match assistant.content.unwrap() {
            ChatCompletionRequestAssistantMessageContent::Text(content) => {
                assert_eq!(content, "original")
            }
            content => panic!("expected text content, got {content:?}"),
        }
    }

    #[test]
    fn chat_choice_to_assistant_accepts_content_override() {
        let assistant = chat_choice_to_assistant_with_content(
            &choice_with_content("malformed tool call"),
            Some("retry with valid json".to_string()),
        )
        .unwrap();

        match assistant.content.unwrap() {
            ChatCompletionRequestAssistantMessageContent::Text(content) => {
                assert_eq!(content, "retry with valid json")
            }
            content => panic!("expected text content, got {content:?}"),
        }
    }
}
