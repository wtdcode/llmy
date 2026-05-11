use llmy_agent::LLMYError;
use llmy_client::req::{
    ChatCompletionMessageToolCallsRaw, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestAssistantMessageRaw,
};
use llmy_client::resp::ChatChoice;
use llmy_types::error::GeneralToolCall;
use llmy_types::other::WithOtherFields;

pub fn chat_choice_to_assistant(
    choice: &ChatChoice,
) -> Result<ChatCompletionRequestAssistantMessage, LLMYError> {
    chat_choice_to_assistant_with_content(choice, choice.message.content.clone())
}

#[allow(deprecated)]
pub fn chat_choice_to_assistant_with_content(
    choice: &ChatChoice,
    content: Option<String>,
) -> Result<ChatCompletionRequestAssistantMessage, LLMYError> {
    let content = content.or_else(|| choice.message.content.clone());
    let assistant = ChatCompletionRequestAssistantMessageRaw {
        content: content.map(llmy_client::req::ChatCompletionRequestAssistantMessageContent::Text),
        refusal: None,
        name: None,
        audio: None,
        tool_calls: choice.message.tool_calls.clone(),
        function_call: choice.message.function_call.clone(),
    };
    Ok(WithOtherFields::new(assistant))
}

#[allow(deprecated)]
pub fn chat_choice_to_toolcalls(choice: &ChatChoice) -> Vec<GeneralToolCall> {
    let mut calls = vec![];

    for tool in choice.message.tool_calls.iter().flatten() {
        let (id, tool_name, args) = match &tool.inner {
            ChatCompletionMessageToolCallsRaw::Function(func) => (
                func.id.clone(),
                func.function.name.clone(),
                func.function.arguments.clone(),
            ),
            ChatCompletionMessageToolCallsRaw::Custom(custom) => (
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
    use llmy_client::req::{ChatCompletionRequestAssistantMessageContent, Role};
    use llmy_client::resp::{ChatChoiceRaw, ChatCompletionResponseMessageRaw};

    #[allow(deprecated)]
    fn choice_with_content(content: &str) -> ChatChoice {
        let message = WithOtherFields::new(ChatCompletionResponseMessageRaw {
            content: Some(content.to_string()),
            refusal: None,
            tool_calls: None,
            annotations: None,
            role: Role::Assistant,
            function_call: None,
            audio: None,
        });
        WithOtherFields::new(ChatChoiceRaw {
            index: 0,
            message,
            finish_reason: None,
            logprobs: None,
        })
    }

    #[test]
    fn chat_choice_to_assistant_preserves_choice_content() {
        let assistant = chat_choice_to_assistant(&choice_with_content("original")).unwrap();

        match assistant.inner.content.as_ref().unwrap() {
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

        match assistant.inner.content.as_ref().unwrap() {
            ChatCompletionRequestAssistantMessageContent::Text(content) => {
                assert_eq!(content, "retry with valid json")
            }
            content => panic!("expected text content, got {content:?}"),
        }
    }
}
