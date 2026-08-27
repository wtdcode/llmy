//! Protocol-neutral conversation model.
//!
//! Conversation-state callers (the agent harness above all) hold their history
//! as [`Message`]s — a typed superset of what every supported protocol can
//! carry — instead of any protocol's wire structs. Each protocol builds its
//! native request directly from these parts and parses its response back into
//! them, so protocol-specific material (signed thinking blocks, encrypted
//! reasoning items, provider extras on tool calls) stays typed and survives
//! multi-turn round trips without loss.
//!
//! Lowering to a protocol that has no slot for a part degrades deliberately:
//! readable reasoning text folds into the chat `reasoning_content` extra,
//! signatures and encrypted payloads are dropped (they are only valid on the
//! protocol that issued them), and [`MessagePart::Opaque`] is replayed only on
//! its own protocol.

use llmy_types::other::{OtherFields, WithOtherFields};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::req::{
    ChatCompletionMessageToolCallRaw, ChatCompletionMessageToolCalls,
    ChatCompletionMessageToolCallsRaw, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestAssistantMessageRaw, ChatCompletionRequestDeveloperMessageContent,
    ChatCompletionRequestDeveloperMessageContentPartRaw, ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContentPartImageRaw,
    ChatCompletionRequestMessageContentPartTextRaw, ChatCompletionRequestMessageRaw,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestSystemMessageContentPartRaw,
    ChatCompletionRequestSystemMessageRaw, ChatCompletionRequestToolMessageContent,
    ChatCompletionRequestToolMessageContentPartRaw, ChatCompletionRequestToolMessageRaw,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    ChatCompletionRequestUserMessageContentPartRaw, ChatCompletionRequestUserMessageRaw,
    FunctionCallRaw, ImageUrlRaw, RawExtensibleChatRequestMessage,
};
use crate::resp::ChatChoice;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

/// One typed part of a conversation turn — the superset of what the supported
/// protocols can carry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MessagePart {
    Text {
        text: String,
    },
    /// An image, by URL (possibly a `data:` URL).
    Image {
        url: String,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: String,
        /// Provider extras riding the call (e.g. Gemini's `thought_signature`),
        /// replayed verbatim when lowering back.
        #[serde(default)]
        extra: OtherFields,
    },
    ToolResult {
        id: String,
        content: String,
    },
    /// Anthropic extended thinking; the `signature` must be replayed verbatim
    /// on that protocol. Chat-protocol reasoning (`reasoning_content`) also
    /// lands here, without a signature.
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Anthropic redacted thinking, opaque by design.
    RedactedThinking {
        data: String,
    },
    /// Responses reasoning; `encrypted_content` is what a stateless next turn
    /// carries back.
    Reasoning {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default)]
        summary: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
    },
    /// Last resort for provider constructs no variant models yet; replayed
    /// verbatim, and only on the protocol that produced it.
    Opaque {
        protocol: String,
        value: Value,
    },
}

/// One conversation turn in protocol-neutral form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub parts: Vec<MessagePart>,
    /// Explicit prompt-cache breakpoint at the end of this message.
    #[serde(default)]
    pub cache_breakpoint: bool,
}

impl Message {
    pub fn new(role: MessageRole, parts: Vec<MessagePart>) -> Self {
        Self {
            role,
            parts,
            cache_breakpoint: false,
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self::new(
            MessageRole::System,
            vec![MessagePart::Text { text: text.into() }],
        )
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self::new(
            MessageRole::User,
            vec![MessagePart::Text { text: text.into() }],
        )
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self::new(
            MessageRole::Assistant,
            vec![MessagePart::Text { text: text.into() }],
        )
    }

    pub fn tool_result(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(
            MessageRole::Tool,
            vec![MessagePart::ToolResult {
                id: id.into(),
                content: content.into(),
            }],
        )
    }

    /// Toggle the explicit cache breakpoint at the end of this message.
    pub fn toggle_cache_breakpoint(&mut self, enabled: bool) {
        self.cache_breakpoint = enabled;
    }

    /// Mark this message as an explicit cache breakpoint — shorthand for
    /// `toggle_cache_breakpoint(true)`, and idempotent.
    pub fn breakpoint(&mut self) {
        self.cache_breakpoint = true;
    }

    /// The concatenated visible text of this message.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for part in &self.parts {
            if let MessagePart::Text { text } = part {
                out.push_str(text);
            }
        }
        out
    }

    // -----------------------------------------------------------------------
    // Lowering to chat-completion form
    // -----------------------------------------------------------------------

    /// Lower this message into chat-completion request messages. Most map 1:1;
    /// `ToolResult` parts split out into their own `tool` messages, and
    /// protocol-native residue with no chat slot degrades deliberately: the
    /// readable reasoning text lands in the `reasoning_content` extra, while
    /// signatures, encrypted payloads and opaque parts are dropped (they are
    /// only valid on the protocol that issued them).
    pub fn to_chat_messages(&self) -> Vec<ChatCompletionRequestMessage> {
        let mut out: Vec<ChatCompletionRequestMessage> = Vec::new();
        match self.role {
            MessageRole::System => {
                out.push(WithOtherFields::new(
                    ChatCompletionRequestMessageRaw::System(
                        ChatCompletionRequestSystemMessageRaw::new_text(self.text()),
                    ),
                ));
            }
            MessageRole::User | MessageRole::Tool => {
                let mut parts: Vec<ChatCompletionRequestUserMessageContentPart> = Vec::new();
                for part in &self.parts {
                    match part {
                        MessagePart::Text { text } => parts.push(WithOtherFields::new(
                            ChatCompletionRequestUserMessageContentPartRaw::Text(
                                WithOtherFields::new(
                                    ChatCompletionRequestMessageContentPartTextRaw {
                                        text: text.clone(),
                                        prompt_cache_breakpoint: None,
                                    },
                                ),
                            ),
                        )),
                        MessagePart::Image { url } => parts.push(WithOtherFields::new(
                            ChatCompletionRequestUserMessageContentPartRaw::ImageUrl(
                                WithOtherFields::new(
                                    ChatCompletionRequestMessageContentPartImageRaw {
                                        image_url: WithOtherFields::new(ImageUrlRaw {
                                            url: url.clone(),
                                            detail: None,
                                        }),
                                        prompt_cache_breakpoint: None,
                                    },
                                ),
                            ),
                        )),
                        MessagePart::ToolResult { id, content } => {
                            Self::flush_user_parts(&mut parts, &mut out);
                            out.push(WithOtherFields::new(ChatCompletionRequestMessageRaw::Tool(
                                ChatCompletionRequestToolMessageRaw::new_text(
                                    content.clone(),
                                    id.clone(),
                                ),
                            )));
                        }
                        // Nothing else belongs in a user turn.
                        _ => {}
                    }
                }
                Self::flush_user_parts(&mut parts, &mut out);
            }
            MessageRole::Assistant => {
                let mut text = String::new();
                let mut reasoning = String::new();
                let mut tool_calls: Vec<ChatCompletionMessageToolCalls> = Vec::new();
                for part in &self.parts {
                    match part {
                        MessagePart::Text { text: chunk } => text.push_str(chunk),
                        MessagePart::Thinking { thinking, .. } => {
                            if !reasoning.is_empty() {
                                reasoning.push('\n');
                            }
                            reasoning.push_str(thinking);
                        }
                        MessagePart::Reasoning { summary, .. } => {
                            for chunk in summary {
                                if !reasoning.is_empty() {
                                    reasoning.push('\n');
                                }
                                reasoning.push_str(chunk);
                            }
                        }
                        MessagePart::ToolCall {
                            id,
                            name,
                            arguments,
                            extra,
                        } => {
                            let mut call = WithOtherFields::new(ChatCompletionMessageToolCallRaw {
                                id: id.clone(),
                                function: WithOtherFields::new(FunctionCallRaw {
                                    name: name.clone(),
                                    arguments: arguments.clone(),
                                }),
                            });
                            call.other = extra.clone();
                            tool_calls.push(WithOtherFields::new(
                                ChatCompletionMessageToolCallsRaw::Function(call),
                            ));
                        }
                        MessagePart::Image { .. }
                        | MessagePart::ToolResult { .. }
                        | MessagePart::RedactedThinking { .. }
                        | MessagePart::Opaque { .. } => {}
                    }
                }
                #[allow(deprecated)]
                let mut assistant =
                    WithOtherFields::new(ChatCompletionRequestAssistantMessageRaw {
                        content: (!text.is_empty())
                            .then(|| ChatCompletionRequestAssistantMessageContent::Text(text)),
                        refusal: None,
                        name: None,
                        audio: None,
                        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                        function_call: None,
                    });
                if !reasoning.is_empty() {
                    assistant
                        .other
                        .insert("reasoning_content".to_string(), Value::String(reasoning));
                }
                out.push(WithOtherFields::new(
                    ChatCompletionRequestMessageRaw::Assistant(assistant),
                ));
            }
        }
        if self.cache_breakpoint
            && let Some(last) = out.last_mut()
        {
            last.inner.toggle_cache_breakpoint(true);
        }
        out
    }

    fn flush_user_parts(
        parts: &mut Vec<ChatCompletionRequestUserMessageContentPart>,
        out: &mut Vec<ChatCompletionRequestMessage>,
    ) {
        if parts.is_empty() {
            return;
        }
        let mut content = ChatCompletionRequestUserMessageContent::Array(std::mem::take(parts));
        content.compact();
        out.push(WithOtherFields::new(ChatCompletionRequestMessageRaw::User(
            WithOtherFields::new(ChatCompletionRequestUserMessageRaw {
                content,
                name: None,
            }),
        )));
    }

    /// Lower a whole conversation into the chat-completion message list.
    pub fn many_to_chat(messages: &[Message]) -> Vec<RawExtensibleChatRequestMessage> {
        messages
            .iter()
            .flat_map(Message::to_chat_messages)
            .map(RawExtensibleChatRequestMessage::from_message)
            .collect()
    }

    // -----------------------------------------------------------------------
    // Reading protocol responses / chat-typed messages back
    // -----------------------------------------------------------------------

    /// The assistant turn of a chat-protocol response, in neutral form.
    /// Provider extras on tool calls (Gemini's `thought_signature`) are
    /// carried; a non-blank `reasoning_content` extra becomes a
    /// [`MessagePart::Thinking`] without signature. A custom tool call is
    /// folded to a function-style [`MessagePart::ToolCall`].
    pub fn from_chat_choice(choice: &ChatChoice) -> Message {
        let mut parts = Vec::new();
        if let Some(reasoning) = choice
            .inner
            .message
            .other
            .get("reasoning_content")
            .and_then(|value| value.as_str())
            && !reasoning.trim().is_empty()
        {
            parts.push(MessagePart::Thinking {
                thinking: reasoning.to_string(),
                signature: None,
            });
        }
        if let Some(content) = &choice.inner.message.inner.content
            && !content.is_empty()
        {
            parts.push(MessagePart::Text {
                text: content.clone(),
            });
        }
        for call in choice.inner.message.inner.tool_calls.iter().flatten() {
            match &call.inner {
                ChatCompletionMessageToolCallsRaw::Function(call) => {
                    parts.push(MessagePart::ToolCall {
                        id: call.inner.id.clone(),
                        name: call.inner.function.inner.name.clone(),
                        arguments: call.inner.function.inner.arguments.clone(),
                        extra: call.other.clone(),
                    })
                }
                ChatCompletionMessageToolCallsRaw::Custom(call) => {
                    parts.push(MessagePart::ToolCall {
                        id: call.inner.id.clone(),
                        name: call.inner.custom_tool.inner.name.clone(),
                        arguments: call.inner.custom_tool.inner.input.clone(),
                        extra: call.other.clone(),
                    })
                }
            }
        }
        Message::new(MessageRole::Assistant, parts)
    }

    /// Fold a chat-typed request message into neutral form — used for tool
    /// outputs coming from the toolbox, which speaks chat types.
    pub fn from_chat_request(msg: &ChatCompletionRequestMessageRaw) -> Message {
        match msg {
            ChatCompletionRequestMessageRaw::System(m) => {
                Message::system(Self::system_text(&m.inner.content))
            }
            ChatCompletionRequestMessageRaw::Developer(m) => {
                Message::system(Self::developer_text(&m.inner.content))
            }
            ChatCompletionRequestMessageRaw::User(m) => {
                let parts = match &m.inner.content {
                    ChatCompletionRequestUserMessageContent::Text(text) => {
                        vec![MessagePart::Text { text: text.clone() }]
                    }
                    ChatCompletionRequestUserMessageContent::Array(chat_parts) => chat_parts
                        .iter()
                        .filter_map(|part| match &part.inner {
                            ChatCompletionRequestUserMessageContentPartRaw::Text(text) => {
                                Some(MessagePart::Text {
                                    text: text.inner.text.clone(),
                                })
                            }
                            ChatCompletionRequestUserMessageContentPartRaw::ImageUrl(image) => {
                                Some(MessagePart::Image {
                                    url: image.inner.image_url.inner.url.clone(),
                                })
                            }
                            _ => None,
                        })
                        .collect(),
                };
                Message::new(MessageRole::User, parts)
            }
            ChatCompletionRequestMessageRaw::Assistant(m) => {
                let mut parts = Vec::new();
                if let Some(content) = &m.inner.content {
                    let text = match content {
                        ChatCompletionRequestAssistantMessageContent::Text(text) => text.clone(),
                        ChatCompletionRequestAssistantMessageContent::Array(chat_parts) => {
                            chat_parts
                                .iter()
                                .map(|part| match &part.inner {
                                    crate::req::ChatCompletionRequestAssistantMessageContentPartRaw::Text(text) => text.inner.text.as_str(),
                                    crate::req::ChatCompletionRequestAssistantMessageContentPartRaw::Refusal(refusal) => refusal.inner.refusal.as_str(),
                                })
                                .collect::<Vec<_>>()
                                .join("")
                        }
                    };
                    if !text.is_empty() {
                        parts.push(MessagePart::Text { text });
                    }
                }
                for call in m.inner.tool_calls.iter().flatten() {
                    if let ChatCompletionMessageToolCallsRaw::Function(call) = &call.inner {
                        parts.push(MessagePart::ToolCall {
                            id: call.inner.id.clone(),
                            name: call.inner.function.inner.name.clone(),
                            arguments: call.inner.function.inner.arguments.clone(),
                            extra: call.other.clone(),
                        });
                    }
                }
                Message::new(MessageRole::Assistant, parts)
            }
            ChatCompletionRequestMessageRaw::Tool(m) => Message::tool_result(
                m.inner.tool_call_id.clone(),
                Self::tool_text(&m.inner.content),
            ),
            ChatCompletionRequestMessageRaw::Function(m) => Message::tool_result(
                m.inner.name.clone(),
                m.inner.content.clone().unwrap_or_default(),
            ),
        }
    }

    fn system_text(content: &ChatCompletionRequestSystemMessageContent) -> String {
        match content {
            ChatCompletionRequestSystemMessageContent::Text(text) => text.clone(),
            ChatCompletionRequestSystemMessageContent::Array(parts) => parts
                .iter()
                .map(|part| {
                    let ChatCompletionRequestSystemMessageContentPartRaw::Text(text) = &part.inner;
                    text.inner.text.as_str()
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    fn developer_text(content: &ChatCompletionRequestDeveloperMessageContent) -> String {
        match content {
            ChatCompletionRequestDeveloperMessageContent::Text(text) => text.clone(),
            ChatCompletionRequestDeveloperMessageContent::Array(parts) => parts
                .iter()
                .map(|part| {
                    let ChatCompletionRequestDeveloperMessageContentPartRaw::Text(text) =
                        &part.inner;
                    text.inner.text.as_str()
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    fn tool_text(content: &ChatCompletionRequestToolMessageContent) -> String {
        match content {
            ChatCompletionRequestToolMessageContent::Text(text) => text.clone(),
            ChatCompletionRequestToolMessageContent::Array(parts) => parts
                .iter()
                .map(|part| {
                    let ChatCompletionRequestToolMessageContentPartRaw::Text(text) = &part.inner;
                    text.inner.text.as_str()
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_assistant_turn_lowers_to_one_chat_message_with_extras_carried() {
        let mut extra = OtherFields::default();
        extra.insert("thought_signature".to_string(), serde_json::json!("c2ln"));
        let message = Message::new(
            MessageRole::Assistant,
            vec![
                MessagePart::Thinking {
                    thinking: "hmm".to_string(),
                    signature: Some("sig-1".to_string()),
                },
                MessagePart::Text {
                    text: "on it".to_string(),
                },
                MessagePart::ToolCall {
                    id: "call_1".to_string(),
                    name: "lookup".to_string(),
                    arguments: "{\"q\":1}".to_string(),
                    extra,
                },
            ],
        );

        let chat = message.to_chat_messages();
        assert_eq!(chat.len(), 1);
        let value = serde_json::to_value(&chat[0]).unwrap();
        assert_eq!(value["role"], "assistant");
        assert_eq!(value["content"], "on it");
        // Readable reasoning folds into the chat extra; the signature has no
        // chat slot and is dropped here (native protocols replay it typed).
        assert_eq!(value["reasoning_content"], "hmm");
        assert_eq!(value["tool_calls"][0]["id"], "call_1");
        assert_eq!(value["tool_calls"][0]["thought_signature"], "c2ln");
    }

    #[test]
    fn tool_results_split_out_and_breakpoints_mark_the_last_message() {
        let mut message = Message::new(
            MessageRole::User,
            vec![
                MessagePart::ToolResult {
                    id: "call_1".to_string(),
                    content: "found".to_string(),
                },
                MessagePart::Text {
                    text: "so?".to_string(),
                },
            ],
        );
        message.breakpoint();

        let chat = message.to_chat_messages();
        assert_eq!(chat.len(), 2);
        let tool = serde_json::to_value(&chat[0]).unwrap();
        assert_eq!(tool["role"], "tool");
        assert_eq!(tool["tool_call_id"], "call_1");
        let user = serde_json::to_value(&chat[1]).unwrap();
        assert_eq!(user["role"], "user");
        // The message-level breakpoint lands on the last lowered message.
        assert_eq!(
            user["content"][0]["prompt_cache_breakpoint"]["mode"],
            "explicit"
        );
        assert!(tool.get("prompt_cache_breakpoint").is_none());
    }

    #[test]
    fn a_chat_choice_reads_back_into_typed_parts() {
        let choice: ChatChoice = serde_json::from_value(serde_json::json!({
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "call the tool",
                "reasoning_content": "I need the tool.",
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                    "thought_signature": "c2ln"
                }]
            },
            "finish_reason": "tool_calls"
        }))
        .unwrap();

        let message = Message::from_chat_choice(&choice);
        assert_eq!(message.role, MessageRole::Assistant);
        assert_eq!(message.parts.len(), 3);
        assert!(matches!(
            &message.parts[0],
            MessagePart::Thinking { thinking, signature: None } if thinking == "I need the tool."
        ));
        assert!(matches!(
            &message.parts[1],
            MessagePart::Text { text } if text == "call the tool"
        ));
        match &message.parts[2] {
            MessagePart::ToolCall {
                id, name, extra, ..
            } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "lookup");
                assert_eq!(
                    extra.get("thought_signature"),
                    Some(&serde_json::json!("c2ln"))
                );
            }
            other => panic!("expected a tool call part, got {other:?}"),
        }

        // Round trip: lowering the read-back message reproduces the extras.
        let value = serde_json::to_value(&message.to_chat_messages()[0]).unwrap();
        assert_eq!(value["tool_calls"][0]["thought_signature"], "c2ln");
        assert_eq!(value["reasoning_content"], "I need the tool.");
    }

    #[test]
    fn a_toolbox_tool_message_folds_into_a_tool_result() {
        let msg = ChatCompletionRequestMessageRaw::Tool(
            crate::req::ChatCompletionRequestToolMessageRaw::new_text("found", "call_1"),
        );
        let message = Message::from_chat_request(&msg);
        assert_eq!(message.role, MessageRole::Tool);
        assert_eq!(
            message.parts,
            vec![MessagePart::ToolResult {
                id: "call_1".to_string(),
                content: "found".to_string()
            }]
        );
    }
}
