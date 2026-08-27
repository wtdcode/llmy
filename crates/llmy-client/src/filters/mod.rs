use std::fmt::Debug;

use crate::req::RawExtensibleChatCompletionRequest;
use crate::resp::RawExtensibleChatCompletionResponse;

mod google;
mod json;
mod no_filter;

pub use google::GoogleContentFilter;
pub use json::{MarkdownTagFilter, strip_markdown_fence};
pub use no_filter::NoFilter;

/// A hook applied to every outgoing request and incoming response, used to
/// paper over provider-specific quirks (rejected fields, malformed tool calls,
/// markdown-wrapped JSON, ...). The default impls are no-ops, so a filter only
/// overrides the direction it cares about.
pub trait OpenAIContentFilter: Send + Sync + Debug {
    fn filter_input(&self, _req: &mut RawExtensibleChatCompletionRequest) {}
    fn filter_output(&self, _resp: &mut RawExtensibleChatCompletionResponse) {}
}

#[derive(Debug, Default)]
pub struct OpenAIContentFilterChain {
    pub filters: Vec<Box<dyn OpenAIContentFilter>>,
}

impl OpenAIContentFilterChain {
    pub fn new(filters: Vec<Box<dyn OpenAIContentFilter>>) -> Self {
        Self { filters }
    }

    fn chain_filter_input(&self, req: &mut RawExtensibleChatCompletionRequest) {
        for filter in &self.filters {
            filter.filter_input(req);
        }
    }

    fn chain_filter_output(&self, resp: &mut RawExtensibleChatCompletionResponse) {
        for filter in &self.filters {
            filter.filter_output(resp);
        }
    }
}

impl OpenAIContentFilter for OpenAIContentFilterChain {
    fn filter_input(&self, req: &mut RawExtensibleChatCompletionRequest) {
        self.chain_filter_input(req);
    }

    fn filter_output(&self, resp: &mut RawExtensibleChatCompletionResponse) {
        self.chain_filter_output(resp);
    }
}

/// Shared test helper: a minimal response carrying the given assistant content.
#[cfg(test)]
pub(crate) fn build_resp(
    content: Option<&str>,
    finish: crate::resp::FinishReason,
) -> RawExtensibleChatCompletionResponse {
    use crate::resp::FinishReason;
    let body = serde_json::json!({
        "id": "chatcmpl-test",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": match finish {
                FinishReason::ToolCalls => "tool_calls",
                FinishReason::Stop => "stop",
                _ => "stop",
            }
        }],
        "created": 1,
        "model": "test",
        "object": "chat.completion"
    });
    serde_json::from_value(body).unwrap()
}
