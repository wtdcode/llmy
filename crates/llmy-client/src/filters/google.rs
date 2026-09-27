use super::OpenAIContentFilter;
use crate::client::LLMRequest;

/// Strips fields that Google's OpenAI-compatible chat completion endpoint rejects.
#[derive(Default, Debug)]
pub struct GoogleContentFilter;

impl OpenAIContentFilter for GoogleContentFilter {
    fn filter_input(&self, req: &mut LLMRequest) {
        // Google models are served over the chat protocol only.
        if let LLMRequest::Chat(req) = req {
            req.prompt_cache_key = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::req::RawExtensibleChatCompletionRequest;

    #[test]
    fn google_filter_strips_prompt_cache_key() {
        let filter = GoogleContentFilter;
        let chat: RawExtensibleChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "google/gemini-2.5-pro",
            "messages": [{"role": "user", "content": "hi"}],
            "prompt_cache_key": "some-key"
        }))
        .unwrap();
        assert_eq!(chat.prompt_cache_key.as_deref(), Some("some-key"));

        let mut req = LLMRequest::Chat(chat);
        filter.filter_input(&mut req);
        let LLMRequest::Chat(chat) = req else {
            panic!("a chat request stays chat");
        };
        assert!(chat.prompt_cache_key.is_none());
    }
}
