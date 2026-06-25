use super::OpenAIContentFilter;
use crate::req::RawExtensibleChatCompletionRequest;

/// Strips fields that Google's OpenAI-compatible chat completion endpoint rejects.
#[derive(Default, Debug)]
pub struct GoogleContentFilter;

impl OpenAIContentFilter for GoogleContentFilter {
    fn filter_input(&self, req: &mut RawExtensibleChatCompletionRequest) {
        req.prompt_cache_key = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn google_filter_strips_prompt_cache_key() {
        let filter = GoogleContentFilter;
        let mut req: RawExtensibleChatCompletionRequest =
            serde_json::from_value(serde_json::json!({
                "model": "google/gemini-2.5-pro",
                "messages": [{"role": "user", "content": "hi"}],
                "prompt_cache_key": "some-key"
            }))
            .unwrap();
        assert_eq!(req.prompt_cache_key.as_deref(), Some("some-key"));
        filter.filter_input(&mut req);
        assert!(req.prompt_cache_key.is_none());
    }
}
