use std::sync::LazyLock;

use regex::Regex;

use super::OpenAIContentFilter;
use crate::resp::RawExtensibleChatCompletionResponse;

static MARKDOWN_TAG_FILTER: LazyLock<MarkdownTagFilter> = LazyLock::new(MarkdownTagFilter::new);

/// Strip the first markdown code fence from `content`, returning the inner text
/// (trimmed), or `None` when there is no non-empty fenced block. Backed by a
/// shared compiled regex, so it is cheap to call repeatedly (e.g. as a parse
/// fallback in [`crate::client`]'s typed completions).
pub fn strip_markdown_fence(content: &str) -> Option<String> {
    MARKDOWN_TAG_FILTER.strip_fence(content)
}

/// Unwraps markdown-fenced model output, e.g.
///
/// ````text
/// ```json
/// {"answer": 42}
/// ```
/// ````
///
/// back into the bare `{"answer": 42}`. Models often wrap JSON they were asked
/// to emit raw inside a ` ```json ` (or plain ` ``` `) code block; this filter
/// strips the surrounding fence so the content parses as JSON. The fence may be
/// the whole message or embedded in surrounding prose — the first fenced block
/// is extracted. Content with no code fence is left untouched.
#[derive(Debug)]
pub struct MarkdownTagFilter {
    fence_re: Regex,
}

impl Default for MarkdownTagFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl MarkdownTagFilter {
    pub fn new() -> Self {
        Self {
            // ```<optional-lang>\n <body> ```  — `(?s)` lets the body span lines.
            fence_re: Regex::new(r"(?s)```[^\n]*\n(.*?)```").unwrap(),
        }
    }

    /// The inner content of the first fenced block, trimmed, or `None` when the
    /// content has no (non-empty) fenced block.
    fn strip_fence(&self, content: &str) -> Option<String> {
        let inner = self.fence_re.captures(content)?.get(1)?.as_str().trim();
        (!inner.is_empty()).then(|| inner.to_string())
    }
}

impl OpenAIContentFilter for MarkdownTagFilter {
    fn filter_output(&self, resp: &mut RawExtensibleChatCompletionResponse) {
        for choice in resp.choices.iter_mut() {
            let Some(content) = choice.message.content.as_deref() else {
                continue;
            };
            if let Some(stripped) = self.strip_fence(content) {
                choice.message.content = Some(stripped);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::build_resp;
    use crate::resp::FinishReason;

    fn filtered(content: &str) -> Option<String> {
        let mut resp = build_resp(Some(content), FinishReason::Stop);
        MarkdownTagFilter::new().filter_output(&mut resp);
        resp.choices[0].message.content.clone()
    }

    #[test]
    fn strips_json_language_fence() {
        assert_eq!(
            filtered("```json\n{\"a\": 1}\n```").as_deref(),
            Some("{\"a\": 1}")
        );
    }

    #[test]
    fn strips_bare_fence() {
        assert_eq!(
            filtered("```\n{\"a\": 1}\n```").as_deref(),
            Some("{\"a\": 1}")
        );
    }

    #[test]
    fn extracts_fenced_block_from_surrounding_prose() {
        let content = "Sure, here is the JSON:\n```json\n{\"a\": 1}\n```\nLet me know!";
        assert_eq!(filtered(content).as_deref(), Some("{\"a\": 1}"));
    }

    #[test]
    fn leaves_plain_json_untouched() {
        assert_eq!(filtered("{\"a\": 1}").as_deref(), Some("{\"a\": 1}"));
    }

    #[test]
    fn leaves_plain_text_untouched() {
        assert_eq!(filtered("just some prose").as_deref(), Some("just some prose"));
    }

    #[test]
    fn ignores_empty_fence() {
        // Nothing inside the fence => leave the content as-is.
        assert_eq!(filtered("```json\n\n```").as_deref(), Some("```json\n\n```"));
    }
}
