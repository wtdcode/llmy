use super::OpenAIContentFilter;

/// A filter that leaves every request and response untouched.
#[derive(Default, Debug)]
pub struct NoFilter;

impl OpenAIContentFilter for NoFilter {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::build_resp;
    use crate::resp::FinishReason;

    #[test]
    fn no_filter_is_a_noop() {
        let filter = NoFilter;
        let content = "<tool_call><function=a></function></tool_call>";
        let mut resp = build_resp(Some(content), FinishReason::ToolCalls);
        filter.filter_output(&mut resp);

        assert_eq!(resp.choices[0].message.content.as_deref(), Some(content));
        assert!(resp.choices[0].message.tool_calls.is_none());
    }
}
