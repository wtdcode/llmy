use std::sync::atomic::{AtomicUsize, Ordering};

use llmy_types::other::WithOtherFields;
use regex::Regex;
use serde_json::{Map, Value};

use super::OpenAIContentFilter;
use crate::req::{
    ChatCompletionMessageToolCallRaw, ChatCompletionMessageToolCalls,
    ChatCompletionMessageToolCallsRaw, FunctionCallRaw,
};
use crate::resp::{FinishReason, RawExtensibleChatCompletionResponse};

/// Salvages mimo's malformed `<tool_call><function=NAME>...</function></tool_call>` content
/// when the model returned `finish_reason: tool_calls` but emitted no actual `tool_calls`.
#[derive(Debug)]
pub struct MiMoContentFilter {
    tool_call_re: Regex,
    function_re: Regex,
    parameter_re: Regex,
    next_id: AtomicUsize,
}

impl Default for MiMoContentFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl MiMoContentFilter {
    pub fn new() -> Self {
        Self {
            tool_call_re: Regex::new(r"(?s)<tool_call>(.*?)</tool_call>").unwrap(),
            function_re: Regex::new(r"(?s)<function=([^>]+)>(.*?)</function>").unwrap(),
            parameter_re: Regex::new(r"(?s)<parameter=([^>]+)>(.*?)</parameter>").unwrap(),
            next_id: AtomicUsize::new(0),
        }
    }

    fn try_parse(&self, content: &str) -> Option<Vec<ChatCompletionMessageToolCalls>> {
        let mut tool_calls = Vec::new();
        for tool_call_caps in self.tool_call_re.captures_iter(content) {
            let inner = tool_call_caps.get(1)?.as_str();
            for function_caps in self.function_re.captures_iter(inner) {
                let name = function_caps.get(1)?.as_str().trim();
                let body = function_caps.get(2)?.as_str();
                if name.is_empty() {
                    continue;
                }
                let mut args = Map::new();
                for param_caps in self.parameter_re.captures_iter(body) {
                    let key = param_caps.get(1)?.as_str().trim();
                    let raw_value = param_caps.get(2)?.as_str().trim();
                    if key.is_empty() {
                        continue;
                    }
                    let value = serde_json::from_str::<Value>(raw_value)
                        .unwrap_or_else(|_| Value::String(raw_value.to_string()));
                    args.insert(key.to_string(), value);
                }
                let arguments = serde_json::to_string(&args).ok()?;
                let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let tool_call = WithOtherFields::new(ChatCompletionMessageToolCallRaw {
                    id: format!("salvage-{}", id),
                    function: WithOtherFields::new(FunctionCallRaw {
                        name: name.to_string(),
                        arguments,
                    }),
                });
                tool_calls.push(WithOtherFields::new(
                    ChatCompletionMessageToolCallsRaw::Function(tool_call),
                ));
            }
        }
        if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        }
    }
}

impl OpenAIContentFilter for MiMoContentFilter {
    fn filter_output(&self, resp: &mut RawExtensibleChatCompletionResponse) {
        for choice in resp.choices.iter_mut() {
            if choice.finish_reason != Some(FinishReason::ToolCalls) {
                continue;
            }
            if choice
                .message
                .tool_calls
                .as_ref()
                .is_some_and(|tcs| !tcs.is_empty())
            {
                continue;
            }
            let Some(content) = choice.message.content.clone() else {
                continue;
            };
            let Some(parsed) = self.try_parse(&content) else {
                continue;
            };
            tracing::warn!(
                "MiMoContentFilter salvaged {} tool call(s) from content",
                parsed.len()
            );
            choice.message.tool_calls = Some(parsed);
            choice.message.content = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::build_resp;

    #[test]
    fn salvages_single_tool_call_with_string_and_numeric_args() {
        let filter = MiMoContentFilter::new();
        let content = "<tool_call>\n<function=read_file>\n<parameter=file_path>CMakeLists.txt</parameter>\n<parameter=line_count>50</parameter>\n</function>\n</tool_call>";
        let mut resp = build_resp(Some(content), FinishReason::ToolCalls);
        filter.filter_output(&mut resp);

        let choice = &resp.choices[0];
        assert!(choice.message.content.is_none());
        let tcs = choice.message.tool_calls.as_ref().unwrap();
        assert_eq!(tcs.len(), 1);
        let ChatCompletionMessageToolCallsRaw::Function(tc) = &tcs[0].inner else {
            panic!("expected function tool call");
        };
        assert_eq!(tc.function.name, "read_file");
        let args: Value = serde_json::from_str(&tc.function.arguments).unwrap();
        assert_eq!(args["file_path"], "CMakeLists.txt");
        assert_eq!(args["line_count"], 50);
    }

    #[test]
    fn salvages_multiple_functions_in_one_tool_call_block() {
        let filter = MiMoContentFilter::new();
        let content = "<tool_call><function=a><parameter=x>1</parameter></function><function=b><parameter=y>two</parameter></function></tool_call>";
        let mut resp = build_resp(Some(content), FinishReason::ToolCalls);
        filter.filter_output(&mut resp);

        let tcs = resp.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tcs.len(), 2);
    }

    #[test]
    fn salvage_ids_are_unique_across_calls() {
        let filter = MiMoContentFilter::new();
        let content = "<tool_call><function=a></function><function=b></function></tool_call>";

        let mut resp1 = build_resp(Some(content), FinishReason::ToolCalls);
        filter.filter_output(&mut resp1);
        let mut resp2 = build_resp(Some(content), FinishReason::ToolCalls);
        filter.filter_output(&mut resp2);

        let collect_ids = |resp: &RawExtensibleChatCompletionResponse| -> Vec<String> {
            resp.choices[0]
                .message
                .tool_calls
                .as_ref()
                .unwrap()
                .iter()
                .map(|tc| match &tc.inner {
                    ChatCompletionMessageToolCallsRaw::Function(f) => f.id.clone(),
                    _ => panic!("expected function"),
                })
                .collect()
        };

        let ids1 = collect_ids(&resp1);
        let ids2 = collect_ids(&resp2);
        assert_eq!(ids1, vec!["salvage-0", "salvage-1"]);
        assert_eq!(ids2, vec!["salvage-2", "salvage-3"]);
    }

    #[test]
    fn does_not_modify_when_finish_reason_is_stop() {
        let filter = MiMoContentFilter::new();
        let content = "<tool_call><function=a></function></tool_call>";
        let mut resp = build_resp(Some(content), FinishReason::Stop);
        filter.filter_output(&mut resp);

        assert_eq!(resp.choices[0].message.content.as_deref(), Some(content));
        assert!(resp.choices[0].message.tool_calls.is_none());
    }

    #[test]
    fn does_not_modify_when_tool_calls_already_present() {
        let filter = MiMoContentFilter::new();
        let mut resp: RawExtensibleChatCompletionResponse =
            serde_json::from_value(serde_json::json!({
                "id": "chatcmpl-test",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "<tool_call><function=a></function></tool_call>",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "real", "arguments": "{}"}
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "created": 1,
                "model": "mimo-v2.5",
                "object": "chat.completion"
            }))
            .unwrap();

        filter.filter_output(&mut resp);

        assert_eq!(
            resp.choices[0].message.tool_calls.as_ref().unwrap().len(),
            1
        );
        let ChatCompletionMessageToolCallsRaw::Function(tc) =
            &resp.choices[0].message.tool_calls.as_ref().unwrap()[0].inner
        else {
            panic!("expected function");
        };
        assert_eq!(tc.function.name, "real");
    }

    #[test]
    fn does_not_modify_when_parse_fails() {
        let filter = MiMoContentFilter::new();
        let content = "this is plain text, no tool call markup at all";
        let mut resp = build_resp(Some(content), FinishReason::ToolCalls);
        filter.filter_output(&mut resp);

        assert_eq!(resp.choices[0].message.content.as_deref(), Some(content));
        assert!(resp.choices[0].message.tool_calls.is_none());
    }
}
