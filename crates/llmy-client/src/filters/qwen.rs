use serde_json::Value;

use super::OpenAIContentFilter;
use crate::client::LLMRequest;
use crate::req::ChatCompletionToolsRaw;

/// Rewrites tool parameter schemas into the single-type shape qwen's
/// server-side tool parser understands.
///
/// Qwen models emit tool-call arguments as text and the endpoint converts
/// each value to the type named by the parameter's schema. That conversion
/// only handles a single-string `type`: a union like `["integer","null"]` —
/// schemars' encoding of `Option<T>` — defeats it and integers come back as
/// strings (`"start_line": "200"`), which then fail schema validation.
/// Collapsing the null unions on the wire fixes the emission; optionality is
/// still expressed by the field's absence from `required`. Validation keeps
/// running on the original, wider schema, and every value the narrowed
/// schema admits is admitted by the original, so nothing valid is refused.
#[derive(Default, Debug)]
pub struct QwenToolSchemaFilter;

impl OpenAIContentFilter for QwenToolSchemaFilter {
    fn filter_input(&self, req: &mut LLMRequest) {
        // Qwen endpoints speak the chat protocol only.
        let LLMRequest::Chat(req) = req else {
            return;
        };
        let Some(tools) = req.tools.as_mut() else {
            return;
        };
        for tool in tools.iter_mut() {
            if let ChatCompletionToolsRaw::Function(function) = &mut tool.inner
                && let Some(parameters) = function.function.parameters.as_mut()
            {
                collapse_null_unions(parameters);
            }
        }
    }
}

/// Recursively rewrite one schema node: drop `"null"` from `type` arrays
/// (unwrapping to a bare string where a single type remains), inline
/// `anyOf`/`oneOf` unions whose only other arm is `{"type": "null"}`, and
/// drop `"default": null` — the leftovers of schemars' `Option<T>` encoding.
fn collapse_null_unions(schema: &mut Value) {
    match schema {
        Value::Object(object) => {
            if let Some(Value::Array(types)) = object.get_mut("type") {
                types.retain(|t| t != "null");
                let collapsed = match types.len() {
                    // `["null"]` alone denotes the null-only type.
                    0 => Some(Value::String("null".to_string())),
                    1 => Some(types.remove(0)),
                    _ => None,
                };
                if let Some(only) = collapsed {
                    object.insert("type".to_string(), only);
                }
            }
            for key in ["anyOf", "oneOf"] {
                let Some(Value::Array(arms)) = object.get_mut(key) else {
                    continue;
                };
                arms.retain(|arm| arm.get("type") != Some(&Value::String("null".to_string())));
                if arms.len() == 1 {
                    let arm = arms.remove(0);
                    object.remove(key);
                    // The surviving arm becomes this node; keys the node
                    // already has (description, title, ...) win.
                    if let Value::Object(fields) = arm {
                        for (field, value) in fields {
                            object.entry(field).or_insert(value);
                        }
                    }
                }
            }
            if object.get("default") == Some(&Value::Null) {
                object.remove("default");
            }
            for value in object.values_mut() {
                collapse_null_unions(value);
            }
        }
        Value::Array(items) => {
            for value in items.iter_mut() {
                collapse_null_unions(value);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::req::RawExtensibleChatCompletionRequest;

    fn collapsed(schema: serde_json::Value) -> serde_json::Value {
        let mut schema = schema;
        collapse_null_unions(&mut schema);
        schema
    }

    #[test]
    fn null_type_unions_collapse_to_the_bare_type() {
        // The exact shape schemars generates for `Option<usize>`.
        assert_eq!(
            collapsed(serde_json::json!({
                "type": ["integer", "null"],
                "format": "uint",
                "minimum": 0,
                "default": null,
                "description": "Optional line number."
            })),
            serde_json::json!({
                "type": "integer",
                "format": "uint",
                "minimum": 0,
                "description": "Optional line number."
            })
        );
        // Several non-null types keep the (null-free) array.
        assert_eq!(
            collapsed(serde_json::json!({"type": ["integer", "string", "null"]})),
            serde_json::json!({"type": ["integer", "string"]})
        );
    }

    #[test]
    fn any_of_null_unions_inline_the_surviving_arm() {
        // The shape schemars generates for `Option<NestedStruct>`.
        assert_eq!(
            collapsed(serde_json::json!({
                "description": "Optional window.",
                "anyOf": [
                    {"type": "object", "properties": {"start": {"type": ["integer", "null"]}}},
                    {"type": "null"}
                ]
            })),
            serde_json::json!({
                "description": "Optional window.",
                "type": "object",
                "properties": {"start": {"type": "integer"}}
            })
        );
    }

    #[test]
    fn nested_properties_are_rewritten_too() {
        assert_eq!(
            collapsed(serde_json::json!({
                "type": "object",
                "properties": {
                    "outer": {
                        "type": ["object", "null"],
                        "properties": {"inner": {"type": ["number", "null"]}}
                    }
                }
            })),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "outer": {
                        "type": "object",
                        "properties": {"inner": {"type": "number"}}
                    }
                }
            })
        );
    }

    #[test]
    fn the_filter_rewrites_every_function_tool_in_a_chat_request() {
        let chat: RawExtensibleChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "qwen3.8-max",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {
                "name": "read_file",
                "parameters": {"type": "object", "properties": {
                    "file_path": {"type": "string"},
                    "start_line": {"type": ["integer", "null"], "default": null}
                }, "required": ["file_path"]}
            }}]
        }))
        .unwrap();

        let mut req = LLMRequest::Chat(chat);
        QwenToolSchemaFilter.filter_input(&mut req);

        let value = serde_json::to_value(&req).unwrap();
        assert_eq!(
            value["tools"][0]["function"]["parameters"]["properties"]["start_line"],
            serde_json::json!({"type": "integer"})
        );
        // Optionality still lives in `required`.
        assert_eq!(
            value["tools"][0]["function"]["parameters"]["required"],
            serde_json::json!(["file_path"])
        );
    }
}
