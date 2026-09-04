//! Scratchpad mode: a JSON document handed to the harness on the command
//! line, rendered into the opening prompt, and mutated by the agent through
//! JSON Pointer operations. Every mutation is persisted atomically (write to
//! a temp file, then rename) so an interrupted run never leaves a torn file.

use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};

use color_eyre::eyre::eyre;
use llmy_types::error::LLMYError;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

/// One JSON Pointer mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ScratchpadOp {
    /// Set the value at the pointer, creating the final object key if needed.
    /// On arrays, an index equal to the length (or `-`) appends.
    Set,
    /// Append the value to the array at the pointer.
    Append,
    /// Delete the value at the pointer (object key or array element).
    Delete,
}

#[derive(Debug)]
struct ScratchpadInner {
    path: PathBuf,
    value: StdMutex<Value>,
}

/// Shared, file-backed JSON scratchpad.
#[derive(Debug, Clone)]
pub struct Scratchpad {
    inner: Arc<ScratchpadInner>,
}

impl Scratchpad {
    /// Load the scratchpad from `path`. A missing file starts as `{}` and is
    /// created on the first mutation.
    pub async fn load(path: PathBuf) -> Result<Self, LLMYError> {
        let value = match tokio::fs::read_to_string(&path).await {
            Ok(content) => serde_json::from_str::<Value>(&content)
                .map_err(|e| eyre!("scratchpad {} is not valid JSON: {}", path.display(), e))?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                tracing::info!(
                    "scratchpad {} does not exist yet; starting empty",
                    path.display()
                );
                Value::Object(serde_json::Map::new())
            }
            Err(error) => return Err(error.into()),
        };
        Ok(Self {
            inner: Arc::new(ScratchpadInner {
                path,
                value: StdMutex::new(value),
            }),
        })
    }

    pub fn render(&self) -> String {
        let value = self.lock_value().clone();
        serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string())
    }

    pub fn render_pointer(&self, pointer: &str) -> Result<String, String> {
        let value = self.lock_value();
        match value.pointer(pointer) {
            Some(sub) => Ok(serde_json::to_string_pretty(sub).unwrap_or_else(|_| sub.to_string())),
            None => Err(format!("nothing exists at pointer {pointer:?}")),
        }
    }

    fn lock_value(&self) -> std::sync::MutexGuard<'_, Value> {
        match self.inner.value.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Apply one mutation and persist the document. The `Err` string is a
    /// model-facing explanation of why the operation is invalid.
    pub async fn apply(
        &self,
        pointer: &str,
        op: ScratchpadOp,
        value: Option<Value>,
    ) -> Result<String, String> {
        let snapshot = {
            let mut doc = self.lock_value();
            Self::apply_to(&mut doc, pointer, op, value)?;
            doc.clone()
        };
        self.persist(&snapshot).await.map_err(|error| {
            format!("the mutation applied but persisting the scratchpad failed: {error}")
        })?;
        Ok(format!("Applied {op:?} at {pointer:?}."))
    }

    fn apply_to(
        doc: &mut Value,
        pointer: &str,
        op: ScratchpadOp,
        value: Option<Value>,
    ) -> Result<(), String> {
        match op {
            ScratchpadOp::Set => {
                let value =
                    value.ok_or_else(|| "`value` is required for the set op".to_string())?;
                if pointer.is_empty() {
                    *doc = value;
                    return Ok(());
                }
                let (parent_pointer, key) = Self::split_pointer(pointer)?;
                let parent = doc
                    .pointer_mut(&parent_pointer)
                    .ok_or_else(|| format!("parent pointer {parent_pointer:?} does not exist"))?;
                match parent {
                    Value::Object(map) => {
                        map.insert(key, value);
                        Ok(())
                    }
                    Value::Array(items) => {
                        if key == "-" {
                            items.push(value);
                            return Ok(());
                        }
                        let index: usize = key.parse().map_err(|_| {
                            format!("array index {key:?} is not a number (use `-` to append)")
                        })?;
                        if index < items.len() {
                            items[index] = value;
                            Ok(())
                        } else if index == items.len() {
                            items.push(value);
                            Ok(())
                        } else {
                            Err(format!(
                                "array index {index} is out of bounds (length {})",
                                items.len()
                            ))
                        }
                    }
                    other => Err(format!(
                        "parent at {parent_pointer:?} is a {}, not an object or array",
                        json_type(other)
                    )),
                }
            }
            ScratchpadOp::Append => {
                let value =
                    value.ok_or_else(|| "`value` is required for the append op".to_string())?;
                let target = doc
                    .pointer_mut(pointer)
                    .ok_or_else(|| format!("nothing exists at pointer {pointer:?}"))?;
                match target {
                    Value::Array(items) => {
                        items.push(value);
                        Ok(())
                    }
                    other => Err(format!(
                        "value at {pointer:?} is a {}, not an array",
                        json_type(other)
                    )),
                }
            }
            ScratchpadOp::Delete => {
                if pointer.is_empty() {
                    return Err("cannot delete the document root".to_string());
                }
                let (parent_pointer, key) = Self::split_pointer(pointer)?;
                let parent = doc
                    .pointer_mut(&parent_pointer)
                    .ok_or_else(|| format!("parent pointer {parent_pointer:?} does not exist"))?;
                match parent {
                    Value::Object(map) => match map.remove(&key) {
                        Some(_) => Ok(()),
                        None => Err(format!("object has no key {key:?}")),
                    },
                    Value::Array(items) => {
                        let index: usize = key
                            .parse()
                            .map_err(|_| format!("array index {key:?} is not a number"))?;
                        if index < items.len() {
                            items.remove(index);
                            Ok(())
                        } else {
                            Err(format!(
                                "array index {index} is out of bounds (length {})",
                                items.len()
                            ))
                        }
                    }
                    other => Err(format!(
                        "parent at {parent_pointer:?} is a {}, not an object or array",
                        json_type(other)
                    )),
                }
            }
        }
    }

    /// Split an RFC 6901 pointer into its parent pointer and the final,
    /// unescaped reference token.
    fn split_pointer(pointer: &str) -> Result<(String, String), String> {
        if !pointer.starts_with('/') {
            return Err(format!(
                "pointer {pointer:?} must start with '/' (RFC 6901)"
            ));
        }
        let split_at = pointer
            .rfind('/')
            .ok_or_else(|| format!("pointer {pointer:?} has no reference token"))?;
        let parent = pointer[..split_at].to_string();
        let token = pointer[split_at + 1..]
            .replace("~1", "/")
            .replace("~0", "~");
        Ok((parent, token))
    }

    async fn persist(&self, value: &Value) -> Result<(), LLMYError> {
        let rendered = serde_json::to_string_pretty(value)
            .map_err(|e| eyre!("failed to serialize scratchpad: {}", e))?;
        let tmp_path = self.inner.path.with_extension("json.tmp");
        tokio::fs::write(&tmp_path, rendered.as_bytes()).await?;
        tokio::fs::rename(&tmp_path, &self.inner.path).await?;
        Ok(())
    }
}

fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Arguments accepted by [`UpdateJsonFieldTool`].
#[derive(Deserialize, JsonSchema)]
pub struct UpdateJsonFieldArgs {
    /// RFC 6901 JSON Pointer to the target (e.g. "/findings/0/severity",
    /// "" for the document root). `~0` escapes `~`, `~1` escapes `/`.
    pub pointer: String,
    /// The operation: set, append (arrays) or delete.
    pub op: ScratchpadOp,
    /// The JSON value for set/append. Ignored for delete.
    #[serde(default)]
    pub value: Option<Value>,
}

/// Mutates the scratchpad JSON document.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = UpdateJsonFieldArgs,
    invoke = update,
    name = "update_json_field",
    description = "Update the scratchpad JSON document at a JSON Pointer (RFC 6901). Ops: `set` writes a value (on arrays, index or `-` to append), `append` pushes onto an array, `delete` removes an object key or array element. Every change is persisted immediately.",
)]
pub struct UpdateJsonFieldTool {
    scratchpad: Scratchpad,
}

impl UpdateJsonFieldTool {
    pub fn new(scratchpad: Scratchpad) -> Self {
        Self { scratchpad }
    }

    async fn update(&self, args: UpdateJsonFieldArgs) -> Result<String, LLMYError> {
        match self
            .scratchpad
            .apply(&args.pointer, args.op, args.value)
            .await
        {
            Ok(message) => Ok(message),
            Err(message) => Ok(format!("update_json_field failed: {message}")),
        }
    }
}

/// Arguments accepted by [`ReadScratchpadTool`].
#[derive(Deserialize, JsonSchema)]
pub struct ReadScratchpadArgs {
    /// Optional JSON Pointer to read a sub-value; omit for the whole
    /// document.
    #[serde(default)]
    pub pointer: Option<String>,
}

/// Reads the current scratchpad state.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = ReadScratchpadArgs,
    invoke = read,
    name = "read_scratchpad",
    description = "Read the current scratchpad JSON (optionally a sub-value at a JSON Pointer). The opening prompt contains the initial version; use this after mutations to see the current state.",
)]
pub struct ReadScratchpadTool {
    scratchpad: Scratchpad,
}

impl ReadScratchpadTool {
    pub fn new(scratchpad: Scratchpad) -> Self {
        Self { scratchpad }
    }

    async fn read(&self, args: ReadScratchpadArgs) -> Result<String, LLMYError> {
        match args.pointer.as_deref() {
            None | Some("") => Ok(self.scratchpad.render()),
            Some(pointer) => match self.scratchpad.render_pointer(pointer) {
                Ok(rendered) => Ok(rendered),
                Err(message) => Ok(format!("read_scratchpad failed: {message}")),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc() -> Value {
        serde_json::from_str(r#"{"a": {"b": [1, 2]}, "s": "x"}"#).expect("valid json")
    }

    #[test]
    fn set_creates_and_replaces_keys() {
        let mut value = doc();
        Scratchpad::apply_to(&mut value, "/a/c", ScratchpadOp::Set, Some(Value::from(3)))
            .expect("set new key");
        Scratchpad::apply_to(&mut value, "/s", ScratchpadOp::Set, Some(Value::from("y")))
            .expect("replace key");
        assert_eq!(value["a"]["c"], 3);
        assert_eq!(value["s"], "y");
    }

    #[test]
    fn set_on_arrays_replaces_or_appends() {
        let mut value = doc();
        Scratchpad::apply_to(
            &mut value,
            "/a/b/0",
            ScratchpadOp::Set,
            Some(Value::from(9)),
        )
        .expect("replace index");
        Scratchpad::apply_to(
            &mut value,
            "/a/b/-",
            ScratchpadOp::Set,
            Some(Value::from(7)),
        )
        .expect("append via -");
        Scratchpad::apply_to(
            &mut value,
            "/a/b/3",
            ScratchpadOp::Set,
            Some(Value::from(8)),
        )
        .expect("append via len index");
        assert_eq!(value["a"]["b"], serde_json::json!([9, 2, 7, 8]));

        let error = Scratchpad::apply_to(
            &mut value,
            "/a/b/9",
            ScratchpadOp::Set,
            Some(Value::from(0)),
        )
        .expect_err("out of bounds");
        assert!(error.contains("out of bounds"));
    }

    #[test]
    fn append_requires_an_array() {
        let mut value = doc();
        Scratchpad::apply_to(
            &mut value,
            "/a/b",
            ScratchpadOp::Append,
            Some(Value::from(3)),
        )
        .expect("append");
        assert_eq!(value["a"]["b"], serde_json::json!([1, 2, 3]));

        let error =
            Scratchpad::apply_to(&mut value, "/s", ScratchpadOp::Append, Some(Value::from(1)))
                .expect_err("append to string");
        assert!(error.contains("not an array"));
    }

    #[test]
    fn delete_removes_keys_and_elements() {
        let mut value = doc();
        Scratchpad::apply_to(&mut value, "/a/b/0", ScratchpadOp::Delete, None).expect("delete idx");
        Scratchpad::apply_to(&mut value, "/s", ScratchpadOp::Delete, None).expect("delete key");
        assert_eq!(value["a"]["b"], serde_json::json!([2]));
        assert!(value.get("s").is_none());

        let error = Scratchpad::apply_to(&mut value, "", ScratchpadOp::Delete, None)
            .expect_err("delete root");
        assert!(error.contains("root"));
    }

    #[test]
    fn pointer_escapes_are_unescaped() {
        let mut value = serde_json::json!({"a/b": {"c~d": 1}});
        Scratchpad::apply_to(
            &mut value,
            "/a~1b/c~0d",
            ScratchpadOp::Set,
            Some(Value::from(2)),
        )
        .expect("escaped pointer");
        assert_eq!(value["a/b"]["c~d"], 2);
    }

    #[test]
    fn set_root_replaces_the_document() {
        let mut value = doc();
        Scratchpad::apply_to(
            &mut value,
            "",
            ScratchpadOp::Set,
            Some(serde_json::json!([1])),
        )
        .expect("set root");
        assert_eq!(value, serde_json::json!([1]));
    }
}
