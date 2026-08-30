//! Tool definitions and the [`ToolBox`] registry used by agents.
//!
//! This module exposes two traits for describing tools that a language model
//! can invoke:
//!
//! * [`Tool`] — the typed, ergonomic trait that user code implements (or has
//!   generated via the [`llmy_agent_derive::tool`] attribute macro, re-exported
//!   from `llmy_agent` as `llmy_agent::tool`). Each `Tool` declares a
//!   strongly-typed `ARGUMENTS` type, a `NAME`, an optional `DESCRIPTION`,
//!   and an `invoke` method that receives already-deserialized arguments.
//! * [`ToolDyn`] — the object-safe counterpart, automatically implemented for
//!   every `Tool`. Agents store tools as `dyn ToolDyn` so that a heterogeneous
//!   set of tools can be kept in a single collection.
//!
//! Tools are grouped together in a [`ToolBox`], which exposes them to the
//! model (via [`ToolBox::openai_objects`]) and dispatches incoming tool calls
//! to the matching implementation.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use dyn_clone::DynClone;
use llmy_client::req::{
    ChatCompletionRequestMessageRaw, ChatCompletionRequestToolMessageContent,
    ChatCompletionRequestToolMessageRaw, ChatCompletionTool, ChatCompletionToolRaw,
    ChatCompletionTools, ChatCompletionToolsRaw, FunctionObjectRaw,
};
use llmy_types::error::{GeneralToolCall, LLMYError};
use llmy_types::other::WithOtherFields;
use schemars::schema_for;
use serde::de::DeserializeOwned;
use tokio::task::JoinSet;
use tracing::debug;

/// Object-safe view of a [`Tool`].
///
/// `ToolDyn` erases the `ARGUMENTS` associated type so that tools of different
/// shapes can be stored together (for example inside a [`ToolBox`]). It is
/// implemented automatically for every `T: Tool + 'static`, so library users
/// rarely need to implement it directly — implement [`Tool`] instead.
///
/// All methods take `&self` and the trait is `Send + Sync + Clone` (via
/// [`dyn_clone`]), which lets a tool be cheaply cloned into background tasks.
pub trait ToolDyn: DynClone + Debug + Send + Sync + std::any::Any {
    /// Returns the tool's name as advertised to the model. Must be unique
    /// within a [`ToolBox`].
    fn name(&self) -> String;
    /// Returns the human-readable description shown to the model, if any.
    fn description(&self) -> Option<String>;
    /// Returns the JSON Schema describing this tool's expected arguments.
    fn schema(&self) -> schemars::Schema;
    /// Whether the model should honour the JSON schema strictly.
    fn strict(&self) -> bool {
        false
    }
    /// Renders the tool as an OpenAI [`ChatCompletionTool`] descriptor,
    /// including its JSON schema, ready to be sent in a chat completion
    /// request.
    fn to_openai_obejct(&self) -> ChatCompletionTool {
        WithOtherFields::new(ChatCompletionToolRaw {
            function: WithOtherFields::new(FunctionObjectRaw {
                name: self.name(),
                description: self.description(),
                parameters: Some(
                    serde_json::to_value(self.schema()).expect("Fail to serialize schema"),
                ),
                strict: Some(self.strict()),
            }),
        })
    }
    /// Renders the tool as an MCP [`rmcp::model::Tool`] descriptor.
    fn to_mcp_tool(&self) -> rmcp::model::Tool {
        let input_schema = serde_json::to_value(self.schema()).expect("Fail to serialize schema");
        let input_schema = input_schema.as_object().cloned().unwrap_or_default();
        rmcp::model::Tool::new_with_raw(
            self.name(),
            self.description().map(Into::into),
            Arc::new(input_schema),
        )
    }
    /// Phase-one gate the agent loop runs on every call of a turn before any
    /// tool executes, on the same parsed arguments the execution path uses.
    /// The arguments are assumed to conform to the tool's schema — schema
    /// verdicts belong to the execution path (`IncorrectToolCall`), never to
    /// validate. `Err(reason)` rejects the call: the
    /// loop wraps it into [`LLMYError::ToolCallRejected`] bound to the wire
    /// call, discards the whole model turn (no tool in the batch runs) and
    /// asks again — the reason is logged, never fed back to the model.
    ///
    /// Contract: `validate` may be called any number of times, must return a
    /// stable verdict for the same arguments, and must have no side effects.
    /// The default accepts everything.
    fn validate(
        &self,
        arguments: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + '_>> {
        let _ = arguments;
        Box::pin(async { Ok(()) })
    }
    /// Invokes the tool with raw JSON-encoded `arguments`. The string is
    /// deserialized into the tool's `ARGUMENTS` type by the blanket impl on
    /// top of [`Tool`].
    fn call(
        &self,
        arguments: String,
    ) -> Pin<Box<dyn Future<Output = Result<String, LLMYError>> + Send + '_>> {
        Box::pin(async move {
            match serde_json::from_str::<serde_json::Value>(&arguments) {
                Ok(value) => self.run(value).await,
                Err(_) => Err(LLMYError::IncorrectToolCall(
                    self.name(),
                    arguments,
                    self.schema(),
                )),
            }
        })
    }
    /// Invokes the tool with a [`serde_json::Value`] as arguments.
    fn run(
        &self,
        arguments: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String, LLMYError>> + Send + '_>>;
}

/// Downcasts a `&dyn ToolDyn` to a concrete tool type.
///
/// # Panics
///
/// Panics if `tool` is not actually an instance of `T`. Use this only when the
/// concrete type is known by construction — for general dispatch, prefer the
/// trait methods on [`ToolDyn`].
pub fn downcast_tool<T: 'static>(tool: &dyn ToolDyn) -> &T {
    (tool as &dyn std::any::Any)
        .downcast_ref::<T>()
        .expect("can not downcast")
}

dyn_clone::clone_trait_object!(ToolDyn);

/// A typed tool that an agent can call.
///
/// Implementors describe the tool with associated constants and an
/// [`Self::invoke`] method that receives already-deserialized arguments. The
/// blanket `impl<T: Tool> ToolDyn for T` takes care of JSON deserialization,
/// schema generation and OpenAI-shaped serialization, so most call sites only
/// ever interact with [`ToolDyn`].
///
/// # Deriving an implementation
///
/// The companion [`llmy_agent_derive::tool`] attribute macro (re-exported as
/// `llmy_agent::tool`, and also reachable through the umbrella crate as
/// `llmy::agent::tool`) can generate this trait for a struct, wiring the
/// associated constants and forwarding `invoke` to a method on the struct:
///
/// ```ignore
/// use llmy_agent::tool;
/// use llmy_types::error::LLMYError;
/// use schemars::JsonSchema;
/// use serde::Deserialize;
///
/// #[derive(Deserialize, JsonSchema)]
/// struct EchoArgs { message: String }
///
/// #[derive(Clone, Debug)]
/// #[tool(
///     description = "Echo a message back",
///     arguments = EchoArgs,
///     invoke = run,
/// )]
/// struct EchoTool;
///
/// impl EchoTool {
///     async fn run(&self, args: EchoArgs) -> Result<String, LLMYError> {
///         Ok(args.message)
///     }
/// }
/// ```
///
/// The macro accepts `description`, `arguments`, `invoke` (required) and an
/// optional `name` (defaulting to the struct identifier in `snake_case`).
pub trait Tool: Send + Sync + DynClone + Debug {
    /// The strongly-typed argument struct. It must implement
    /// [`serde::de::DeserializeOwned`] (to be parsed from the model's JSON
    /// payload) and [`schemars::JsonSchema`] (to generate the schema sent to
    /// the model).
    type ARGUMENTS: DeserializeOwned + schemars::JsonSchema + Sized + Send;
    /// Unique name advertised to the model.
    const NAME: &str;
    /// Optional human-readable description shown to the model.
    const DESCRIPTION: Option<&str>;
    /// Whether the model should be asked to honour the JSON schema strictly.
    /// Maps to OpenAI's `strict` field on the function descriptor.
    const STRICT: bool = false;

    /// Performs the tool's actual work on already-deserialized `arguments`
    /// and returns the textual result that will be sent back to the model.
    fn invoke(
        &self,
        arguments: Self::ARGUMENTS,
    ) -> impl Future<Output = Result<String, LLMYError>> + Send;

    /// Phase-one gate on the same deserialized `arguments` [`Self::invoke`]
    /// receives, run by the agent loop before any tool of the turn executes.
    /// The arguments are already schema-checked — assume they conform; a
    /// mismatch never reaches here (the execution path reports it as
    /// [`LLMYError::IncorrectToolCall`]).
    /// `Err(reason)` rejects the call: the loop wraps it into
    /// [`LLMYError::ToolCallRejected`] bound to the wire call and discards
    /// the whole model turn — nothing has run yet, so the rejection costs
    /// zero side effects — then asks again; the reason is logged,
    /// never fed back to the model.
    ///
    /// Contract: `validate` may be called any number of times, must return a
    /// stable verdict for the same arguments, and must have no side effects.
    /// The default accepts everything.
    fn validate(
        &self,
        arguments: Self::ARGUMENTS,
    ) -> impl Future<Output = Result<(), String>> + Send {
        let _ = arguments;
        async { Ok(()) }
    }
}

impl<T: Tool + DynClone + 'static> ToolDyn for T {
    fn name(&self) -> String {
        Self::NAME.to_string()
    }
    fn description(&self) -> Option<String> {
        Self::DESCRIPTION.map(|v| v.to_string())
    }
    fn schema(&self) -> schemars::Schema {
        schema_for!(T::ARGUMENTS)
    }
    fn strict(&self) -> bool {
        T::STRICT
    }

    fn run(
        &self,
        arguments: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String, LLMYError>> + Send + '_>> {
        Box::pin(async move {
            match serde_json::from_value::<T::ARGUMENTS>(arguments.clone()) {
                Ok(args) => self.invoke(args).await,
                Err(_) => Err(LLMYError::IncorrectToolCall(
                    T::NAME.to_string(),
                    arguments.to_string(),
                    schema_for!(T::ARGUMENTS),
                )),
            }
        })
    }

    fn validate(
        &self,
        arguments: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + '_>> {
        Box::pin(async move {
            match serde_json::from_value::<T::ARGUMENTS>(arguments) {
                Ok(args) => Tool::validate(self, args).await,
                // The loop gates arguments on [`ToolDyn::schema`] before
                // calling validate, so a failed parse here means schemars and
                // serde disagree about an edge of the type. Theoretically
                // unreachable; abstain and let the execution path report it.
                Err(e) => {
                    tracing::error!(
                        "validate for {} got arguments its schema admits but its type refuses: {}",
                        T::NAME,
                        e
                    );
                    Ok(())
                }
            }
        })
    }
}

/// A name-keyed registry of tools available to an agent.
///
/// `ToolBox` owns its tools behind `Arc<Box<dyn ToolDyn>>`, so cloning the
/// box is cheap and the same set of tools can be shared across concurrent
/// invocations. Tools are stored in a [`BTreeMap`], so iteration order is
/// stable and sorted by name.
#[derive(Default, Clone, Debug)]
pub struct ToolBox {
    tools: BTreeMap<String, Arc<Box<dyn ToolDyn>>>,
}

impl ToolBox {
    /// Creates an empty `ToolBox`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Phase-one batch gate: runs every call's [`ToolDyn::validate`] before
    /// anything executes, so a rejection costs zero side effects across the
    /// whole turn. The first rejection comes back as
    /// [`LLMYError::ToolCallRejected`] bound to its wire call. Unknown tools,
    /// unparseable arguments and arguments that fail the tool's own
    /// [`ToolDyn::schema`] are skipped, not rejected — schema problems keep
    /// their existing `IncorrectToolCall` soft path in the execution phase —
    /// so `validate` only ever runs on conforming arguments.
    pub async fn validate_calls(&self, calls: &[GeneralToolCall]) -> Result<(), LLMYError> {
        for call in calls {
            let Some(tool) = self.tools.get(&call.tool_name) else {
                continue;
            };
            let Ok(arguments) = serde_json::from_str::<serde_json::Value>(&call.tool_args) else {
                continue;
            };
            // Gate on the schema the tool advertises to the model
            // ([`ToolDyn::schema`]), upholding validate's assumption that its
            // arguments conform. An uncompilable schema cannot gate anything
            // and falls through to validate's own abstention.
            let schema =
                serde_json::to_value(tool.schema()).unwrap_or(serde_json::Value::Bool(true));
            let conforms = jsonschema::validator_for(&schema)
                .map(|validator| validator.is_valid(&arguments))
                .unwrap_or(true);
            if !conforms {
                continue;
            }
            if let Err(reason) = tool.validate(arguments).await {
                return Err(LLMYError::ToolCallRejected(call.clone(), reason));
            }
        }
        Ok(())
    }

    /// Renders the registered tool names, optionally with their descriptions.
    ///
    /// When `details` is `true` each entry is formatted as
    /// `` `name`: "description" ``; otherwise only the bare name is returned.
    /// Useful when surfacing the tool list inside a system prompt.
    pub fn render_tools(&self, details: bool) -> Vec<String> {
        self.tools
            .iter()
            .map(|(name, tool)| {
                if details {
                    format!(
                        "`{}`: {:?}", // description may contain new lines
                        name,
                        tool.description()
                            .unwrap_or_else(|| "no description is provided".to_string())
                    )
                } else {
                    name.clone()
                }
            })
            .collect()
    }

    /// Merges another `ToolBox` into `self`. Tools in `rhs` overwrite any
    /// existing entries that share a name.
    pub fn extend(&mut self, rhs: Self) {
        self.tools.extend(rhs.tools.into_iter());
    }

    /// Returns whether a tool with the given name is registered.
    pub fn has_tool(&self, tool: &String) -> bool {
        self.tools.contains_key(tool)
    }

    /// Renders every registered tool as an MCP [`rmcp::model::Tool`]
    /// descriptor.
    pub fn mcp_tools(&self) -> Vec<rmcp::model::Tool> {
        self.tools.values().map(|t| t.to_mcp_tool()).collect()
    }

    /// Renders every registered tool as an OpenAI `ChatCompletionTools`
    /// entry, ready to be attached to a chat completion request.
    pub fn openai_objects(&self) -> Vec<ChatCompletionTools> {
        self.tools
            .iter()
            .map(|t| WithOtherFields::new(ChatCompletionToolsRaw::Function(t.1.to_openai_obejct())))
            .collect()
    }

    /// Registers a typed [`Tool`]. Equivalent to boxing it and calling
    /// [`Self::add_dyn_tool`].
    pub fn add_tool<T: Tool + 'static>(&mut self, tool: T) {
        self.add_dyn_tool(Box::new(tool) as _);
    }

    /// Registers an already-erased [`ToolDyn`]. The tool's
    /// [`ToolDyn::name`] is used as the registry key, so adding a tool whose
    /// name collides with an existing one will replace the previous entry.
    pub fn add_dyn_tool(&mut self, tool: Box<dyn ToolDyn>) {
        self.tools.insert(tool.name(), Arc::new(tool));
    }

    /// Removes the tool registered under `name`, returning `true` if one was
    /// present. The inverse of [`Self::add_dyn_tool`].
    pub fn remove_tool(&mut self, name: &str) -> bool {
        self.tools.remove(name).is_some()
    }

    /// Invokes a single tool by name with the given JSON-encoded arguments.
    ///
    /// Returns `None` if no tool with that name is registered. Otherwise
    /// returns `Some` with the tool's result (or an [`LLMYError`] from
    /// argument parsing or the tool itself).
    pub async fn invoke(
        &self,
        tool_name: String,
        arguments: String,
    ) -> Option<Result<String, LLMYError>> {
        if let Some(tool) = self.tools.get(&tool_name) {
            debug!("Invoking tool {} with arguments {}", &tool_name, &arguments);
            Some(tool.call(arguments).await)
        } else {
            None
        }
    }

    pub async fn invoke_value(
        &self,
        tool_name: String,
        arguments: serde_json::Value,
    ) -> Option<Result<String, LLMYError>> {
        if let Some(tool) = self.tools.get(&tool_name) {
            debug!("Invoking tool {} with arguments {}", &tool_name, &arguments);
            Some(tool.run(arguments).await)
        } else {
            None
        }
    }

    /// Concurrently invokes every call in `calls`, spawning each one onto a
    /// [`tokio::task::JoinSet`].
    ///
    /// Each result is paired with the original [`GeneralToolCall`] so the
    /// caller can correlate it back to a specific invocation. Use
    /// [`Self::invoke_many_sequential`] when ordering matters or when tools
    /// must not run in parallel.
    pub async fn invoke_many(
        &self,
        calls: Vec<GeneralToolCall>,
    ) -> Vec<(GeneralToolCall, Option<Result<String, LLMYError>>)> {
        let mut js = JoinSet::new();
        for call in calls {
            let tb = self.clone();
            js.spawn(async move {
                let tc: GeneralToolCall = call.clone();
                tracing::info!("Calling {}", &tc);
                (tc, tb.invoke(call.tool_name, call.tool_args).await)
            });
        }

        js.join_all().await
    }

    /// Sequentially invokes every call in `calls`, awaiting each one before
    /// starting the next. Preserves input order and avoids any concurrency
    /// between tools — pick this over [`Self::invoke_many`] when tools share
    /// non-`Sync` state or must observe one another's side effects.
    pub async fn invoke_many_sequential(
        &self,
        calls: Vec<GeneralToolCall>,
    ) -> Vec<(GeneralToolCall, Option<Result<String, LLMYError>>)> {
        let mut out = Vec::with_capacity(calls.len());

        for call in calls {
            let tc: GeneralToolCall = call.clone();
            tracing::debug!("Calling {}", &tc);
            out.push((tc, self.invoke(call.tool_name, call.tool_args).await));
        }

        out
    }

    /// Concurrent variant of [`Self::invoke_many`] that wraps each successful
    /// result in a [`ChatCompletionRequestMessage`] (a tool message tagged
    /// with the originating `tool_id`), ready to be appended to a
    /// conversation history.
    pub async fn agent_invoke_many(
        &self,
        calls: Vec<GeneralToolCall>,
    ) -> Vec<(
        GeneralToolCall,
        Option<Result<ChatCompletionRequestMessageRaw, LLMYError>>,
    )> {
        let invokes = self.invoke_many(calls).await;
        Self::agent_messages_from_invokes(invokes)
    }

    /// Sequential variant of [`Self::agent_invoke_many`].
    pub async fn agent_invoke_many_sequential(
        &self,
        calls: Vec<GeneralToolCall>,
    ) -> Vec<(
        GeneralToolCall,
        Option<Result<ChatCompletionRequestMessageRaw, LLMYError>>,
    )> {
        let invokes = self.invoke_many_sequential(calls).await;
        Self::agent_messages_from_invokes(invokes)
    }

    fn agent_messages_from_invokes(
        invokes: Vec<(GeneralToolCall, Option<Result<String, LLMYError>>)>,
    ) -> Vec<(
        GeneralToolCall,
        Option<Result<ChatCompletionRequestMessageRaw, LLMYError>>,
    )> {
        let mut out = vec![];
        for (call, result) in invokes {
            let id = call.tool_id.clone();
            let result = result.map(|v| {
                v.map(|s| {
                    let tool_msg = ChatCompletionRequestToolMessageRaw {
                        content: ChatCompletionRequestToolMessageContent::Text(s),
                        tool_call_id: id,
                    };
                    ChatCompletionRequestMessageRaw::Tool(WithOtherFields::new(tool_msg))
                })
            });

            out.push((call, result));
        }

        out
    }
}
