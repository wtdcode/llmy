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

use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionTool, ChatCompletionTools,
    FunctionObject,
};
use dyn_clone::DynClone;
use llmy_types::error::{GeneralToolCall, LLMYError};
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
    /// Renders the tool as an OpenAI [`ChatCompletionTool`] descriptor,
    /// including its JSON schema, ready to be sent in a chat completion
    /// request.
    fn to_openai_obejct(&self) -> ChatCompletionTool;
    /// Invokes the tool with raw JSON-encoded `arguments`. The string is
    /// deserialized into the tool's `ARGUMENTS` type by the blanket impl on
    /// top of [`Tool`].
    fn call(
        &self,
        arguments: String,
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

    /// Builds the OpenAI [`ChatCompletionTool`] descriptor for this tool.
    ///
    /// The default implementation uses [`Self::NAME`], [`Self::DESCRIPTION`]
    /// and a JSON schema generated from [`Self::ARGUMENTS`]. Override only if
    /// you need to customise the descriptor in ways the trait does not expose.
    fn to_openai_obejct(&self) -> ChatCompletionTool {
        ChatCompletionTool {
            function: FunctionObject {
                name: Self::NAME.to_string(),
                description: Self::DESCRIPTION.map(|e| e.to_string()),
                parameters: Some(
                    serde_json::to_value(schema_for!(Self::ARGUMENTS))
                        .expect("Fail to generate schema?!"),
                ),
                strict: Some(Self::STRICT),
            },
        }
    }
    /// Deserializes `arguments` from JSON and forwards to [`Self::invoke`].
    ///
    /// On parse failure the returned error is
    /// [`LLMYError::IncorrectToolCall`], carrying the tool name, the offending
    /// payload and the expected schema so the model can be re-prompted.
    fn call(&self, arguments: String) -> impl Future<Output = Result<String, LLMYError>> + Send {
        async move {
            match serde_json::from_str::<Self::ARGUMENTS>(&arguments) {
                Ok(args) => self.invoke(args).await,
                Err(_) => Err(LLMYError::IncorrectToolCall(
                    Self::NAME.to_string(),
                    arguments.clone(),
                    schema_for!(Self::ARGUMENTS),
                )),
            }
        }
    }

    /// Performs the tool's actual work on already-deserialized `arguments`
    /// and returns the textual result that will be sent back to the model.
    fn invoke(
        &self,
        arguments: Self::ARGUMENTS,
    ) -> impl Future<Output = Result<String, LLMYError>> + Send;
}

impl<T: Tool + DynClone + 'static> ToolDyn for T {
    fn name(&self) -> String {
        Self::NAME.to_string()
    }
    fn description(&self) -> Option<String> {
        Self::DESCRIPTION.map(|v| v.to_string())
    }
    fn call(
        &self,
        arguments: String,
    ) -> Pin<Box<dyn Future<Output = Result<String, LLMYError>> + Send + '_>> {
        Box::pin(self.call(arguments))
    }

    fn to_openai_obejct(&self) -> ChatCompletionTool {
        self.to_openai_obejct()
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

    /// Renders every registered tool as an OpenAI `ChatCompletionTools`
    /// entry, ready to be attached to a chat completion request.
    pub fn openai_objects(&self) -> Vec<ChatCompletionTools> {
        self.tools
            .iter()
            .map(|t| ChatCompletionTools::Function(t.1.to_openai_obejct()))
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
            tracing::info!("Calling {}", &tc);
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
        Option<Result<ChatCompletionRequestMessage, LLMYError>>,
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
        Option<Result<ChatCompletionRequestMessage, LLMYError>>,
    )> {
        let invokes = self.invoke_many_sequential(calls).await;
        Self::agent_messages_from_invokes(invokes)
    }

    fn agent_messages_from_invokes(
        invokes: Vec<(GeneralToolCall, Option<Result<String, LLMYError>>)>,
    ) -> Vec<(
        GeneralToolCall,
        Option<Result<ChatCompletionRequestMessage, LLMYError>>,
    )> {
        let mut out = vec![];
        for (call, result) in invokes {
            let id = call.tool_id.clone();
            let result = result.map(|v| {
                v.map(|s| {
                    ChatCompletionRequestToolMessage {
                        content: ChatCompletionRequestToolMessageContent::Text(s),
                        tool_call_id: id,
                    }
                    .into()
                })
            });

            out.push((call, result));
        }

        out
    }
}
