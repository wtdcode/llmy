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
                // `schemars::Schema` is a transparent wrapper over
                // `serde_json::Value`, so this is a move, not a serialize.
                parameters: Some(self.schema().to_value()),
                strict: Some(self.strict()),
            }),
        })
    }
    /// Renders the tool as an MCP [`rmcp::model::Tool`] descriptor.
    fn to_mcp_tool(&self) -> rmcp::model::Tool {
        let input_schema = self.schema().to_value();
        let input_schema = input_schema.as_object().cloned().unwrap_or_default();
        rmcp::model::Tool::new_with_raw(
            self.name(),
            self.description().map(Into::into),
            Arc::new(input_schema),
        )
    }
    /// Phase-one gate the agent loop runs on every call of a turn before any
    /// tool executes, on the same parsed arguments the execution path uses.
    /// The arguments are guaranteed to conform to the tool's schema — the
    /// batch gate ([`ToolBox::validate_calls`]) has already discarded the
    /// turn as `IncorrectToolCall` for malformed calls before validate runs.
    /// `Err(reason)` rejects the call: the
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
/// The macro accepts `description`, `arguments`, `invoke` (required), an
/// optional `name` (defaulting to the struct identifier in `snake_case`),
/// an optional `validate` naming a method forwarded as [`Tool::validate`],
/// and an optional `strict` bool wiring [`Tool::STRICT`].
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
    /// mismatch never reaches here (the batch gate discards the turn as
    /// [`LLMYError::IncorrectToolCall`] first).
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

struct ToolEntryInner {
    tool: Box<dyn ToolDyn>,
    /// Compiled once from the advertised schema at registration instead of
    /// per call. Registration through [`ToolBox::add_dyn_tool`] refuses an
    /// uncompilable schema outright, so `None` only happens on the typed
    /// [`ToolBox::add_tool`] path in the freak case where `schema_for!`
    /// output still fails to compile (e.g. an exotic
    /// `#[schemars(regex(...))]` pattern) — schema validation then abstains
    /// for that tool.
    validator: Option<jsonschema::Validator>,
}

/// One registered tool plus its compiled schema validator, shared behind a
/// single `Arc` so cloning a [`ToolBox`] or wrapping a tool costs one
/// refcount bump.
#[derive(Clone)]
pub struct ToolEntry {
    inner: Arc<ToolEntryInner>,
}

impl Debug for ToolEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolEntry")
            .field("tool", &self.inner.tool.name())
            .field("has_validator", &self.inner.validator.is_some())
            .finish()
    }
}

impl ToolEntry {
    fn compile(tool: &dyn ToolDyn) -> Result<jsonschema::Validator, LLMYError> {
        jsonschema::validator_for(tool.schema().as_value()).map_err(|error| {
            color_eyre::eyre::eyre!(
                "schema of tool {} is not a valid JSON Schema document: {}",
                tool.name(),
                error
            )
            .into()
        })
    }

    /// Refuses a tool whose advertised schema does not compile.
    fn strict(tool: Box<dyn ToolDyn>) -> Result<Self, LLMYError> {
        let validator = Self::compile(tool.as_ref())?;
        Ok(Self {
            inner: Arc::new(ToolEntryInner {
                tool,
                validator: Some(validator),
            }),
        })
    }

    /// Registers the tool even when its schema does not compile; schema
    /// validation abstains for it and the failure is logged loudly. Only for
    /// typed tools, where this is not supposed to be reachable at all.
    fn lenient(tool: Box<dyn ToolDyn>) -> Self {
        let validator = match Self::compile(tool.as_ref()) {
            Ok(validator) => Some(validator),
            Err(error) => {
                tracing::error!(
                    "typed tool produced an uncompilable schema, validation abstains for it: {}",
                    error
                );
                None
            }
        };
        Self {
            inner: Arc::new(ToolEntryInner { tool, validator }),
        }
    }

    /// The registered tool.
    pub fn tool(&self) -> &dyn ToolDyn {
        self.inner.tool.as_ref()
    }

    fn validator(&self) -> Option<&jsonschema::Validator> {
        self.inner.validator.as_ref()
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
    tools: BTreeMap<String, ToolEntry>,
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

    /// Iterates over the registered tools as `(name, entry)` pairs in
    /// sorted name order. Useful for wrapping or inspecting every tool of a
    /// box (e.g. building a recording adapter around each one) — a cloned
    /// [`ToolEntry`] is a cheap shared handle to the tool.
    pub fn entries(&self) -> impl Iterator<Item = (&String, &ToolEntry)> {
        self.tools.iter()
    }

    /// Phase-one batch gate: runs before anything executes, so a failure
    /// here costs zero side effects across the whole turn. Two verdicts
    /// discard the model's turn:
    ///
    /// - a malformed call — arguments that don't parse as JSON or don't
    ///   conform to the tool's advertised [`ToolDyn::schema`] — comes back
    ///   as [`LLMYError::IncorrectToolCall`];
    /// - content the tool's own [`ToolDyn::validate`] refuses comes back as
    ///   [`LLMYError::ToolCallRejected`].
    ///
    /// Either way the agent loop drops the whole turn and re-asks the model
    /// from a clean context — nothing about the failed attempt leaks into
    /// what it sees next. Unknown tool names are skipped, not rejected: they
    /// keep their soft "tool not defined" result in the execution phase, so
    /// the model learns the actual roster instead of retrying blind.
    ///
    /// On success the arguments parsed here are returned (aligned with
    /// `calls`; `None` for unknown tools) so the execution phase can reuse
    /// them via [`Self::invoke_many_parsed`] instead of parsing the wire
    /// strings a second time. Schema checks run against the validator
    /// compiled once at registration ([`Self::add_dyn_tool`]).
    pub async fn validate_calls(
        &self,
        calls: &[GeneralToolCall],
    ) -> Result<Vec<Option<serde_json::Value>>, LLMYError> {
        let mut parsed = Vec::with_capacity(calls.len());
        for call in calls {
            let Some(entry) = self.tools.get(&call.tool_name) else {
                parsed.push(None);
                continue;
            };
            let Ok(arguments) = serde_json::from_str::<serde_json::Value>(&call.tool_args) else {
                return Err(LLMYError::IncorrectToolCall(
                    call.tool_name.clone(),
                    call.tool_args.clone(),
                    entry.tool().schema(),
                ));
            };
            // Gate on the schema the tool advertises to the model,
            // upholding validate's assumption that its arguments conform.
            // A tool whose schema did not compile has no validator and
            // abstains.
            if let Some(validator) = entry.validator()
                && !validator.is_valid(&arguments)
            {
                return Err(LLMYError::IncorrectToolCall(
                    call.tool_name.clone(),
                    call.tool_args.clone(),
                    entry.tool().schema(),
                ));
            }
            if let Err(reason) = entry.tool().validate(arguments.clone()).await {
                return Err(LLMYError::ToolCallRejected(call.clone(), reason));
            }
            parsed.push(Some(arguments));
        }
        Ok(parsed)
    }

    /// Renders the registered tool names, optionally with their descriptions.
    ///
    /// When `details` is `true` each entry is formatted as
    /// `` `name`: "description" ``; otherwise only the bare name is returned.
    /// Useful when surfacing the tool list inside a system prompt.
    pub fn render_tools(&self, details: bool) -> Vec<String> {
        self.tools
            .iter()
            .map(|(name, entry)| {
                if details {
                    format!(
                        "`{}`: {:?}", // description may contain new lines
                        name,
                        entry
                            .tool()
                            .description()
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
        self.tools
            .values()
            .map(|entry| entry.tool().to_mcp_tool())
            .collect()
    }

    /// Renders every registered tool as an OpenAI `ChatCompletionTools`
    /// entry, ready to be attached to a chat completion request.
    pub fn openai_objects(&self) -> Vec<ChatCompletionTools> {
        self.tools
            .values()
            .map(|entry| {
                WithOtherFields::new(ChatCompletionToolsRaw::Function(
                    entry.tool().to_openai_obejct(),
                ))
            })
            .collect()
    }

    /// Registers a typed [`Tool`]. Its schema comes from `schema_for!` on
    /// the `ARGUMENTS` type, which emits a valid JSON Schema document, so
    /// registration is infallible — unlike [`Self::add_dyn_tool`], whose
    /// schema can come from anywhere.
    pub fn add_tool<T: Tool + 'static>(&mut self, tool: T) {
        self.insert_entry(ToolEntry::lenient(Box::new(tool) as _));
    }

    /// Registers an already-erased [`ToolDyn`]. The tool's
    /// [`ToolDyn::name`] is used as the registry key, so adding a tool whose
    /// name collides with an existing one will replace the previous entry.
    /// The advertised schema is compiled into a validator here, once, so
    /// [`Self::validate_calls`] never recompiles it per call — and because
    /// an erased tool's schema can come from anywhere (an MCP server, a
    /// hand-written impl), a schema that is not a valid JSON Schema document
    /// refuses the tool.
    pub fn add_dyn_tool(&mut self, tool: Box<dyn ToolDyn>) -> Result<(), LLMYError> {
        self.insert_entry(ToolEntry::strict(tool)?);
        Ok(())
    }

    fn insert_entry(&mut self, entry: ToolEntry) {
        self.tools.insert(entry.tool().name(), entry);
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
        if let Some(entry) = self.tools.get(&tool_name) {
            debug!("Invoking tool {} with arguments {}", &tool_name, &arguments);
            Some(entry.tool().call(arguments).await)
        } else {
            None
        }
    }

    pub async fn invoke_value(
        &self,
        tool_name: String,
        arguments: serde_json::Value,
    ) -> Option<Result<String, LLMYError>> {
        if let Some(entry) = self.tools.get(&tool_name) {
            debug!("Invoking tool {} with arguments {}", &tool_name, &arguments);
            Some(entry.tool().run(arguments).await)
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
        self.invoke_many_parsed(calls.into_iter().map(|call| (call, None)).collect())
            .await
    }

    /// Concurrent invocation of `(call, parsed_args)` pairs. A `Some` value
    /// — the arguments [`Self::validate_calls`] already parsed — executes
    /// directly without re-parsing the wire string; a `None` falls back to
    /// parsing the call's argument string (unknown tools land here and keep
    /// their "not defined" result).
    pub async fn invoke_many_parsed(
        &self,
        calls: Vec<(GeneralToolCall, Option<serde_json::Value>)>,
    ) -> Vec<(GeneralToolCall, Option<Result<String, LLMYError>>)> {
        let mut js = JoinSet::new();
        for (call, parsed) in calls {
            let tb = self.clone();
            js.spawn(async move {
                tracing::info!("Calling {}", &call);
                let result = match parsed {
                    Some(value) => tb.invoke_value(call.tool_name.clone(), value).await,
                    None => {
                        tb.invoke(call.tool_name.clone(), call.tool_args.clone())
                            .await
                    }
                };
                (call, result)
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
        self.invoke_many_parsed_sequential(calls.into_iter().map(|call| (call, None)).collect())
            .await
    }

    /// Sequential variant of [`Self::invoke_many_parsed`].
    pub async fn invoke_many_parsed_sequential(
        &self,
        calls: Vec<(GeneralToolCall, Option<serde_json::Value>)>,
    ) -> Vec<(GeneralToolCall, Option<Result<String, LLMYError>>)> {
        let mut out = Vec::with_capacity(calls.len());

        for (call, parsed) in calls {
            tracing::debug!("Calling {}", &call);
            let result = match parsed {
                Some(value) => self.invoke_value(call.tool_name.clone(), value).await,
                None => {
                    self.invoke(call.tool_name.clone(), call.tool_args.clone())
                        .await
                }
            };
            out.push((call, result));
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

    /// [`Self::agent_invoke_many`] over `(call, parsed_args)` pairs, reusing
    /// the arguments [`Self::validate_calls`] already parsed.
    pub async fn agent_invoke_many_parsed(
        &self,
        calls: Vec<(GeneralToolCall, Option<serde_json::Value>)>,
    ) -> Vec<(
        GeneralToolCall,
        Option<Result<ChatCompletionRequestMessageRaw, LLMYError>>,
    )> {
        let invokes = self.invoke_many_parsed(calls).await;
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

    /// Sequential variant of [`Self::agent_invoke_many_parsed`].
    pub async fn agent_invoke_many_parsed_sequential(
        &self,
        calls: Vec<(GeneralToolCall, Option<serde_json::Value>)>,
    ) -> Vec<(
        GeneralToolCall,
        Option<Result<ChatCompletionRequestMessageRaw, LLMYError>>,
    )> {
        let invokes = self.invoke_many_parsed_sequential(calls).await;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A hand-implemented dyn tool advertising a schema that is not a valid
    /// JSON Schema document — the shape an ill-behaved MCP server produces.
    #[derive(Debug, Clone)]
    struct BadSchemaTool;

    impl ToolDyn for BadSchemaTool {
        fn name(&self) -> String {
            "bad_schema_tool".to_string()
        }

        fn description(&self) -> Option<String> {
            None
        }

        fn schema(&self) -> schemars::Schema {
            serde_json::from_str(r#"{"type": "no-such-type"}"#)
                .expect("the schema wrapper accepts any value")
        }

        fn run(
            &self,
            _arguments: serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = Result<String, LLMYError>> + Send + '_>> {
            Box::pin(async { Ok("ok".to_string()) })
        }
    }

    #[test]
    fn a_dyn_tool_with_an_invalid_schema_is_refused() {
        let mut tools = ToolBox::new();
        let error = tools
            .add_dyn_tool(Box::new(BadSchemaTool))
            .expect_err("invalid schema must refuse registration");
        assert!(error.to_string().contains("bad_schema_tool"), "{error}");
        assert_eq!(tools.len(), 0);
    }
}
