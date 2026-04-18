use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::{collections::HashMap, fmt::Debug};

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

pub trait ToolDyn: DynClone + Debug + Send + Sync + std::any::Any {
    fn name(&self) -> String;
    fn to_openai_obejct(&self) -> ChatCompletionTool;
    fn call(
        &self,
        arguments: String,
    ) -> Pin<Box<dyn Future<Output = Result<String, LLMYError>> + Send + '_>>;
}

pub fn downcast_tool<T: 'static>(tool: &dyn ToolDyn) -> &T {
    (tool as &dyn std::any::Any)
        .downcast_ref::<T>()
        .expect("can not downcast")
}

dyn_clone::clone_trait_object!(ToolDyn);

pub trait Tool: Send + Sync + DynClone + Debug {
    type ARGUMENTS: DeserializeOwned + schemars::JsonSchema + Sized + Send;
    const NAME: &str;
    const DESCRIPTION: Option<&str>;
    const STRICT: bool = false;

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

    fn invoke(
        &self,
        arguments: Self::ARGUMENTS,
    ) -> impl Future<Output = Result<String, LLMYError>> + Send;
}

impl<T: Tool + DynClone + 'static> ToolDyn for T {
    fn name(&self) -> String {
        Self::NAME.to_string()
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

#[derive(Default, Clone, Debug)]
pub struct ToolBox {
    tools: HashMap<String, Arc<Box<dyn ToolDyn>>>,
}

impl ToolBox {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn extend(&mut self, rhs: Self) {
        self.tools.extend(rhs.tools.into_iter());
    }

    pub fn has_tool(&self, tool: &String) -> bool {
        self.tools.contains_key(tool)
    }

    pub fn openai_objects(&self) -> Vec<ChatCompletionTools> {
        self.tools
            .iter()
            .map(|t| ChatCompletionTools::Function(t.1.to_openai_obejct()))
            .collect()
    }

    pub fn add_tool<T: Tool + 'static>(&mut self, tool: T) {
        self.add_dyn_tool(Box::new(tool) as _);
    }

    pub fn add_dyn_tool(&mut self, tool: Box<dyn ToolDyn>) {
        self.tools.insert(tool.name(), Arc::new(tool));
    }

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

    pub async fn agent_invoke_many(
        &self,
        calls: Vec<GeneralToolCall>,
    ) -> Result<Vec<(GeneralToolCall, ChatCompletionRequestMessage)>, LLMYError> {
        let invokes = self.invoke_many(calls).await;

        let mut out = vec![];
        for (call, result) in invokes {
            let result = result
                .unwrap()
                .map_err(|e| LLMYError::ToolCallError(call.clone(), Box::new(e)))?;
            let id = call.tool_id.clone();
            out.push((
                call,
                ChatCompletionRequestToolMessage {
                    content: ChatCompletionRequestToolMessageContent::Text(result),
                    tool_call_id: id,
                }
                .into(),
            ));
        }

        Ok(out)
    }
}
