use llmy::agent::Tool as LlmyTool;
use llmy_agent::Tool as LlmyAgentTool;
use llmy_agent_derive::tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema, Default)]
struct EchoArgs {
    value: String,
}

#[derive(Debug, Clone)]
#[tool(
    description = "Echo the provided string.",
    arguments = EchoArgs,
    name = "echo_text",
    invoke = echo
)]
struct EchoTool;

impl EchoTool {
    async fn echo(&self, arguments: EchoArgs) -> Result<String, llmy_agent::LLMYError> {
        Ok(arguments.value)
    }
}

#[derive(Deserialize, JsonSchema, Default)]
struct DefaultNameArgs {
    count: usize,
}

#[derive(Debug, Clone)]
#[tool(arguments = DefaultNameArgs, invoke = count)]
struct DefaultNameTool;

impl DefaultNameTool {
    async fn count(&self, arguments: DefaultNameArgs) -> Result<String, llmy_agent::LLMYError> {
        Ok(arguments.count.to_string())
    }
}

#[tokio::test]
async fn generates_tool_impl_with_explicit_metadata() {
    let tool = EchoTool;

    assert_eq!(<EchoTool as LlmyAgentTool>::NAME, "echo_text");
    assert_eq!(EchoTool::DESCRIPTION, Some("Echo the provided string."));
    assert_eq!(
        tool.invoke(EchoArgs { value: "hi".into() }).await.unwrap(),
        "hi"
    );
    assert_eq!(
        tool.call(r#"{"value":"hello"}"#.into()).await.unwrap(),
        "hello"
    );

    let openai_object = tool.to_openai_obejct();
    assert_eq!(openai_object.function.name, "echo_text");
    assert_eq!(
        openai_object.function.description.as_deref(),
        Some("Echo the provided string.")
    );
    assert!(openai_object.function.parameters.is_some());
}

#[tokio::test]
async fn defaults_name_to_snake_case() {
    let tool = DefaultNameTool;

    assert_eq!(DefaultNameTool::NAME, "default_name_tool");
    assert_eq!(DefaultNameTool::DESCRIPTION, None);
    assert_eq!(tool.call(r#"{"count":3}"#.into()).await.unwrap(), "3");
}

#[tokio::test]
async fn supports_llmy_reexported_tool_trait() {
    let tool = DefaultNameTool;

    assert_eq!(<DefaultNameTool as LlmyTool>::NAME, "default_name_tool");
    assert_eq!(
        tool.invoke(DefaultNameArgs { count: 7 }).await.unwrap(),
        "7"
    );
}
