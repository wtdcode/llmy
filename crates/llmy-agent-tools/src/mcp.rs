use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use llmy_agent::tool::{ToolBox, ToolDyn};
use llmy_types::error::LLMYError;
use rmcp::model::{CallToolRequestParams, ReadResourceRequestParams};
use rmcp::serve_client;
use rmcp::service::{Peer, RoleClient, RunningService};

type McpPeer = Arc<Peer<RoleClient>>;

pub struct McpClient {
    _service: RunningService<RoleClient, ()>,
    peer: McpPeer,
}

impl McpClient {
    /// Auto-detect connection type: HTTP URLs use streamable HTTP, otherwise
    /// the string is split as a shell command for stdio transport.
    pub async fn connect(server: &str) -> Result<Self, LLMYError> {
        if server.starts_with("http://") || server.starts_with("https://") {
            Self::connect_http(server).await
        } else {
            let parts: Vec<&str> = server.split_whitespace().collect();
            let (command, args) = parts.split_first().ok_or_else(|| {
                LLMYError::IO(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "empty MCP server command",
                ))
            })?;
            Self::connect_stdio(command, args).await
        }
    }

    pub async fn connect_stdio(command: &str, args: &[&str]) -> Result<Self, LLMYError> {
        let mut cmd = tokio::process::Command::new(command);
        cmd.args(args);
        let process = rmcp::transport::TokioChildProcess::new(cmd)?;
        let service = serve_client((), process).await?;
        let peer = Arc::new(service.peer().clone());
        Ok(Self {
            _service: service,
            peer,
        })
    }

    pub async fn connect_http(url: &str) -> Result<Self, LLMYError> {
        let transport = rmcp::transport::StreamableHttpClientTransport::from_uri(url);
        let service = serve_client((), transport).await?;
        let peer = Arc::new(service.peer().clone());
        Ok(Self {
            _service: service,
            peer,
        })
    }

    pub async fn to_toolbox(&self) -> Result<ToolBox, LLMYError> {
        let mut toolbox = ToolBox::new();

        let tools = self.peer.list_all_tools().await?;
        for tool in tools {
            toolbox.add_dyn_tool(Box::new(McpRemoteTool {
                peer: self.peer.clone(),
                tool,
            }));
        }

        let resources = self.peer.list_all_resources().await?;
        for resource in resources {
            toolbox.add_dyn_tool(Box::new(McpResourceTool {
                peer: self.peer.clone(),
                resource,
            }));
        }

        Ok(toolbox)
    }
}

#[derive(Clone, Debug)]
struct McpRemoteTool {
    peer: McpPeer,
    tool: rmcp::model::Tool,
}

impl ToolDyn for McpRemoteTool {
    fn name(&self) -> String {
        self.tool.name.to_string()
    }

    fn description(&self) -> Option<String> {
        self.tool.description.as_ref().map(|d| d.to_string())
    }

    fn schema(&self) -> schemars::Schema {
        serde_json::from_value(serde_json::Value::Object((*self.tool.input_schema).clone()))
            .unwrap_or_default()
    }

    fn run(
        &self,
        arguments: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String, LLMYError>> + Send + '_>> {
        Box::pin(async move {
            tracing::info!("MCP client call_tool: {} {}", &self.tool.name, &arguments);
            let args = arguments.as_object().cloned();
            let mut params = CallToolRequestParams::new(self.tool.name.clone());
            params.arguments = args;
            let result = self.peer.call_tool(params).await?;
            Ok(result
                .content
                .iter()
                .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
                .collect::<Vec<_>>()
                .join("\n"))
        })
    }
}

#[derive(Clone, Debug)]
struct McpResourceTool {
    peer: McpPeer,
    resource: rmcp::model::Resource,
}

impl ToolDyn for McpResourceTool {
    fn name(&self) -> String {
        format!("read_resource_{}", self.resource.raw.name)
    }

    fn description(&self) -> Option<String> {
        Some(
            self.resource
                .raw
                .description
                .clone()
                .unwrap_or_else(|| format!("Read resource: {}", self.resource.raw.uri)),
        )
    }

    fn schema(&self) -> schemars::Schema {
        schemars::Schema::default()
    }

    fn run(
        &self,
        _arguments: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String, LLMYError>> + Send + '_>> {
        Box::pin(async move {
            tracing::info!("MCP client read_resource: {}", &self.resource.raw.uri);
            let params = ReadResourceRequestParams::new(self.resource.raw.uri.clone());
            let result = self.peer.read_resource(params).await?;
            let texts: Vec<String> = result
                .contents
                .into_iter()
                .map(|c| match c {
                    rmcp::model::ResourceContents::TextResourceContents { text, .. } => text,
                    rmcp::model::ResourceContents::BlobResourceContents { blob, .. } => blob,
                })
                .collect();
            Ok(texts.join("\n"))
        })
    }
}
