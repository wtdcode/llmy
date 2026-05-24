use std::future::Future;
use std::sync::Arc;

use llmy_types::error::LLMYError;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content, ListToolsResult, PaginatedRequestParams,
    ServerInfo,
};
use rmcp::transport::io::stdio;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::{ErrorData, ServerHandler, serve_server};

use crate::tool::ToolBox;

#[derive(Clone)]
pub struct McpToolBox {
    toolbox: ToolBox,
    server_info: ServerInfo,
}

impl McpToolBox {
    pub fn new(toolbox: ToolBox, server_info: ServerInfo) -> Self {
        Self {
            toolbox,
            server_info,
        }
    }

    pub async fn serve_stdio(self) -> Result<(), LLMYError> {
        let transport = stdio();
        let server = serve_server(self, transport).await?;
        server.waiting().await?;
        Ok(())
    }

    pub async fn serve_http(self, addr: impl tokio::net::ToSocketAddrs) -> Result<(), LLMYError> {
        let config = StreamableHttpServerConfig::default();
        let session_manager = Arc::new(LocalSessionManager::default());
        let service = StreamableHttpService::new(move || Ok(self.clone()), session_manager, config);
        let router = axum::Router::new().fallback_service(service);
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, router)
            .await
            .map_err(|e| LLMYError::IO(e.into()))?;
        Ok(())
    }

    fn to_mcp_tools(&self) -> Vec<rmcp::model::Tool> {
        self.toolbox.mcp_tools()
    }
}

impl ServerHandler for McpToolBox {
    fn get_info(&self) -> ServerInfo {
        self.server_info.clone()
    }

    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: rmcp::service::RequestContext<rmcp::service::RoleServer>,
    ) -> impl Future<Output = Result<ListToolsResult, ErrorData>> + Send + '_ {
        std::future::ready(Ok(ListToolsResult::with_all_items(self.to_mcp_tools())))
    }

    fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: rmcp::service::RequestContext<rmcp::service::RoleServer>,
    ) -> impl Future<Output = Result<CallToolResult, ErrorData>> + Send + '_ {
        async move {
            let name = request.name.to_string();
            let arguments = request
                .arguments
                .map(serde_json::Value::Object)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            tracing::info!("MCP server call_tool: {} {}", &name, &arguments);

            match self.toolbox.invoke_value(name, arguments).await {
                Some(Ok(result)) => Ok(CallToolResult::success(vec![Content::text(result)])),
                Some(Err(e)) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
                None => Err(ErrorData::invalid_params(
                    "tool not found",
                    Some(serde_json::json!({ "name": request.name })),
                )),
            }
        }
    }
}
