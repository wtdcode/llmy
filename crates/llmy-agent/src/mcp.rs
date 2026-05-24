use llmy_types::error::LLMYError;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content, ListToolsResult, PaginatedRequestParams,
    ServerInfo,
};
use rmcp::transport::io::stdio;
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

    pub async fn serve_network(
        self,
        addr: impl tokio::net::ToSocketAddrs,
    ) -> Result<(), LLMYError> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        loop {
            let (stream, _) = listener.accept().await?;
            let handler = self.clone();
            tokio::spawn(async move {
                let _ = serve_server(handler, stream).await;
            });
        }
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
