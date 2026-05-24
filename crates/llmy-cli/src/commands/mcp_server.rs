use std::path::{Path, PathBuf};

use clap::Args;
use llmy_agent::mcp::McpToolBox;
use llmy_agent::tool::ToolBox;
use llmy_agent_tools::files::{FindFileTool, ListDirectoryTool, ReadFileTool};
use rmcp::model::{Implementation, ServerCapabilities, ServerInfo};
use rmcp::transport::StreamableHttpServerConfig;

#[derive(Args)]
pub struct McpServerArgs {
    /// Root directory to serve files from.
    #[arg(long, default_value = ".")]
    root: PathBuf,

    /// Listen address. If omitted, serves over stdio.
    #[arg(long)]
    listen: Option<String>,
}

pub async fn run_mcp_server(args: McpServerArgs) -> color_eyre::Result<()> {
    let root = if args.root == Path::new(".") {
        std::env::current_dir()?
    } else {
        args.root.canonicalize()?
    };

    let mut toolbox = ToolBox::new();
    toolbox.add_tool(ReadFileTool::new(root.clone()));
    toolbox.add_tool(ListDirectoryTool::new_root(root.clone()));
    toolbox.add_tool(FindFileTool::new(root));

    let server_info =
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_server_info(
            Implementation::new("llmy-mcp-files", env!("CARGO_PKG_VERSION")),
        );

    let server = McpToolBox::new(toolbox, server_info);

    if let Some(addr) = args.listen {
        eprintln!("Listening on {addr}");
        server
            .serve_http(addr, StreamableHttpServerConfig::default())
            .await?;
    } else {
        server.serve_stdio().await?;
    }

    Ok(())
}
