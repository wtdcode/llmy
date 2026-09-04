use std::io::IsTerminal;

use clap::{Parser, Subcommand};

mod commands;

use commands::{
    ChatArgs, CodegraphArgs, DumpClientArgs, DumpReqArgs, HarnessArgs, ListReqArgs, McpServerArgs,
    TokenizerArgs, run_chat, run_codegraph, run_dump_client, run_dump_req, run_harness,
    run_list_req, run_mcp_server, run_models, run_tokenizer,
};

#[derive(Parser)]
#[command(name = "llmy", about = "All-in-one LLM utilities.")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Chat with a model interactively
    Chat(ChatArgs),
    /// Count tokens in text using various encodings
    Tokenizer(TokenizerArgs),
    /// List all supported models with pricing info
    Models,
    /// List recorded LLM debug requests from a sqlite3 LLM_DEBUG database
    ListReq(ListReqArgs),
    /// Dump a single LLM debug request from a sqlite3 LLM_DEBUG database
    DumpReq(DumpReqArgs),
    /// Dump every request of one client into XML conversations + raw JSONL
    DumpClient(DumpClientArgs),
    /// Run an MCP file server exposing read/list/find tools
    McpServer(McpServerArgs),
    /// Run a single-shot agent harness to completion over a workspace
    Harness(Box<HarnessArgs>),
    /// Build and inspect smart-contract code graphs (Solidity/Rust/Move)
    Codegraph(CodegraphArgs),
}

async fn main_entry() -> color_eyre::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Chat(args) => run_chat(args).await,
        Commands::Tokenizer(args) => run_tokenizer(args),
        Commands::Models => run_models(),
        Commands::ListReq(args) => run_list_req(args).await,
        Commands::DumpReq(args) => run_dump_req(args).await,
        Commands::DumpClient(args) => run_dump_client(args).await,
        Commands::McpServer(args) => run_mcp_server(args).await,
        Commands::Harness(args) => run_harness(*args).await,
        Commands::Codegraph(args) => run_codegraph(args).await,
    }
}

fn main() {
    let use_colors = std::io::stdout().is_terminal()
        && std::io::stderr().is_terminal()
        && std::env::var("NO_COLOR") == Err(std::env::VarError::NotPresent);
    if use_colors {
        color_eyre::install().expect("init color_eyre");
    } else {
        color_eyre::config::HookBuilder::new()
            .theme(color_eyre::config::Theme::new())
            .install()
            .expect("init no color color_eyre");
    }
    let direnv_exists = std::env::var("DIRENV_DIR").is_ok();
    if !direnv_exists {
        if let Ok(dot_file) = std::env::var("DOT") {
            dotenvy::from_path_override(dot_file).expect("can not read dotenvy");
        } else {
            // Allows failure and do not override
            let _ = dotenvy::dotenv();
        }
    }

    // Logs go to stderr so stdout stays clean for pipeable command output
    // (chat replies, structured harness results).
    let sub = tracing_subscriber::FmtSubscriber::builder()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(tracing::Level::INFO.into())
                .from_env()
                .expect("env contains non-utf8"),
        )
        .with_ansi(use_colors)
        .with_writer(std::io::stderr)
        .finish();
    tracing::subscriber::set_global_default(sub).expect("can not set default tracing");
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("can not build tokio")
        .block_on(main_entry())
        .expect("ok")
}
