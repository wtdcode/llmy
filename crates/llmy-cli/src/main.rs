use std::{io::IsTerminal, path::PathBuf};

use clap::{Parser, Subcommand};

mod commands;

use commands::{ChatArgs, TokenizerArgs, run_chat, run_models, run_tokenizer};

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
}

async fn main_entry() -> color_eyre::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Chat(args) => run_chat(args).await,
        Commands::Tokenizer(args) => run_tokenizer(args),
        Commands::Models => run_models(),
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
    if let Ok(dot_file) = std::env::var("DOT") {
        dotenvy::from_path(dot_file).expect("can not read dotenvy");
    } else {
        let direnv_exists = PathBuf::from(".envrc").exists();
        if !direnv_exists {
            // Allows failure
            let _ = dotenvy::dotenv();
        }
    }
    let sub = tracing_subscriber::FmtSubscriber::builder()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(tracing::Level::INFO.into())
                .from_env()
                .expect("env contains non-utf8"),
        )
        .with_ansi(use_colors)
        .finish();
    tracing::subscriber::set_global_default(sub).expect("can not set default tracing");
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("can not build tokio")
        .block_on(main_entry())
        .expect("ok")
}
