use std::path::PathBuf;

use clap::{Args, Subcommand};
use llmy_codegraph::{CodeGraphBuilder, CodeGraphStore, CodegraphContext};

#[derive(Args)]
pub struct CodegraphArgs {
    #[command(subcommand)]
    command: CodegraphCommand,
}

#[derive(Subcommand)]
enum CodegraphCommand {
    /// Build the code graph and store it in a SQLite cache.
    Index(IndexArgs),
    /// Build (or load) the code graph and print the module overview.
    Overview(OverviewArgs),
}

#[derive(Args)]
struct IndexArgs {
    /// Project root to index.
    #[arg(long, default_value = ".")]
    root: PathBuf,

    /// SQLite database to store the snapshot in.
    #[arg(long, default_value = "llmy-codegraph.sqlite3")]
    db: String,
}

#[derive(Args)]
struct OverviewArgs {
    /// Project root to index.
    #[arg(long, default_value = ".")]
    root: PathBuf,
}

pub async fn run_codegraph(args: CodegraphArgs) -> color_eyre::Result<()> {
    match args.command {
        CodegraphCommand::Index(index) => {
            let root = index.root.canonicalize()?;
            let builder = CodeGraphBuilder::new(root.clone());
            let result = builder.build().await?;
            let store = CodeGraphStore::open(&index.db).await?;
            store.save(&root.display().to_string(), &result).await?;
            println!(
                "indexed {} files into {}: {} ({} parse errors)",
                result.files.len(),
                store.path(),
                result.graph.counts(),
                result.total_parse_errors()
            );
        }
        CodegraphCommand::Overview(overview) => {
            let root = overview.root.canonicalize()?;
            let result = CodeGraphBuilder::new(root.clone()).build().await?;
            let context = CodegraphContext::new(result.graph, root);
            println!("{}", context.render_overview());
        }
    }
    Ok(())
}
