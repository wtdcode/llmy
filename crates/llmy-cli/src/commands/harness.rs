use std::path::PathBuf;

use clap::Args;
use color_eyre::eyre::eyre;
use llmy_agent::tool::ToolBox;
use llmy_clap::OpenAISetup;
use llmy_codegraph::{CodeGraphBuilder, CodeGraphStore, CodegraphContext};
use llmy_harness_app::prompts::render_codegraph_section;
use llmy_harness_app::{
    HarnessOptions, HarnessRunStatus, HarnessRunner, MemoryOptions, SystemPromptTemplate,
    TaskConfig,
};

#[derive(Args)]
pub struct HarnessArgs {
    #[command(flatten)]
    openai: OpenAISetup,

    /// The task prompt, inline.
    #[arg(long, conflicts_with = "prompt_file")]
    prompt: Option<String>,

    /// Read the task prompt from a file ("-" for stdin).
    #[arg(long)]
    prompt_file: Option<PathBuf>,

    /// Root directory the run works on.
    #[arg(long, default_value = ".")]
    root: PathBuf,

    /// SQLite database holding run state (tool calls, background tasks).
    #[arg(long, default_value = "llmy-harness.sqlite3")]
    state_db: String,

    /// Hard cap on agent steps.
    #[arg(long, default_value_t = 300)]
    max_steps: u64,

    /// Tool results beyond this many characters are truncated in context
    /// (the full result stays readable via read_tool_output).
    #[arg(long, default_value_t = 20_000)]
    tool_result_chars: usize,

    /// Enable the bash / background-task / monitor tool family.
    #[arg(long, default_value_t = false)]
    bash: bool,

    /// Foreground window before a bash command moves to the background (ms).
    #[arg(long, default_value_t = 180_000)]
    bash_foreground_ms: u64,

    /// Enable scratchpad mode with this JSON file.
    #[arg(long)]
    scratchpad: Option<PathBuf>,

    /// Require structured output conforming to this JSON schema (a file
    /// path, or inline JSON starting with '{').
    #[arg(long)]
    output_schema: Option<String>,

    /// Nudges after a non-conforming stop before the run fails.
    #[arg(long, default_value_t = 3)]
    output_attempts: u64,

    /// Enable the knowledge-graph memory, persisted in this SQLite database.
    #[arg(long)]
    memory_db: Option<String>,

    /// Memory instruction text replacing the built-in default.
    #[arg(long, conflicts_with = "memory_instruction_file")]
    memory_instruction: Option<String>,

    /// Read the memory instruction from a file.
    #[arg(long)]
    memory_instruction_file: Option<PathBuf>,

    /// Refuse to finish until the run wrote memory (requires --memory-db).
    #[arg(long, default_value_t = false)]
    force_memory: bool,

    /// How many times the memory gate nudges before giving up.
    #[arg(long, default_value_t = 3)]
    force_memory_attempts: u64,

    /// Cap on the memory index rendered into the system prompt.
    #[arg(long, default_value_t = 8_000)]
    memory_index_chars: usize,

    /// Build the code graph over the root and expose the codegraph tools.
    #[arg(long, default_value_t = false)]
    codegraph: bool,

    /// Optional SQLite cache for the code graph (see `llmy codegraph index`).
    #[arg(long)]
    codegraph_db: Option<String>,

    /// Fixed prompt cache key for the conversation.
    #[arg(long, env = "LLM_HARNESS_CACHE_KEY")]
    cache_key: Option<String>,

    /// Run tool calls sequentially instead of concurrently.
    #[arg(long, default_value_t = false)]
    sequential_tools: bool,

    /// Extra text appended to the system prompt.
    #[arg(long)]
    append_system_prompt: Option<String>,

    /// Base system prompt template (currently: claude). Defaults to the
    /// harness's own prompt when omitted.
    #[arg(long)]
    system_prompt_template: Option<SystemPromptTemplate>,
}

impl HarnessArgs {
    async fn resolve_prompt(&self) -> color_eyre::Result<String> {
        match (&self.prompt, &self.prompt_file) {
            (Some(prompt), None) => Ok(prompt.clone()),
            (None, Some(path)) if path.as_os_str() == "-" => {
                let mut input = String::new();
                use std::io::Read;
                std::io::stdin().read_to_string(&mut input)?;
                Ok(input)
            }
            (None, Some(path)) => Ok(tokio::fs::read_to_string(path).await?),
            (Some(_), Some(_)) => Err(eyre!("--prompt and --prompt-file are mutually exclusive")),
            (None, None) => Err(eyre!("one of --prompt or --prompt-file is required")),
        }
    }

    async fn resolve_output_schema(&self) -> color_eyre::Result<Option<serde_json::Value>> {
        let Some(raw) = &self.output_schema else {
            return Ok(None);
        };
        let content = if raw.trim_start().starts_with('{') {
            raw.clone()
        } else {
            tokio::fs::read_to_string(raw).await?
        };
        Ok(Some(serde_json::from_str(&content)?))
    }

    async fn resolve_memory(&self) -> color_eyre::Result<Option<MemoryOptions>> {
        let Some(db_path) = &self.memory_db else {
            if self.force_memory {
                return Err(eyre!("--force-memory requires --memory-db"));
            }
            return Ok(None);
        };
        let instruction = match (&self.memory_instruction, &self.memory_instruction_file) {
            (Some(text), None) => Some(text.clone()),
            (None, Some(path)) => Some(tokio::fs::read_to_string(path).await?),
            (None, None) => None,
            (Some(_), Some(_)) => {
                return Err(eyre!(
                    "--memory-instruction and --memory-instruction-file are mutually exclusive"
                ));
            }
        };
        Ok(Some(MemoryOptions {
            db_path: db_path.clone(),
            instruction,
            force: self.force_memory,
            force_attempts: self.force_memory_attempts,
            index_max_chars: self.memory_index_chars,
        }))
    }

    /// Build (or load) the code graph and hand back its tools plus the
    /// system prompt section.
    async fn resolve_codegraph(
        &self,
        root: &PathBuf,
    ) -> color_eyre::Result<(ToolBox, Vec<String>)> {
        if !self.codegraph {
            return Ok((ToolBox::new(), vec![]));
        }
        let builder = CodeGraphBuilder::new(root.clone());
        let root_key = root.display().to_string();

        let graph = match &self.codegraph_db {
            Some(db_path) => {
                let store = CodeGraphStore::open(db_path).await?;
                let fingerprint = builder.fingerprint().await?;
                match store.load_fresh(&root_key, &fingerprint).await? {
                    Some(graph) => {
                        eprintln!("codegraph: loaded cached index ({})", graph.counts());
                        graph
                    }
                    None => {
                        let result = builder.build().await?;
                        store.save(&root_key, &result).await?;
                        eprintln!(
                            "codegraph: indexed {} files ({}), {} parse errors",
                            result.files.len(),
                            result.graph.counts(),
                            result.total_parse_errors()
                        );
                        result.graph
                    }
                }
            }
            None => {
                let result = builder.build().await?;
                eprintln!(
                    "codegraph: indexed {} files ({}), {} parse errors",
                    result.files.len(),
                    result.graph.counts(),
                    result.total_parse_errors()
                );
                result.graph
            }
        };

        let context = CodegraphContext::new(graph, root.clone());
        let section = render_codegraph_section(&context.render_overview());
        Ok((context.tool_box(), vec![section]))
    }
}

pub async fn run_harness(args: HarnessArgs) -> color_eyre::Result<()> {
    let prompt = args.resolve_prompt().await?;
    let root = if args.root.as_os_str() == "." {
        std::env::current_dir()?
    } else {
        args.root.canonicalize()?
    };

    let settings = args.openai.settings();
    let llm = args.openai.clone().to_llm().await;

    let output_schema = args.resolve_output_schema().await?;
    let memory = args.resolve_memory().await?;
    let (extra_tools, extra_sections) = args.resolve_codegraph(&root).await?;

    let options = HarnessOptions {
        root,
        state_db_path: args.state_db.clone(),
        prompt,
        max_steps: args.max_steps,
        tool_result_chars: args.tool_result_chars,
        bash: args.bash.then(|| TaskConfig {
            foreground_timeout_ms: args.bash_foreground_ms,
            ..TaskConfig::default()
        }),
        scratchpad: args.scratchpad.clone(),
        output_schema,
        output_attempts: args.output_attempts,
        memory,
        cache_key: args.cache_key.clone(),
        sequential_tools: args.sequential_tools,
        append_system_prompt: args.append_system_prompt.clone(),
        system_prompt_template: args.system_prompt_template.unwrap_or_default(),
    };

    let runner = HarnessRunner::prepare(options, &llm, extra_tools, extra_sections).await?;
    eprintln!("harness: run #{} started", runner.run_id());
    let outcome = runner.run(&llm, &settings, Some("harness")).await?;
    eprintln!(
        "harness: run #{} {} after {} steps",
        outcome.run_id,
        outcome.status.render(),
        outcome.steps
    );

    match (&outcome.structured, &outcome.final_text) {
        (Some(structured), _) => println!("{}", serde_json::to_string_pretty(structured)?),
        (None, Some(text)) => println!("{}", text),
        (None, None) => {}
    }

    if outcome.status != HarnessRunStatus::Completed {
        return Err(eyre!(
            "harness run #{} ended with status {}",
            outcome.run_id,
            outcome.status.render()
        ));
    }
    Ok(())
}
