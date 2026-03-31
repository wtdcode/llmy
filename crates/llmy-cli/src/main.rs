use std::{
    io::{BufRead, IsTerminal, Read, Write},
    path::PathBuf,
};

use clap::{Parser, Subcommand};
use llmy_clap::OpenAISetup;
use llmy_client::client::LLM;

use async_openai::types::chat::{
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs,
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
}

#[derive(Parser)]
struct ChatArgs {
    #[command(flatten)]
    openai: OpenAISetup,

    /// Optional system prompt
    #[arg(long)]
    system: Option<String>,
}

#[derive(Parser)]
struct TokenizerArgs {
    /// Input file to tokenize (reads from stdin if omitted)
    #[arg(long)]
    input: Option<String>,

    /// Model identifier (e.g. "openai/gpt-4o", "anthropic/claude-sonnet-4")
    #[arg(long)]
    model: Option<String>,

    /// Encoding name directly (cl100k_base, o200k_base, p50k_base, claude)
    #[arg(long)]
    encoding: Option<String>,

    /// Show per-token output instead of just a count
    #[arg(long, default_value_t = false)]
    verbose: bool,

    /// List all known model IDs
    #[arg(long, default_value_t = false)]
    list_models: bool,
}

fn run_tokenizer(args: TokenizerArgs) -> color_eyre::Result<()> {
    if args.list_models {
        for id in llmy_tokenizer::model_ids() {
            let enc = llmy_tokenizer::encoding_for_model(id)
                .map(|e| e.as_str())
                .unwrap_or("?");
            println!("{id}  ({enc})");
        }
        return Ok(());
    }

    // Determine encoding
    let encoding = if let Some(ref model) = args.model {
        llmy_tokenizer::encoding_for_model(model)
            .ok_or_else(|| color_eyre::eyre::eyre!("Unknown model: {model}"))?
    } else if let Some(ref enc_name) = args.encoding {
        llmy_tokenizer::Encoding::from_str(enc_name)
            .ok_or_else(|| color_eyre::eyre::eyre!("Unknown encoding: {enc_name}"))?
    } else {
        color_eyre::eyre::bail!("Provide --model or --encoding");
    };

    // Read input
    let text = if let Some(ref path) = args.input {
        std::fs::read_to_string(path)?
    } else {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        buf
    };

    let tokens = llmy_tokenizer::encode(&text, encoding);

    if args.verbose {
        let bpe = llmy_tokenizer::get_bpe(encoding);
        for (i, &tok) in tokens.iter().enumerate() {
            let decoded = bpe.decode(vec![tok]).unwrap_or_else(|_| format!("<{tok}>"));
            println!("{i}\t{tok}\t{decoded:?}");
        }
    }

    println!("{}", tokens.len());
    Ok(())
}

async fn run_chat(args: ChatArgs) -> color_eyre::Result<()> {
    let settings = args.openai.settings();
    let llm: LLM = args.openai.to_llm();
    let system = args
        .system
        .as_deref()
        .unwrap_or("You are a helpful assistant.");

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let is_tty = stdin.is_terminal();

    let sys_msg: ChatCompletionRequestMessage = ChatCompletionRequestSystemMessageArgs::default()
        .content(system)
        .build()?
        .into();
    let mut messages: Vec<ChatCompletionRequestMessage> = vec![sys_msg];

    loop {
        if is_tty {
            print!("You: ");
            stdout.flush()?;
        }

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        let user_msg: ChatCompletionRequestMessage =
            ChatCompletionRequestUserMessageArgs::default()
                .content(input)
                .build()?
                .into();
        messages.push(user_msg);

        let mut req = CreateChatCompletionRequestArgs::default();
        req.messages(messages.clone())
            .model(llm.model.to_string())
            .temperature(settings.llm_temperature)
            .presence_penalty(settings.llm_presence_penalty)
            .max_completion_tokens(settings.llm_max_completion_tokens);
        if let Some(ref effort) = settings.reasoning_effort {
            req.reasoning_effort(effort.0.clone());
        }
        let req = req.build()?;

        let resp = llm
            .complete_once_with_retry(
                &req,
                None,
                Some(settings.timeout()),
                Some(settings.llm_retry),
            )
            .await?;

        if let Some(choice) = resp.choices.first() {
            let content = choice.message.content.as_deref().unwrap_or("");
            if is_tty {
                println!("\nAssistant: {}\n", content);
            } else {
                println!("{}", content);
            }

            let assistant_msg: ChatCompletionRequestMessage =
                ChatCompletionRequestAssistantMessageArgs::default()
                    .content(content)
                    .build()?
                    .into();
            messages.push(assistant_msg);
        }
    }

    Ok(())
}

fn run_models() {
    let models = llmy_tokenizer::models();
    let mut entries: Vec<_> = models.iter().collect();
    entries.sort_by_key(|(id, _)| *id);

    println!(
        "{}\t{}\t{}\t{}\t{}",
        "Model", "Input (per 1M)", "Output (per 1M)", "Context Window", "Encoding"
    );
    for (id, config) in entries {
        let (input, output) = match config.pricing {
            Some(p) => (
                format!("${:.2}", p.input * 1_000_000.0),
                format!("${:.2}", p.output * 1_000_000.0),
            ),
            None => ("-".to_string(), "-".to_string()),
        };
        println!(
            "{}\t{}\t{}\t{}\t{}",
            id, input, output, config.context_window, config.encoding
        );
    }
}

async fn main_entry() -> color_eyre::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Chat(args) => run_chat(args).await,
        Commands::Tokenizer(args) => run_tokenizer(args),
        Commands::Models => {
            run_models();
            Ok(())
        }
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
