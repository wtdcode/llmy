use std::{
    io::{BufRead, IsTerminal},
    path::{Path, PathBuf},
};

use clap::Args;
use llmy_agent::{agent::Agent, tool::ToolBox};
use llmy_agent_tools::files::{FindFileTool, ListDirectoryTool, ReadFileTool, WriteFileTool};
use llmy_clap::OpenAISetup;
use llmy_client::client::LLM;
use rustyline::{DefaultEditor, error::ReadlineError};

#[derive(Args)]
pub struct ChatArgs {
    #[command(flatten)]
    openai: OpenAISetup,

    /// Optional system prompt
    #[arg(long)]
    system: Option<String>,

    /// Enable basic file tools for the agent. If omitted, plain chat mode is used.
    #[arg(long, value_name = "ROOT", num_args = 0..=1, default_missing_value = ".")]
    agent_files: Option<PathBuf>,
}

pub async fn run_chat(args: ChatArgs) -> color_eyre::Result<()> {
    let settings = args.openai.settings();
    let llm: LLM = args.openai.to_llm();
    let system = args
        .system
        .as_deref()
        .unwrap_or("You are a helpful assistant.");
    let files_root = resolve_files_root(args.agent_files)?;
    let system_prompt = build_system_prompt(system, files_root.as_deref());
    let tools = build_toolbox(files_root.clone());
    let mut agent = Agent::new(system_prompt, tools, "llmy-cli-chat".to_string());

    let stdin = std::io::stdin();
    let is_tty = stdin.is_terminal();

    let mut reader = ChatReader::new(is_tty)?;
    while let Some(input) = reader.read_next()? {
        agent
            .step_by_user(input, &llm, Some("chat"), Some(settings.clone()))
            .await?;

        while print_last_step(&agent, is_tty) {
            agent
                .step(&llm, Some("chat"), Some(settings.clone()))
                .await?;
        }
    }

    Ok(())
}

fn resolve_files_root(root: Option<PathBuf>) -> color_eyre::Result<Option<PathBuf>> {
    root.map(|path| {
        if path == Path::new(".") {
            std::env::current_dir().map_err(Into::into)
        } else {
            path.canonicalize().map_err(Into::into)
        }
    })
    .transpose()
}

fn build_toolbox(files_root: Option<PathBuf>) -> ToolBox {
    let mut toolbox = ToolBox::new();

    if let Some(root) = files_root {
        toolbox.add_tool(ReadFileTool::new(root.clone()));
        toolbox.add_tool(ListDirectoryTool::new_root(root.clone()));
        toolbox.add_tool(FindFileTool::new(root.clone()));
        toolbox.add_tool(WriteFileTool::new(root));
    }

    toolbox
}

fn build_system_prompt(base: &str, files_root: Option<&Path>) -> String {
    if let Some(root) = files_root {
        format!(
            "{base}\n\nYou can use sandboxed file tools rooted at {}. All tool paths must be relative to this root. Use the available file tools when you need to inspect or modify files.",
            root.display()
        )
    } else {
        base.to_string()
    }
}

fn print_last_step(agent: &Agent, is_tty: bool) -> bool {
    let last_step = agent
        .last_step()
        .as_ref()
        .expect("agent step completed without recording last_step");

    if let Some(msg) = last_step.assistant_message() {
        if is_tty {
            println!("\nAssistant: {}\n", msg);
        } else {
            println!("{}", msg);
        }
    }

    last_step.did_tool_call()
}

enum ChatReader {
    Interactive(DefaultEditor),
    Plain(std::io::Stdin),
}

impl ChatReader {
    fn new(is_tty: bool) -> color_eyre::Result<Self> {
        if is_tty {
            Ok(Self::Interactive(DefaultEditor::new()?))
        } else {
            Ok(Self::Plain(std::io::stdin()))
        }
    }

    fn read_next(&mut self) -> color_eyre::Result<Option<String>> {
        loop {
            match self {
                Self::Interactive(editor) => match editor.readline("You: ") {
                    Ok(line) => {
                        let input = line.trim();
                        if input.is_empty() {
                            continue;
                        }
                        let _ = editor.add_history_entry(input);
                        return Ok(Some(input.to_string()));
                    }
                    Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                        return Ok(None);
                    }
                    Err(error) => return Err(error.into()),
                },
                Self::Plain(stdin) => {
                    let mut input = String::new();
                    if stdin.lock().read_line(&mut input)? == 0 {
                        return Ok(None);
                    }

                    let input = input.trim();
                    if input.is_empty() {
                        continue;
                    }

                    return Ok(Some(input.to_string()));
                }
            }
        }
    }
}
