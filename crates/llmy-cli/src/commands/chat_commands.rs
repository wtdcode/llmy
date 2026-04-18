use clap::{Parser, Subcommand};
use llmy_client::{billing::ModelBilling, client::LLM, settings::LLMSettings};
use llmy_harness::Agent;

#[derive(Debug)]
pub(crate) enum ChatInput {
    User(String),
    Command(ChatCommand),
}

#[derive(Debug, Parser)]
#[command(name = "/", no_binary_name = true, disable_help_flag = true)]
struct ChatCommandLine {
    #[command(subcommand)]
    command: ChatCommand,
}

#[derive(Debug, Subcommand)]
pub(crate) enum ChatCommand {
    Compact,
    Context,
    Memory,
    Tokens,
}

pub(crate) fn parse_chat_input(input: &str) -> Result<ChatInput, String> {
    if !input.starts_with('/') {
        return Ok(ChatInput::User(input.to_string()));
    }

    let command_text = input.trim_start_matches('/');
    let Some(tokens) = shlex::split(command_text) else {
        return Err(
            "Failed to parse command line: unmatched quotes or escape sequence.".to_string(),
        );
    };

    if tokens.is_empty() {
        return Err("Empty command. Try /compact, /context, /memory, or /tokens.".to_string());
    }

    ChatCommandLine::try_parse_from(tokens)
        .map(|parsed| ChatInput::Command(parsed.command))
        .map_err(|error| error.to_string())
}

pub(crate) async fn run_chat_command(
    command: ChatCommand,
    agent: &mut Agent,
    llm: &LLM,
    debug_prefix: Option<&str>,
    settings: Option<LLMSettings>,
    is_tty: bool,
) -> color_eyre::Result<()> {
    match command {
        ChatCommand::Compact => {
            *agent = agent.compact(llm, debug_prefix, settings).await?;
            print_command_output("Conversation compacted.", is_tty);
        }
        ChatCommand::Context => {
            print_command_output(&agent.render_context(), is_tty);
        }
        ChatCommand::Memory => {
            let rendered = agent
                .render_memory()
                .await
                .unwrap_or_else(|| "Shared memory is not enabled for this chat.".to_string());
            print_command_output(&rendered, is_tty);
        }
        ChatCommand::Tokens => {
            let billing = llm.billing_snapshot().await;
            print_command_output(&format_token_usage(&billing), is_tty);
        }
    }

    Ok(())
}

fn print_command_output(output: &str, is_tty: bool) {
    if is_tty {
        println!("\n{}\n", output);
    } else {
        println!("{}", output);
    }
}

fn format_token_usage(billing: &ModelBilling) -> String {
    let uncached_input_tokens = billing.input_tokens.saturating_sub(billing.cache_tokens);
    let non_reasoning_output_tokens = billing
        .output_tokens
        .saturating_sub(billing.reasoning_tokens);

    [
        "Session token usage:".to_string(),
        format!(
            "input_tokens: total={} uncached={} cached={}",
            billing.input_tokens, uncached_input_tokens, billing.cache_tokens
        ),
        format!(
            "output_tokens: total={} response={} reasoning={}",
            billing.output_tokens, non_reasoning_output_tokens, billing.reasoning_tokens
        ),
        format!(
            "estimated_cost_usd: {:.4} / {}",
            billing.current, billing.cap
        ),
    ]
    .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmy_client::billing::ModelBilling;

    #[test]
    fn parse_chat_input_keeps_plain_user_text() {
        let parsed = parse_chat_input("hello world").unwrap();

        match parsed {
            ChatInput::User(text) => assert_eq!(text, "hello world"),
            ChatInput::Command(_) => panic!("expected plain user input"),
        }
    }

    #[test]
    fn parse_chat_input_parses_compact_command() {
        let parsed = parse_chat_input("/compact").unwrap();

        match parsed {
            ChatInput::Command(ChatCommand::Compact) => {}
            ChatInput::User(_) => panic!("expected compact command"),
            ChatInput::Command(other) => panic!("expected compact command, got {:?}", other),
        }
    }

    #[test]
    fn parse_chat_input_parses_context_command() {
        let parsed = parse_chat_input("/context").unwrap();

        match parsed {
            ChatInput::Command(ChatCommand::Context) => {}
            ChatInput::User(_) => panic!("expected context command"),
            ChatInput::Command(other) => panic!("expected context command, got {:?}", other),
        }
    }

    #[test]
    fn parse_chat_input_parses_memory_command() {
        let parsed = parse_chat_input("/memory").unwrap();

        match parsed {
            ChatInput::Command(ChatCommand::Memory) => {}
            ChatInput::User(_) => panic!("expected memory command"),
            ChatInput::Command(other) => panic!("expected memory command, got {:?}", other),
        }
    }

    #[test]
    fn parse_chat_input_parses_tokens_command() {
        let parsed = parse_chat_input("/tokens").unwrap();

        match parsed {
            ChatInput::Command(ChatCommand::Tokens) => {}
            ChatInput::User(_) => panic!("expected tokens command"),
            ChatInput::Command(other) => panic!("expected tokens command, got {:?}", other),
        }
    }

    #[test]
    fn format_token_usage_renders_cached_and_reasoning_breakdown() {
        let rendered = format_token_usage(&ModelBilling {
            input_tokens: 120,
            output_tokens: 45,
            cache_tokens: 20,
            reasoning_tokens: 5,
            current: 0.0123,
            cap: 10.0,
        });

        assert!(rendered.contains("Session token usage:"));
        assert!(rendered.contains("input_tokens: total=120 uncached=100 cached=20"));
        assert!(rendered.contains("output_tokens: total=45 response=40 reasoning=5"));
        assert!(rendered.contains("estimated_cost_usd: 0.0123 / 10"));
    }
}
