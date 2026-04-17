use std::io::Read;

use clap::Args;

#[derive(Args)]
pub struct TokenizerArgs {
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

pub fn run_tokenizer(args: TokenizerArgs) -> color_eyre::Result<()> {
    if args.list_models {
        for id in llmy_tokenizer::model_ids() {
            let enc = llmy_tokenizer::encoding_for_model(id)
                .map(|e| e.as_str())
                .unwrap_or("?");
            println!("{id}  ({enc})");
        }
        return Ok(());
    }

    let encoding = if let Some(ref model) = args.model {
        llmy_tokenizer::encoding_for_model(model)
            .ok_or_else(|| color_eyre::eyre::eyre!("Unknown model: {model}"))?
    } else if let Some(ref enc_name) = args.encoding {
        llmy_tokenizer::Encoding::from_str(enc_name)
            .ok_or_else(|| color_eyre::eyre::eyre!("Unknown encoding: {enc_name}"))?
    } else {
        color_eyre::eyre::bail!("Provide --model or --encoding");
    };

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
