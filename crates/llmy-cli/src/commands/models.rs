pub fn run_models() -> color_eyre::Result<()> {
    let mut entries = llmy_tokenizer::models();
    entries.sort_by_key(|(id, _)| *id);

    println!(
        "{}\t{}\t{}\t{}\t{}\t{}",
        "Model", "Input (per 1M)", "Output (per 1M)", "Max Input", "Max Output", "Encoding"
    );
    for (id, config) in entries {
        let per_million = rust_decimal::dec!(1_000_000);
        let (input, output) = match config.pricing {
            Some(p) => (
                format!("${:.2}", p.input * per_million),
                format!("${:.2}", p.output * per_million),
            ),
            None => ("-".to_string(), "-".to_string()),
        };
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            id, input, output, config.max_input_tokens, config.max_tokens, config.encoding
        );
    }

    Ok(())
}
