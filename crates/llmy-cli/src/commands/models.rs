pub fn run_models() -> color_eyre::Result<()> {
    let models = llmy_tokenizer::models();
    let mut entries: Vec<_> = models.iter().collect();
    entries.sort_by_key(|(id, _)| *id);

    println!(
        "{}\t{}\t{}\t{}\t{}\t{}",
        "Model", "Input (per 1M)", "Output (per 1M)", "Max Input", "Max Output", "Encoding"
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
            "{}\t{}\t{}\t{}\t{}\t{}",
            id, input, output, config.max_input_tokens, config.max_tokens, config.encoding
        );
    }

    Ok(())
}
