# LLMY

All-in-one LLM utilities for Rust — plug OpenAI / Azure settings straight into [clap](https://crates.io/crates/clap), track spend with built-in billing, and replay every request when things go wrong.

## Quick start

Add the dependency (the root crate re-exports everything):

```toml
[dependencies]
llmy = "0.2"
```

## Features

### 1. Clap integration — up to 3 LLM slots

`llmy-clap` provides three generated arg structs (`OpenAISetup`, `OptOpenAISetup`, `OptOptOpenAISetup`) so you can wire one, two, or three LLMs into any clap-based CLI with zero boilerplate. Each slot is controlled by its own set of env-vars / flags, and can be converted to the core `LLM` client in one call.

```rust
use clap::Parser;
use llmy::clap::OpenAISetup;      // primary
use llmy::clap::OptOpenAISetup;   // optional secondary

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    llm: OpenAISetup,

    #[command(flatten)]
    fallback_llm: OptOpenAISetup,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // One-liner: clap args → ready-to-use async LLM client
    let llm = cli.llm.to_llm();

    let resp = llm
        .prompt_once_with_retry(
            "You are a helpful assistant.",
            "Explain async Rust in one sentence.",
            None,
            None,
        )
        .await
        .unwrap();

    println!("{}", resp.choices[0].message.content.as_deref().unwrap_or(""));
}
```

Run it:

```bash
# OpenAI
OPENAI_API_KEY=sk-... cargo run -- --model gpt-4o

# Azure
OPENAI_API_KEY=... cargo run -- \
    --azure-openai-endpoint https://my.openai.azure.com \
    --azure-deployment gpt-4o \
    --model gpt-4o
```

Every setting (temperature, timeout, retries, max tokens, reasoning effort, tool choice, …) is exposed as a flag **and** an env-var:

| Flag | Env var | Default |
|------|---------|---------|
| `--model` | `OPENAI_API_MODEL` | `o1` |
| `--llm-temperature` | `LLM_TEMPERATURE` | `0.8` |
| `--llm-max-completion-tokens` | `LLM_MAX_COMPLETION_TOKENS` | `16384` |
| `--llm-retry` | `LLM_RETRY` | `5` |
| `--llm-prompt-timeout` | `LLM_PROMPT_TIMEOUT` | `1200` (s) |
| `--llm-stream` | `LLM_STREAM` | `false` |
| `--reasoning-effort` | `LLM_REASONING_EFFORT` | — |

The second and third slots use the prefixes `OPT_` and `OPT_OPT_` for their env-vars (e.g. `OPT_OPENAI_API_KEY`, `OPT_OPT_OPENAI_API_MODEL`).

---

### 2. Detailed debug logging (`LLM_DEBUG`)

Point `LLM_DEBUG` at a directory and every LLM round-trip is saved as an XML-like `.xml` (not strict XML — just an easy-to-skim tagged format) **and** a raw `.json` — perfect for post-mortem debugging or dataset building.

```bash
LLM_DEBUG=./debug_logs OPENAI_API_KEY=sk-... cargo run
```

This creates a per-process subfolder with numbered files:

```
debug_logs/
└── 48291-0-main/
    ├── llm-000000000001.xml
    ├── llm-000000000001.json
    ├── llm-000000000002.xml
    └── llm-000000000002.json
```

The `.xml` file looks like:

```xml
=====================
<Request>
<SYSTEM>
You are a helpful assistant.
</SYSTEM>
<USER>
Explain async Rust in one sentence.
</USER>
<tool name="search", description="Search the web", strict=false>
{
  "type": "object",
  "properties": { "query": { "type": "string" } }
}
</tool>
</Request>
=====================
=====================
<Response>
<ASSISTANT>
Async Rust lets you write concurrent code ...
</ASSISTANT>
</Response>
=====================
```

The `.json` companion contains the full serialised `CreateChatCompletionRequest` / `CreateChatCompletionResponse` objects for programmatic analysis.

---

### 3. Built-in billing with automatic budget enforcement

`llmy` ships with up-to-date per-token pricing for 30+ models (GPT-4o, o1, o3, GPT-5 family, Gemini, …). Token usage is tracked in real-time including **cached-input** and **reasoning** token discounts. When spend exceeds the budget cap the client returns `LLMYError::Billing` immediately — no more surprise bills.

```rust
use llmy::client::{LLM, SupportedConfig};
use llmy::models::OpenAIModel;
use llmy::client::settings::LLMSettings;

let llm = LLM::new(
    SupportedConfig::new("https://api.openai.com/v1", "sk-..."),
    OpenAIModel::GPT4O,
    5.0, // budget cap in USD
    LLMSettings::default(),
    None,
    None,
);

match llm.prompt_once("system", "user", None, None).await {
    Ok(resp) => { /* … */ }
    Err(llmy::LLMYError::Billing(cap, current)) => {
        eprintln!("Budget exceeded: ${:.4} / ${:.2}", current, cap);
    }
    Err(e) => { eprintln!("Error: {e}"); }
}
```

Via clap the cap defaults to **$10** and can be overridden:

```bash
cargo run -- --biling-cap 2.5 --model gpt-4o-mini
```

A sample of the built-in pricing table (USD per 1 M tokens):

| Model | Input | Output | Cached input |
|-------|-------|--------|-------------|
| `gpt-4o` | 2.50 | 10.00 | 1.25 |
| `gpt-4o-mini` | 0.15 | 0.60 | 0.075 |
| `o1` | 15.00 | 60.00 | 7.50 |
| `o3` | 2.00 | 8.00 | 0.50 |
| `o4-mini` | 1.10 | 4.40 | 0.275 |
| `gpt-4.1` | 2.00 | 8.00 | 0.50 |

For models not in the list, pass pricing inline:

```bash
cargo run -- --model "my-custom-model,1.0,4.0,0.5"
#                      name,         in, out, cached
```

## License

MIT