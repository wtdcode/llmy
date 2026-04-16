# LLMY

All-in-one LLM utilities for Rust — plug OpenAI / Azure settings straight into [clap](https://crates.io/crates/clap), track spend with built-in billing, and replay every request when things go wrong.

## CLI

Install the command-line tool:

```bash
cargo install llmy-cli
```

### `llmy chat` — interactive chat

```bash
OPENAI_API_KEY=sk-... llmy chat --model gpt-4o
```

```
You: Explain async Rust in one sentence.
Assistant: Async Rust uses futures and an executor to let you write non-blocking,
concurrent code with zero-cost abstractions at compile time.
```

Supports `--system` for a custom system prompt. Reads from stdin when not a TTY.

### `llmy tokenizer` — count tokens offline

```bash
echo "Hello, world!" | llmy tokenizer --model openai/gpt-4o
# 4

llmy tokenizer --encoding cl100k_base --input my_prompt.txt --verbose
# 0  9906   "Hello"
# 1  11    ","
# 2  1917  " world"
# 3  0     "!"
# 4
```

### `llmy models` — list supported models

```
Model                           Input (per 1M)  Output (per 1M) Max Input  Max Output  Encoding
anthropic/claude-sonnet-4       $3.00           $15.00          136000     64000       claude
google/gemini-2.5-flash         $0.30           $2.50           936000     64000       o200k_base
google/gemini-2.5-pro           $1.25           $10.00          983040     65536       o200k_base
openai/gpt-4.1                  $2.00           $8.00           1014808    32768       o200k_base
openai/gpt-4o                   $2.50           $10.00          111616     16384       o200k_base
openai/gpt-4o-mini              $0.15           $0.60           111616     16384       o200k_base
openai/o1                       $15.00          $60.00          100000     100000      o200k_base
openai/o3                       $2.00           $8.00           100000     100000      o200k_base
openai/o4-mini                  $1.10           $4.40           100000     100000      o200k_base
…                               (112 models total)
```

## Library

Add the dependency (the root crate re-exports everything):

```toml
[dependencies]
llmy = "0.3"
```

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

`llmy` ships with up-to-date per-token pricing for 110+ models (GPT-4o, o1, o3, GPT-5 family, Claude, Gemini, …). Token usage is tracked in real-time including **cached-input** and **reasoning** token discounts. When spend exceeds the budget cap the client returns `LLMYError::Billing` immediately — no more surprise bills.

```rust
use llmy::client::{LLM, SupportedConfig};
use llmy::client::settings::LLMSettings;

let settings = LLMSettings::default();
let model = "gpt-4o".parse().unwrap();

let llm = LLM::new(
    SupportedConfig::new("https://api.openai.com/v1", "sk-..."),
    model,
    5.0, // budget cap in USD
    settings,
    None,
    None,
);

match llm.prompt_once("system", "user", None, None, None).await {
    Ok(resp) => { /* … */ }
    Err(llmy::LLMYError::Billing(cap, current)) => {
        eprintln!("Budget exceeded: ${:.4} / ${:.2}", current, cap);
    }
    Err(e) => { eprintln!("Error: {e}"); }
}
```

Via clap the cap defaults to **$10** and can be overridden:

```bash
cargo run -- --billing-cap 2.5 --model gpt-4o-mini
```

For models not in the built-in list, pass pricing inline:

```bash
cargo run -- --model "my-custom-model,1.0,4.0,0.5"
#                      name,         in, out, cached
```

---

### 4. Offline token estimation

`llmy` includes a built-in tokenizer with fast, offline BPE token estimation for 110+ models across OpenAI, Anthropic, Google, and more. Encodings and model metadata are baked into the binary at compile time — no network calls, no data files to ship.

Four encodings are supported: **cl100k_base**, **o200k_base**, **p50k_base** (OpenAI / tiktoken) and **claude** (Anthropic).

```rust
use llmy::tokenizer::{encode, count_tokens, count_tokens_for_model, Encoding};

// Encode text into token IDs
let tokens: Vec<u32> = encode("Hello, world!", Encoding::O200kBase);

// Count tokens directly
let n = count_tokens("Hello, world!", Encoding::Cl100kBase);

// Or let the library resolve the encoding from a model ID
let n = count_tokens_for_model("Hello, world!", "openai/gpt-4o"); // Some(4)
let n = count_tokens_for_model("Hello, world!", "anthropic/claude-sonnet-4"); // Some(4)
```

The model registry is generated from the same source-of-truth JSON used by the billing system, so model look-ups, pricing, and token counts always stay in sync.

---

### 5. Defining tools with `Tool` and `#[tool(...)]`

`llmy-agent` models callable tools as a Rust trait. A tool has a strongly typed argument struct, a stable tool name, an optional description, and an async `invoke` method that returns `Result<String, LLMYError>`.

You can depend on either the focused crate pair:

```toml
[dependencies]
llmy-agent = "0.4"
llmy-agent-derive = "0.4"
```

or the root crate plus the derive crate:

```toml
[dependencies]
llmy = "0.4"
llmy-agent-derive = "0.4"
```

The trait contract is:

```rust
use std::future::Future;

use llmy_agent::{LLMYError, Tool};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

pub trait Tool: Send + Sync + std::fmt::Debug {
    type ARGUMENTS: DeserializeOwned + JsonSchema + Send;
    const NAME: &str;
    const DESCRIPTION: Option<&str>;

    fn invoke(
        &self,
        arguments: Self::ARGUMENTS,
    ) -> impl Future<Output = Result<String, LLMYError>> + Send;
}
```

In practice you usually write the typed arguments and the async method, then let `llmy-agent-derive` generate the `impl Tool` for you:

```rust
use std::path::PathBuf;

use llmy::agent::LLMYError;
use llmy_agent_derive::tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema, Default)]
pub struct ReadFileArgs {
    /// The path of the file to read
    pub file_path: PathBuf,
}

#[derive(Debug, Clone)]
#[tool(
    arguments = ReadFileArgs,
    invoke = read_file,
    description = "Read file contents from `file_path`.",
    name = "read_file",
)]
pub struct ReadFileTool {
    pub cwd: PathBuf,
}

impl ReadFileTool {
    pub async fn read_file(&self, args: ReadFileArgs) -> Result<String, LLMYError> {
        let path = self.cwd.join(args.file_path);
        Ok(tokio::fs::read_to_string(path).await?)
    }
}
```

Notes:

- `arguments` and `invoke` are required in `#[tool(...)]`.
- `description` is optional.
- `name` is optional; if omitted, the struct name is converted to `snake_case`, for example `ReadFileTool -> read_file_tool`.
- The generated impl works with either `llmy_agent::Tool` or `llmy::agent::Tool`.

## License

MIT