# LLMY

All-in-one LLM utilities for Rust — plug OpenAI / Azure settings straight into [clap](https://crates.io/crates/clap), track spend with built-in billing, replay every request when things go wrong, and bridge tools between your agent and the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP).

## Harnessing An Agent

The harness layer gives you a concrete in-memory `Agent` that can hold conversation state, expose tools to the model, and run a full user turn through any tool-call loop. A minimal coding agent only needs a system prompt, an `LLM`, and a `ToolBox` with the tools you want to expose.

The example below builds a basic agent that can read files, list directories, and search for files by glob pattern:

```toml
[dependencies]
clap = { version = "4", features = ["derive"] }
llmy = "0.13"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use std::path::PathBuf;

use clap::Parser;
use llmy::agent::tool::ToolBox;
use llmy::agent::tools::files::{FindFileTool, ListDirectoryTool, ReadFileTool};
use llmy::clap::OpenAISetup;
use llmy::harness::Agent;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    llm: OpenAISetup,

    #[arg(long, default_value = ".")]
    root: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), llmy::LLMYError> {
    let cli = Cli::parse();
    let settings = cli.llm.settings();
    let llm = cli.llm.to_llm().await;

    let mut tools = ToolBox::new();
    tools.add_tool(ReadFileTool::new(cli.root.clone()));
    tools.add_tool(ListDirectoryTool::new_root(cli.root.clone()));
    tools.add_tool(FindFileTool::new(cli.root.clone()));

    let mut agent = Agent::new(
        "You are a coding assistant. Use the available file tools whenever you need to inspect the workspace.".to_string(),
        tools,
        "readme-basic-agent".to_string(),
    );

    let result = agent
        .loop_step_user(
            "List the root directory, find Rust files under src, and then read Cargo.toml."
                .to_string(),
            &llm,
            Some("readme-basic-agent"),
            Some(settings),
        )
        .await?;

    if let Some(message) = result.assistant_message() {
        println!("{message}");
    }

    Ok(())
}
```

Run it with your OpenAI settings:

```bash
OPENAI_API_KEY=sk-... cargo run -- --model gpt-4o --root .
```

## MCP Support

`llmy` has first-class support for the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP), both as a server and as a client.

### MCP Server — expose a `ToolBox` as an MCP server

Any `ToolBox` can be served as an MCP server over stdio or Streamable HTTP:

```rust
use llmy::agent::tool::ToolBox;
use llmy::agent::mcp::McpToolBox;
use llmy::agent::tools::files::{FindFileTool, ListDirectoryTool, ReadFileTool};
use rmcp::model::{Implementation, ServerCapabilities, ServerInfo};
use rmcp::transport::StreamableHttpServerConfig;

let mut tools = ToolBox::new();
tools.add_tool(ReadFileTool::new(".".into()));
tools.add_tool(ListDirectoryTool::new_root(".".into()));
tools.add_tool(FindFileTool::new(".".into()));

let server_info = ServerInfo::new(
    ServerCapabilities::builder().enable_tools().build(),
).with_server_info(Implementation::new("my-server", "0.1.0"));

let server = McpToolBox::new(tools, server_info);

// Serve over stdio (for use with MCP clients like Claude Desktop):
// server.serve_stdio().await?;

// Or serve over HTTP:
// server.serve_http("127.0.0.1:3000", StreamableHttpServerConfig::default()).await?;
```

### MCP Client — connect to remote MCP servers

Connect to any MCP server and import its tools into a `ToolBox`:

```rust
use llmy::agent::tools::mcp::McpClient;

// Auto-detects HTTP vs stdio based on the URL scheme:
let client = McpClient::connect("http://127.0.0.1:3000").await?;
// Or for stdio: McpClient::connect("npx some-mcp-server").await?;

let remote_tools = client.to_toolbox().await?;

// Merge into your agent's toolbox:
tools.extend(remote_tools);
```

The client wraps each remote MCP tool as a `ToolDyn`, so they integrate seamlessly with the agent loop. MCP resources are also exposed as read-only tools.

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

#### Connecting to MCP servers

Use `--mcp-server` (repeatable) to connect to MCP servers and make their tools available to the agent:

```bash
# HTTP server
llmy chat --model gpt-4o --mcp-server http://127.0.0.1:3000

# Stdio server (command is split on whitespace)
llmy chat --model gpt-4o --mcp-server "npx some-mcp-server"

# Multiple servers
llmy chat --model gpt-4o \
    --mcp-server http://localhost:3000 \
    --mcp-server "npx another-server"
```

### `llmy mcp-server` — serve file tools over MCP

Run a built-in MCP server that exposes `read_file`, `list_dir`, and `find_file` tools:

```bash
# Over stdio (for Claude Desktop, etc.)
llmy mcp-server --root /path/to/project

# Over HTTP
llmy mcp-server --root . --listen 127.0.0.1:3005
```

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

### `llmy list-req` / `llmy dump-req` — inspect a SQLite `LLM_DEBUG` run

Browse the requests stored when `LLM_DEBUG` points at a SQLite database (see [Detailed debug logging](#2-detailed-debug-logging-llm_debug)).

```bash
# All requests of the latest client, with header
LLM_DEBUG=./run.sqlite3 llmy list-req

# Filter by client id and/or cache key
llmy list-req --db ./run.sqlite3 --client-id 2 --cache-key chat

# Show one request (human view: metadata + conversation)
llmy dump-req --db ./run.sqlite3 --req-id 17

# Same row as a single JSON object (LLMDebugRow)
llmy dump-req --req-id 17 --json
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
llmy = "0.13"
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

    // One-liner: clap args -> ready-to-use async LLM client
    let llm = cli.llm.to_llm().await;

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

Every setting (temperature, timeout, retries, max tokens, reasoning effort, tool choice, ...) is exposed as a flag **and** an env-var:

| Flag | Env var | Default |
|------|---------|---------|
| `--model` | `OPENAI_API_MODEL` | `o1` |
| `--llm-temperature` | `LLM_TEMPERATURE` | — |
| `--llm-presence-penalty` | `LLM_PRESENCE_PENALTY` | — |
| `--llm-max-completion-tokens` | `LLM_MAX_COMPLETION_TOKENS` | — |
| `--top-p` | `LLM_TOP_P` | — |
| `--llm-retry` | `LLM_RETRY` | `5` |
| `--llm-prompt-timeout` | `LLM_PROMPT_TIMEOUT` | `1200` (s) |
| `--llm-stream` | `LLM_STREAM` | `false` |
| `--reasoning-effort` | `LLM_REASONING_EFFORT` | — |

The second and third slots use the prefixes `OPT_` and `OPT_OPT_` for their env-vars (e.g. `OPT_OPENAI_API_KEY`, `OPT_OPT_OPENAI_API_MODEL`).

---

### 2. Detailed debug logging (`LLM_DEBUG`)

`LLM_DEBUG` accepts two backends, picked by the value's shape:

- **Folder** (any other path): one `.xml` + `.json` pair per round-trip, written under a per-process subfolder.
- **SQLite** (value starting with `sqlite3://` or ending in `sqlite3`): every request is a row in a long-lived database, queryable via the bundled CLI (`llmy list-req` / `llmy dump-req`) or any sqlite client.

#### Folder backend

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

#### SQLite backend

```bash
LLM_DEBUG=./run.sqlite3 OPENAI_API_KEY=sk-... cargo run
# or
LLM_DEBUG=sqlite3:///abs/path/to/run.sqlite3 ...
```

On startup a row is inserted into a `client` table and every request lands in `llm_debug` (model, endpoint URL, azure deployment, cache key, raw request/response JSON, per-request token counts, running USD spend, full conversation, timestamp). Inspect a run with the CLI:

```bash
# Most recent client only, optionally filtered by cache key
llmy list-req --cache-key chat
# id  client  ts                    model    endpoint                    deployment  cache_key  input  cached  output  reasoning  usage_usd  resp
# 1   3       2026-05-08 19:39:23   gpt-4o   https://api.openai.com/v1/  -           chat       142    0       38      0          0.001020   ok

# Drill into a single row (--json dumps the full LLMDebugRow struct)
llmy dump-req --req-id 1
llmy dump-req --req-id 1 --json
```

You can also embed the read API directly: `Sqlite3DebugDB::open_existing(url)` + `list_filtered` / `get_row` return strongly-typed `LLMDebugRow` values.

---

### 3. Built-in billing with automatic budget enforcement

`llmy` ships with up-to-date per-token pricing for 110+ models (GPT-4o, o1, o3, GPT-5 family, Claude, Gemini, ...). Token usage is tracked in real-time including **cached-input** and **reasoning** token discounts. When spend exceeds the budget cap the client returns `LLMYError::Billing` immediately — no more surprise bills.

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
    None, // Option<DebugBackend>; see LLM::new_async for LLM_DEBUG-style strings
);

match llm.prompt_once("system", "user", None, None, None).await {
    Ok(resp) => { /* ... */ }
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
llmy-agent = "0.13"
llmy-agent-derive = "0.13"
```

or the root crate plus the derive crate:

```toml
[dependencies]
llmy = "0.13"
llmy-agent-derive = "0.13"
```

The `Tool` trait defines the typed interface that users implement:

```rust
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

The companion `ToolDyn` trait provides the object-safe, type-erased interface used by `ToolBox` and the agent loop. A blanket `impl ToolDyn for T: Tool` bridges the two automatically — users only implement `Tool`.

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

## Cargo Features

### TLS backend

`rustls` is the default. Cargo features are additive, so selecting native-tls
means turning the default off rather than adding to it:

```toml
# rustls (default)
llmy = "0.18"

# native-tls, linking the system OpenSSL
llmy = { version = "0.18", default-features = false, features = ["native-tls"] }

# native-tls, compiling OpenSSL from source (static / musl-friendly)
llmy = { version = "0.18", default-features = false, features = ["native-tls-vendored"] }
```

Enabling both leaves both stacks linked in. It builds, but is almost certainly
not what you want.

The switch covers every TLS user in the graph: `async-openai` (the OpenAI HTTP
client), `rmcp` (the streamable-HTTP MCP client), and — when `embed` is on —
`hf-hub` and the `ort` binary download. `sqlx` is SQLite-only here and links no
TLS at all.

### Embeddings

Off by default, because they pull in `fastembed` + `ort-sys`, which has no musl
support.

| Feature | Effect |
| --- | --- |
| `embed` | Exposes `llmy::ebmed` (`SimilarityModel`, `Embeding`). No memory search. |
| `memory-embed-search` | Embedding-backed `search_memory` in `AgentMemoryContext`. Implies `embed`. |

To mix backends — say hf-hub over rustls but the ort download over native-tls —
use the granular toggles instead: `hf-hub-rustls-tls`, `hf-hub-native-tls`,
`ort-download-binaries-rustls-tls`, `ort-download-binaries-native-tls`.

> **Caveat.** `embed` links rustls in regardless of backend: `hf-hub` 0.5
> declares its `ureq` dependency without `default-features = false`, and
> `ureq`'s default is rustls, so no downstream feature can switch it off. If a
> rustls-free build is a hard requirement, leave `embed` off.

## License

MIT
