use std::{
    fmt::Debug,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use tokio::io::AsyncWriteExt;

use color_eyre::eyre::eyre;
use itertools::Itertools;
use llmy_types::error::LLMYError;
use serde::{Deserialize, Serialize};
use sqlx::{
    Row,
    sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions},
};

use crate::billing::ModelBilling;
use crate::req::{
    ChatCompletionMessageToolCalls, ChatCompletionMessageToolCallsRaw,
    ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestAssistantMessageContentPartRaw,
    ChatCompletionRequestDeveloperMessageContent,
    ChatCompletionRequestDeveloperMessageContentPartRaw, ChatCompletionRequestMessage,
    ChatCompletionRequestMessageRaw, ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestSystemMessageContentPartRaw, ChatCompletionRequestToolMessageContent,
    ChatCompletionRequestToolMessageContentPartRaw, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPartRaw, ChatCompletionToolsRaw,
    CreateChatCompletionRequestRaw, RawExtensibleChatCompletionRequest,
};
use crate::resp::{ChatCompletionResponseMessage, RawExtensibleChatCompletionResponse};

pub fn completion_to_role(msg: &ChatCompletionRequestMessage) -> &'static str {
    match &msg.inner {
        ChatCompletionRequestMessageRaw::Assistant(_) => "ASSISTANT",
        ChatCompletionRequestMessageRaw::Developer(_) => "DEVELOPER",
        ChatCompletionRequestMessageRaw::Function(_) => "FUNCTION",
        ChatCompletionRequestMessageRaw::System(_) => "SYSTEM",
        ChatCompletionRequestMessageRaw::Tool(_) => "TOOL",
        ChatCompletionRequestMessageRaw::User(_) => "USER",
    }
}

pub fn toolcall_to_string(t: &ChatCompletionMessageToolCalls) -> String {
    match &t.inner {
        ChatCompletionMessageToolCallsRaw::Function(t) => {
            format!(
                "<toolcall name=\"{}\">\n{}\n</toolcall>",
                &t.function.name, &t.function.arguments
            )
        }
        ChatCompletionMessageToolCallsRaw::Custom(t) => {
            format!(
                "<customtoolcall name=\"{}\">\n{}\n</customtoolcall>",
                &t.custom_tool.name, &t.custom_tool.input
            )
        }
    }
}

pub fn response_to_string(resp: &ChatCompletionResponseMessage) -> String {
    let mut s = String::new();
    if let Some(content) = resp.content.as_ref() {
        s += content;
        s += "\n";
    }

    if let Some(tools) = resp.tool_calls.as_ref() {
        s += &tools.iter().map(toolcall_to_string).join("\n");
    }

    if let Some(refusal) = &resp.refusal {
        s += refusal;
        s += "\n";
    }

    let role = format!("{:?}", resp.role).to_uppercase();

    format!("<{}>\n{}\n</{}>\n", &role, s, &role)
}

#[allow(deprecated)]
pub fn completion_to_string(msg: &ChatCompletionRequestMessage) -> String {
    const CONT: &str = "<cont/>\n";
    const NONE: &str = "<none/>\n";
    let role = completion_to_role(msg);
    let content = match &msg.inner {
        ChatCompletionRequestMessageRaw::Assistant(ass) => {
            let msg = ass
                .content
                .as_ref()
                .map(|ass| match ass {
                    ChatCompletionRequestAssistantMessageContent::Text(s) => s.clone(),
                    ChatCompletionRequestAssistantMessageContent::Array(arr) => arr
                        .iter()
                        .map(|v| match &v.inner {
                            ChatCompletionRequestAssistantMessageContentPartRaw::Text(s) => {
                                s.text.clone()
                            }
                            ChatCompletionRequestAssistantMessageContentPartRaw::Refusal(rf) => {
                                rf.refusal.clone()
                            }
                        })
                        .join(CONT),
                })
                .unwrap_or(NONE.to_string());
            let tool_calls = ass
                .tool_calls
                .iter()
                .flatten()
                .map(toolcall_to_string)
                .join("\n");
            format!("{}\n{}", msg, tool_calls)
        }
        ChatCompletionRequestMessageRaw::Developer(dev) => match &dev.content {
            ChatCompletionRequestDeveloperMessageContent::Text(t) => t.clone(),
            ChatCompletionRequestDeveloperMessageContent::Array(arr) => arr
                .iter()
                .map(|v| match &v.inner {
                    ChatCompletionRequestDeveloperMessageContentPartRaw::Text(v) => v.text.clone(),
                })
                .join(CONT),
        },
        ChatCompletionRequestMessageRaw::Function(f) => {
            f.content.clone().unwrap_or(NONE.to_string())
        }
        ChatCompletionRequestMessageRaw::System(sys) => match &sys.content {
            ChatCompletionRequestSystemMessageContent::Text(t) => t.clone(),
            ChatCompletionRequestSystemMessageContent::Array(arr) => arr
                .iter()
                .map(|v| match &v.inner {
                    ChatCompletionRequestSystemMessageContentPartRaw::Text(t) => t.text.clone(),
                })
                .join(CONT),
        },
        ChatCompletionRequestMessageRaw::Tool(tool) => match &tool.content {
            ChatCompletionRequestToolMessageContent::Text(t) => t.clone(),
            ChatCompletionRequestToolMessageContent::Array(arr) => arr
                .iter()
                .map(|v| match &v.inner {
                    ChatCompletionRequestToolMessageContentPartRaw::Text(t) => t.text.clone(),
                })
                .join(CONT),
        },
        ChatCompletionRequestMessageRaw::User(usr) => match &usr.content {
            ChatCompletionRequestUserMessageContent::Text(t) => t.clone(),
            ChatCompletionRequestUserMessageContent::Array(arr) => arr
                .iter()
                .map(|v| match &v.inner {
                    ChatCompletionRequestUserMessageContentPartRaw::Text(t) => t.text.clone(),
                    ChatCompletionRequestUserMessageContentPartRaw::ImageUrl(img) => {
                        format!("<img url=\"{}\"/>", &img.image_url.url)
                    }
                    ChatCompletionRequestUserMessageContentPartRaw::InputAudio(audio) => {
                        format!("<audio>{}</audio>", audio.input_audio.data)
                    }
                    ChatCompletionRequestUserMessageContentPartRaw::File(f) => {
                        format!("<file>{:?}</file>", f)
                    }
                })
                .join(CONT),
        },
    };

    format!("<{}>\n{}\n</{}>\n", role, content, role)
}

pub(crate) async fn rewrite_json<T: Serialize + Debug>(
    fpath: &Path,
    t: &T,
) -> Result<(), LLMYError> {
    let mut json_fp = fpath.to_path_buf();
    json_fp.set_file_name(format!(
        "{}.json",
        json_fp
            .file_stem()
            .ok_or_else(|| eyre!("no filename"))?
            .to_str()
            .ok_or_else(|| eyre!("non-utf fname"))?
    ));

    let mut fp = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .write(true)
        .open(&json_fp)
        .await?;
    let s = match serde_json::to_string(&t) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("can not serialize due to {}", e);
            format!("{:?}", &t)
        }
    };
    fp.write_all(s.as_bytes()).await?;
    fp.write_all(b"\n").await?;
    fp.flush().await?;

    Ok(())
}

pub(crate) async fn save_llm_user(
    fpath: &PathBuf,
    request: &RawExtensibleChatCompletionRequest,
) -> Result<(), LLMYError> {
    let mut fp = tokio::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&fpath)
        .await?;
    fp.write_all(b"=====================\n<Request>\n").await?;
    for it in request.messages.iter() {
        let msg = completion_to_string(it);
        fp.write_all(msg.as_bytes()).await?;
    }

    let mut tools = vec![];
    for tool in request
        .tools
        .as_ref()
        .map(|t| t.iter())
        .into_iter()
        .flatten()
    {
        let s = match &tool.inner {
            ChatCompletionToolsRaw::Function(tool) => {
                format!(
                    "<tool name=\"{}\", description=\"{}\", strict={}>\n{}\n</tool>",
                    &tool.function.name,
                    &tool.function.description.clone().unwrap_or_default(),
                    tool.function.strict.unwrap_or_default(),
                    tool.function
                        .parameters
                        .as_ref()
                        .map(serde_json::to_string_pretty)
                        .transpose()?
                        .unwrap_or_default()
                )
            }
            ChatCompletionToolsRaw::Custom(tool) => {
                format!(
                    "<customtool name=\"{}\", description=\"{:?}\"></customtool>",
                    tool.custom.name, tool.custom.description
                )
            }
        };
        tools.push(s);
    }
    fp.write_all(tools.join("\n").as_bytes()).await?;
    fp.write_all(b"\n</Request>\n=====================\n")
        .await?;
    fp.flush().await?;

    rewrite_json(fpath, request).await?;

    Ok(())
}

pub(crate) fn extract_raw_text_with_other(request: &RawExtensibleChatCompletionRequest) -> String {
    let request = request.to_chat_completion_request();
    extract_raw_text(&request)
}

pub(crate) async fn save_llm_resp(
    fpath: &PathBuf,
    resp: &RawExtensibleChatCompletionResponse,
) -> Result<(), LLMYError> {
    let mut fp = tokio::fs::OpenOptions::new()
        .create(false)
        .append(true)
        .write(true)
        .open(&fpath)
        .await?;
    fp.write_all(b"=====================\n<Response>\n").await?;
    for it in &resp.choices {
        let msg = response_to_string(&it.message);
        fp.write_all(msg.as_bytes()).await?;
    }
    fp.write_all(b"\n</Response>\n=====================\n")
        .await?;
    fp.flush().await?;

    rewrite_json(fpath, resp).await?;

    Ok(())
}

/// Extract raw text from a chat completion request for token estimation.
/// Text content is extracted as-is; tool calls and tool definitions are
/// serialized to JSON so their token cost is still approximated.
#[allow(deprecated)]
pub fn extract_raw_text(req: &CreateChatCompletionRequestRaw) -> String {
    let mut parts: Vec<String> = Vec::new();

    for msg in &req.messages {
        match &msg.inner {
            ChatCompletionRequestMessageRaw::System(sys) => match &sys.content {
                ChatCompletionRequestSystemMessageContent::Text(t) => parts.push(t.clone()),
                ChatCompletionRequestSystemMessageContent::Array(arr) => {
                    for p in arr {
                        match &p.inner {
                            ChatCompletionRequestSystemMessageContentPartRaw::Text(t) => {
                                parts.push(t.text.clone())
                            }
                        }
                    }
                }
            },
            ChatCompletionRequestMessageRaw::User(usr) => match &usr.content {
                ChatCompletionRequestUserMessageContent::Text(t) => parts.push(t.clone()),
                ChatCompletionRequestUserMessageContent::Array(arr) => {
                    for p in arr {
                        match &p.inner {
                            ChatCompletionRequestUserMessageContentPartRaw::Text(t) => {
                                parts.push(t.text.clone())
                            }
                            _ => {}
                        }
                    }
                }
            },
            ChatCompletionRequestMessageRaw::Assistant(ass) => {
                if let Some(content) = &ass.content {
                    match content {
                        ChatCompletionRequestAssistantMessageContent::Text(t) => {
                            parts.push(t.clone())
                        }
                        ChatCompletionRequestAssistantMessageContent::Array(arr) => {
                            for p in arr {
                                match &p.inner {
                                    ChatCompletionRequestAssistantMessageContentPartRaw::Text(
                                        t,
                                    ) => parts.push(t.text.clone()),
                                    ChatCompletionRequestAssistantMessageContentPartRaw::Refusal(
                                        r,
                                    ) => parts.push(r.refusal.clone()),
                                }
                            }
                        }
                    }
                }
                if let Some(tcs) = &ass.tool_calls {
                    for tc in tcs {
                        if let Ok(s) = serde_json::to_string(tc) {
                            parts.push(s);
                        }
                    }
                }
            }
            ChatCompletionRequestMessageRaw::Tool(tool) => match &tool.content {
                ChatCompletionRequestToolMessageContent::Text(t) => parts.push(t.clone()),
                ChatCompletionRequestToolMessageContent::Array(arr) => {
                    for p in arr {
                        match &p.inner {
                            ChatCompletionRequestToolMessageContentPartRaw::Text(t) => {
                                parts.push(t.text.clone())
                            }
                        }
                    }
                }
            },
            ChatCompletionRequestMessageRaw::Developer(dev) => match &dev.content {
                ChatCompletionRequestDeveloperMessageContent::Text(t) => parts.push(t.clone()),
                ChatCompletionRequestDeveloperMessageContent::Array(arr) => {
                    for p in arr {
                        match &p.inner {
                            ChatCompletionRequestDeveloperMessageContentPartRaw::Text(t) => {
                                parts.push(t.text.clone())
                            }
                        }
                    }
                }
            },
            ChatCompletionRequestMessageRaw::Function(f) => {
                if let Some(c) = &f.content {
                    parts.push(c.clone());
                }
            }
        }
    }

    if let Some(tools) = &req.tools {
        for tool in tools {
            if let Ok(s) = serde_json::to_string(tool) {
                parts.push(s);
            }
        }
    }

    parts.join("\n")
}

/// Per-request metadata that is recorded on the debug row and is not derivable
/// from the chat completion request alone.
#[derive(Debug, Clone)]
pub struct DebugRowContext {
    pub model_name: String,
    pub endpoint: String,
    pub azure_deployment: Option<String>,
    pub cache_key: Option<String>,
    pub cap_usd: f64,
}

/// Per-request token usage recorded on a debug row alongside billing state.
#[derive(Debug, Clone)]
pub struct DebugUsage {
    pub input_without_cached_tokens: u64,
    pub cached_tokens: u64,
    pub output_without_reasoning_tokens: u64,
    pub reasoning_tokens: u64,
}

/// Returned by `DebugBackend::start`; opaque pointer to the row/file the
/// subsequent `record_*` calls should target.
#[derive(Debug, Clone)]
pub enum DebugHandle {
    Sqlite3 { row_id: i64 },
    Folder { fpath: PathBuf },
}

/// Folder-based debug backend: writes one `<prefix>-<idx>.xml` + `.json`
/// pair per request inside a unique subdirectory.
#[derive(Debug)]
pub struct FolderDebug {
    folder: PathBuf,
    index: AtomicU64,
}

impl FolderDebug {
    /// Build a folder backend rooted at `base`. A unique subdirectory of the
    /// form `<pid>-<n>-<prefix>` is created so concurrent processes do not
    /// clobber each other.
    pub fn new(base: &Path, debug_prefix: Option<&str>) -> Result<Self, LLMYError> {
        let pid = std::process::id();
        let prefix = match debug_prefix {
            Some(p) if !p.is_empty() => p.to_lowercase(),
            _ => "main".to_string(),
        };
        let mut cnt = 0u64;
        let folder = loop {
            let test_path = base.join(format!("{}-{}-{}", pid, cnt, prefix));
            if !test_path.exists() {
                std::fs::create_dir_all(&test_path)?;
                tracing::debug!("The path to save LLM interactions is {:?}", &test_path);
                break test_path;
            }
            cnt += 1;
        };
        Ok(Self {
            folder,
            index: AtomicU64::new(0),
        })
    }

    pub fn folder(&self) -> &Path {
        &self.folder
    }

    fn allocate_path(&self, debug_prefix: &str) -> PathBuf {
        let idx = self.index.fetch_add(1, Ordering::SeqCst);
        self.folder
            .join(format!("{}-{:0>12}.xml", debug_prefix, idx))
    }
}

/// Single LLM debug row as stored in `llm_debug`. Mirrors the table schema
/// 1:1 so external readers can decode rows without re-implementing the
/// query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMDebugRow {
    pub id: i64,
    pub client_id: i64,
    pub debug_prefix: String,
    pub model_name: String,
    pub endpoint: String,
    pub azure_deployment: Option<String>,
    pub cache_key: Option<String>,
    pub raw_req: String,
    pub raw_resp: Option<String>,
    pub input_without_cached_tokens: Option<i64>,
    pub cached_tokens: Option<i64>,
    pub output_without_reasoning_tokens: Option<i64>,
    pub reasoning_tokens: Option<i64>,
    pub resp_text_content: Option<String>,
    pub full_conversation: String,
    pub current_usage_usd: f64,
    pub cap_usd: f64,
    pub timestamp: i64,
}

impl LLMDebugRow {
    fn from_row(row: &sqlx::sqlite::SqliteRow) -> Result<Self, sqlx::Error> {
        Ok(Self {
            id: row.try_get("id")?,
            client_id: row.try_get("client_id")?,
            debug_prefix: row.try_get("debug_prefix")?,
            model_name: row.try_get("model_name")?,
            endpoint: row.try_get("endpoint")?,
            azure_deployment: row.try_get("azure_deployment")?,
            cache_key: row.try_get("cache_key")?,
            raw_req: row.try_get("raw_req")?,
            raw_resp: row.try_get("raw_resp")?,
            input_without_cached_tokens: row.try_get("input_without_cached_tokens")?,
            cached_tokens: row.try_get("cached_tokens")?,
            output_without_reasoning_tokens: row.try_get("output_without_reasoning_tokens")?,
            reasoning_tokens: row.try_get("reasoning_tokens")?,
            resp_text_content: row.try_get("resp_text_content")?,
            full_conversation: row.try_get("full_conversation")?,
            current_usage_usd: row.try_get("current_usage_usd")?,
            cap_usd: row.try_get("cap_usd")?,
            timestamp: row.try_get("timestamp")?,
        })
    }
}

const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS client (
    id INTEGER PRIMARY KEY AUTOINCREMENT
);
CREATE TABLE IF NOT EXISTS llm_debug (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id INTEGER NOT NULL,
    debug_prefix TEXT NOT NULL DEFAULT 'llm',
    model_name TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    azure_deployment TEXT,
    cache_key TEXT,
    raw_req TEXT NOT NULL,
    raw_resp TEXT,
    input_without_cached_tokens INTEGER,
    cached_tokens INTEGER,
    output_without_reasoning_tokens INTEGER,
    reasoning_tokens INTEGER,
    resp_text_content TEXT,
    full_conversation TEXT NOT NULL,
    current_usage_usd REAL NOT NULL,
    cap_usd REAL NOT NULL,
    timestamp INTEGER NOT NULL,
    FOREIGN KEY(client_id) REFERENCES client(id)
);
CREATE INDEX IF NOT EXISTS idx_llm_debug_cache_key ON llm_debug(cache_key);
CREATE INDEX IF NOT EXISTS idx_llm_debug_client_id ON llm_debug(client_id);
"#;

/// Best-effort migrations applied after `SCHEMA_SQL`. Each statement is
/// executed individually and its error is swallowed if it indicates the
/// migration is already in place (e.g. duplicate column).
const MIGRATIONS_SQL: &[&str] =
    &["ALTER TABLE llm_debug ADD COLUMN debug_prefix TEXT NOT NULL DEFAULT 'llm'"];

/// SQLite-backed debug store. [`Sqlite3DebugDB::open`] inserts a fresh
/// `client` row and binds the resulting id to this handle so all future
/// writes share that client. Read-only consumers should use
/// [`Sqlite3DebugDB::open_existing`] instead — it skips the client insert.
/// Clones share the same pool and id, so the LLM client and external
/// readers can coexist safely.
#[derive(Debug, Clone)]
pub struct Sqlite3DebugDB {
    pool: SqlitePool,
    client_id: Option<i64>,
    path: String,
}

impl Sqlite3DebugDB {
    /// Open or create the SQLite database at the given location. Accepts
    /// either a `sqlite3://<path>` URL or a bare path that ends in
    /// `sqlite3` — the latter is the common case (e.g.
    /// `LLM_DEBUG=./run.sqlite3`). Inserts a new `client` row to scope
    /// subsequent writes.
    pub async fn open(url: &str) -> Result<Self, LLMYError> {
        let (pool, path) = Self::open_pool(url).await?;
        let client_id = Self::insert_client(&pool).await?;
        Ok(Self {
            pool,
            client_id: Some(client_id),
            path,
        })
    }

    /// Open the SQLite database without allocating a new client row. Suitable
    /// for read-only consumers (CLI listing/dumping). The schema is still
    /// ensured so that the file is valid even if the writer never ran.
    pub async fn open_existing(url: &str) -> Result<Self, LLMYError> {
        let (pool, path) = Self::open_pool(url).await?;
        Ok(Self {
            pool,
            client_id: None,
            path,
        })
    }

    async fn open_pool(url: &str) -> Result<(SqlitePool, String), LLMYError> {
        let path = if let Some(rest) = url.strip_prefix("sqlite3://") {
            rest.to_string()
        } else {
            url.to_string()
        };

        // Catch the most common confusing failure: the path already exists but
        // is a directory (e.g. the old folder backend was previously pointed
        // at a `*.sqlite3` path and created a folder there).
        let fs_path = Path::new(&path);
        if fs_path.is_dir() {
            return Err(eyre!(
                "sqlite debug db path {} is a directory; remove it (or pick another LLM_DEBUG value) — sqlite needs a file",
                path
            )
            .into());
        }

        let opts = SqliteConnectOptions::new()
            .filename(&path)
            .create_if_missing(true)
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(opts)
            .await
            .map_err(|e| eyre!("failed to open sqlite debug db at {}: {}", path, e))?;

        for stmt in SCHEMA_SQL.split(';') {
            let trimmed = stmt.trim();
            if trimmed.is_empty() {
                continue;
            }
            sqlx::query(trimmed)
                .execute(&pool)
                .await
                .map_err(|e| eyre!("failed to apply sqlite debug schema: {}", e))?;
        }
        for migration in MIGRATIONS_SQL {
            if let Err(e) = sqlx::query(migration).execute(&pool).await {
                let msg = e.to_string();
                if msg.contains("duplicate column name") || msg.contains("already exists") {
                    continue;
                }
                return Err(eyre!(
                    "failed to apply sqlite debug migration {:?}: {}",
                    migration,
                    e
                )
                .into());
            }
        }
        Ok((pool, path))
    }

    /// Build a debug DB sharing an existing pool. The caller is responsible
    /// for ensuring the schema already exists.
    pub fn from_pool(pool: SqlitePool, client_id: Option<i64>) -> Self {
        Self {
            pool,
            client_id,
            path: String::new(),
        }
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    pub fn client_id(&self) -> Option<i64> {
        self.client_id
    }

    fn require_client_id(&self) -> Result<i64, LLMYError> {
        self.client_id
            .ok_or_else(|| eyre!("Sqlite3DebugDB has no client_id; opened read-only?").into())
    }

    /// The id of the most recently inserted client, or `None` if the table
    /// is empty.
    pub async fn latest_client_id(&self) -> Result<Option<i64>, LLMYError> {
        let row = sqlx::query("SELECT MAX(id) AS id FROM client")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| eyre!("failed to query latest client id: {}", e))?;
        Ok(row.try_get::<Option<i64>, _>("id").unwrap_or(None))
    }

    /// List rows optionally filtered by client and/or cache key, in
    /// ascending id order.
    pub async fn list_filtered(
        &self,
        client_id: Option<i64>,
        cache_key: Option<&str>,
    ) -> Result<Vec<LLMDebugRow>, LLMYError> {
        let mut sql = String::from("SELECT * FROM llm_debug WHERE 1=1");
        if client_id.is_some() {
            sql.push_str(" AND client_id = ?");
        }
        if cache_key.is_some() {
            sql.push_str(" AND cache_key = ?");
        }
        sql.push_str(" ORDER BY id ASC");

        let mut q = sqlx::query(&sql);
        if let Some(cid) = client_id {
            q = q.bind(cid);
        }
        if let Some(ck) = cache_key {
            q = q.bind(ck);
        }
        let rows = q
            .fetch_all(&self.pool)
            .await
            .map_err(|e| eyre!("failed to list llm_debug rows: {}", e))?;
        rows.iter()
            .map(LLMDebugRow::from_row)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| eyre!("failed to decode llm_debug rows: {}", e).into())
    }

    async fn insert_client(pool: &SqlitePool) -> Result<i64, LLMYError> {
        let row = sqlx::query("INSERT INTO client DEFAULT VALUES RETURNING id")
            .fetch_one(pool)
            .await
            .map_err(|e| eyre!("failed to insert debug client row: {}", e))?;
        let id: i64 = row
            .try_get("id")
            .map_err(|e| eyre!("client id missing: {}", e))?;
        Ok(id)
    }

    async fn insert_request(
        &self,
        debug_prefix: &str,
        ctx: &DebugRowContext,
        req: &RawExtensibleChatCompletionRequest,
    ) -> Result<i64, LLMYError> {
        let raw_req = serde_json::to_string(req)?;
        let full_conversation = format_request_conversation(req);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let client_id = self.require_client_id()?;
        let row = sqlx::query(
            r#"INSERT INTO llm_debug (
                client_id, debug_prefix, model_name, endpoint, azure_deployment, cache_key,
                raw_req, full_conversation, current_usage_usd, cap_usd, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id"#,
        )
        .bind(client_id)
        .bind(debug_prefix)
        .bind(&ctx.model_name)
        .bind(&ctx.endpoint)
        .bind(ctx.azure_deployment.as_deref())
        .bind(ctx.cache_key.as_deref())
        .bind(raw_req)
        .bind(full_conversation)
        .bind(0.0_f64)
        .bind(ctx.cap_usd)
        .bind(timestamp)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| eyre!("failed to insert llm_debug row: {}", e))?;
        let id: i64 = row
            .try_get("id")
            .map_err(|e| eyre!("row id missing: {}", e))?;
        Ok(id)
    }

    async fn update_response(
        &self,
        row_id: i64,
        req: &RawExtensibleChatCompletionRequest,
        resp: &RawExtensibleChatCompletionResponse,
    ) -> Result<(), LLMYError> {
        let raw_resp = serde_json::to_string(resp)?;
        let full_conversation = format_full_conversation(req, resp);
        let resp_text = first_choice_text(resp);
        sqlx::query(
            r#"UPDATE llm_debug
               SET raw_resp = ?, resp_text_content = ?, full_conversation = ?
               WHERE id = ?"#,
        )
        .bind(raw_resp)
        .bind(resp_text)
        .bind(full_conversation)
        .bind(row_id)
        .execute(&self.pool)
        .await
        .map_err(|e| eyre!("failed to update llm_debug row {}: {}", row_id, e))?;
        Ok(())
    }

    async fn update_billing(
        &self,
        row_id: i64,
        billing: &ModelBilling,
        usage: &DebugUsage,
    ) -> Result<(), LLMYError> {
        sqlx::query(
            r#"UPDATE llm_debug
               SET input_without_cached_tokens = ?,
                   cached_tokens = ?,
                   output_without_reasoning_tokens = ?,
                   reasoning_tokens = ?,
                   current_usage_usd = ?,
                   cap_usd = ?
               WHERE id = ?"#,
        )
        .bind(usage.input_without_cached_tokens as i64)
        .bind(usage.cached_tokens as i64)
        .bind(usage.output_without_reasoning_tokens as i64)
        .bind(usage.reasoning_tokens as i64)
        .bind(billing.current)
        .bind(billing.cap)
        .bind(row_id)
        .execute(&self.pool)
        .await
        .map_err(|e| eyre!("failed to update llm_debug billing for {}: {}", row_id, e))?;
        Ok(())
    }

    async fn update_error(&self, row_id: i64, error: &str) -> Result<(), LLMYError> {
        let payload = serde_json::to_string(&serde_json::json!({ "error": error }))?;
        sqlx::query("UPDATE llm_debug SET raw_resp = ? WHERE id = ?")
            .bind(payload)
            .bind(row_id)
            .execute(&self.pool)
            .await
            .map_err(|e| eyre!("failed to update llm_debug error for {}: {}", row_id, e))?;
        Ok(())
    }

    pub async fn get_row(&self, id: i64) -> Result<Option<LLMDebugRow>, LLMYError> {
        let row = sqlx::query("SELECT * FROM llm_debug WHERE id = ?")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| eyre!("failed to fetch llm_debug row {}: {}", id, e))?;
        row.map(|r| LLMDebugRow::from_row(&r))
            .transpose()
            .map_err(|e| eyre!("failed to decode llm_debug row {}: {}", id, e).into())
    }

    pub async fn list_by_client(&self, client_id: i64) -> Result<Vec<LLMDebugRow>, LLMYError> {
        let rows = sqlx::query("SELECT * FROM llm_debug WHERE client_id = ? ORDER BY id ASC")
            .bind(client_id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| {
                eyre!(
                    "failed to list llm_debug rows for client {}: {}",
                    client_id,
                    e
                )
            })?;
        rows.iter()
            .map(LLMDebugRow::from_row)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| eyre!("failed to decode llm_debug rows: {}", e).into())
    }
}

/// Unified debug interface used by `LLM`. Concrete backends (folder of
/// XML/JSON files vs SQLite) implement the same lifecycle: `start` to open
/// a row/file for the request, then any of `record_error`,
/// `record_response`, `record_billing` to fill it in.
#[derive(Debug)]
pub enum DebugBackend {
    Sqlite3(Sqlite3DebugDB),
    Folder(FolderDebug),
}

impl DebugBackend {
    /// Detect which backend the env-style debug string is asking for and
    /// build it. Returns the backend or any I/O error encountered while
    /// preparing it.
    pub async fn from_env_string(s: &str, debug_prefix: Option<&str>) -> Result<Self, LLMYError> {
        if is_sqlite_target(s) {
            let db = Sqlite3DebugDB::open(s).await?;
            Ok(DebugBackend::Sqlite3(db))
        } else {
            let folder = FolderDebug::new(Path::new(s), debug_prefix)?;
            Ok(DebugBackend::Folder(folder))
        }
    }

    /// Begin a debug record for an outgoing request. Returns `None` only on
    /// internal error — debug failures must never block the LLM call, so we
    /// log and swallow.
    pub async fn start(
        &self,
        debug_prefix: &str,
        ctx: DebugRowContext,
        req: &RawExtensibleChatCompletionRequest,
    ) -> Option<DebugHandle> {
        match self {
            DebugBackend::Sqlite3(db) => match db.insert_request(debug_prefix, &ctx, req).await {
                Ok(id) => Some(DebugHandle::Sqlite3 { row_id: id }),
                Err(e) => {
                    tracing::warn!("Fail to insert debug request row: {}", e);
                    None
                }
            },
            DebugBackend::Folder(fb) => {
                let fpath = fb.allocate_path(debug_prefix);
                if let Err(e) = save_llm_user(&fpath, req).await {
                    tracing::warn!("Fail to save user due to {}", e);
                }
                Some(DebugHandle::Folder { fpath })
            }
        }
    }

    pub async fn record_error(&self, handle: &DebugHandle, error: &str) {
        match (self, handle) {
            (DebugBackend::Sqlite3(db), DebugHandle::Sqlite3 { row_id }) => {
                if let Err(e) = db.update_error(*row_id, error).await {
                    tracing::warn!("Fail to record debug error: {}", e);
                }
            }
            (DebugBackend::Folder(_), DebugHandle::Folder { fpath }) => {
                if let Err(je) = rewrite_json(fpath, &serde_json::json!({ "error": error })).await {
                    tracing::warn!("can not save error: {} due to json error {}", error, je);
                }
            }
            _ => tracing::warn!("DebugHandle / DebugBackend variant mismatch"),
        }
    }

    pub async fn record_response(
        &self,
        handle: &DebugHandle,
        req: &RawExtensibleChatCompletionRequest,
        resp: &RawExtensibleChatCompletionResponse,
    ) {
        match (self, handle) {
            (DebugBackend::Sqlite3(db), DebugHandle::Sqlite3 { row_id }) => {
                if let Err(e) = db.update_response(*row_id, req, resp).await {
                    tracing::warn!("Fail to record debug response: {}", e);
                }
            }
            (DebugBackend::Folder(_), DebugHandle::Folder { fpath }) => {
                if let Err(e) = save_llm_resp(fpath, resp).await {
                    tracing::warn!("Fail to save resp due to {}", e);
                }
            }
            _ => tracing::warn!("DebugHandle / DebugBackend variant mismatch"),
        }
    }

    pub async fn record_billing(
        &self,
        handle: &DebugHandle,
        billing: &ModelBilling,
        usage: &DebugUsage,
    ) {
        match (self, handle) {
            (DebugBackend::Sqlite3(db), DebugHandle::Sqlite3 { row_id }) => {
                if let Err(e) = db.update_billing(*row_id, billing, usage).await {
                    tracing::warn!("Fail to record debug billing: {}", e);
                }
            }
            (DebugBackend::Folder(_), DebugHandle::Folder { fpath }) => {
                if let Err(e) = rewrite_json(fpath, billing).await {
                    tracing::warn!("can not write {} to debug file due to {}", billing, e);
                }
            }
            _ => tracing::warn!("DebugHandle / DebugBackend variant mismatch"),
        }
    }
}

/// True when the env var content asks for the SQLite backend.
pub fn is_sqlite_target(s: &str) -> bool {
    s.starts_with("sqlite3://") || s.ends_with("sqlite3")
}

fn first_choice_text(resp: &RawExtensibleChatCompletionResponse) -> Option<String> {
    resp.choices
        .first()
        .and_then(|c| c.inner.message.content.clone())
}

fn format_request_conversation(req: &RawExtensibleChatCompletionRequest) -> String {
    let mut s = String::new();
    for it in req.messages.iter() {
        s.push_str(&completion_to_string(it));
    }
    s
}

fn format_full_conversation(
    req: &RawExtensibleChatCompletionRequest,
    resp: &RawExtensibleChatCompletionResponse,
) -> String {
    let mut s = format_request_conversation(req);
    for ch in &resp.choices {
        s.push_str(&response_to_string(&ch.message));
    }
    s
}

#[cfg(test)]
#[allow(deprecated)]
mod sqlite_tests {
    use super::*;
    use crate::req::{ChatCompletionRequestMessageRaw, ChatCompletionRequestUserMessageRaw, Role};
    use crate::resp::{
        ChatChoiceRaw, ChatCompletionResponseMessageRaw, CreateChatCompletionResponseRaw,
    };
    use llmy_types::other::WithOtherFields;

    fn dummy_req() -> RawExtensibleChatCompletionRequest {
        let user = ChatCompletionRequestMessageRaw::User(
            ChatCompletionRequestUserMessageRaw::new_text("hello"),
        );
        let mut raw = CreateChatCompletionRequestRaw::default();
        raw.model = "gpt-test".to_string();
        raw.messages = vec![WithOtherFields::new(user)];
        RawExtensibleChatCompletionRequest::new(raw)
    }

    fn dummy_resp() -> RawExtensibleChatCompletionResponse {
        let message = WithOtherFields::new(ChatCompletionResponseMessageRaw {
            content: Some("world".to_string()),
            refusal: None,
            tool_calls: None,
            annotations: None,
            role: Role::Assistant,
            function_call: None,
            audio: None,
        });
        let choice = WithOtherFields::new(ChatChoiceRaw {
            index: 0,
            message,
            finish_reason: None,
            logprobs: None,
        });
        let raw = CreateChatCompletionResponseRaw {
            id: "test".to_string(),
            choices: vec![choice],
            created: 1,
            model: "gpt-test".to_string(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion".to_string(),
            usage: None,
        };
        RawExtensibleChatCompletionResponse::new(raw)
    }

    #[tokio::test]
    async fn sqlite_target_detection() {
        assert!(is_sqlite_target("sqlite3:///tmp/foo.db"));
        assert!(is_sqlite_target("./run.sqlite3"));
        assert!(!is_sqlite_target("./debug-folder"));
    }

    #[tokio::test]
    async fn sqlite_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("debug.sqlite3");
        let url = path.to_str().unwrap();
        let db = Sqlite3DebugDB::open(url).await.unwrap();

        let req = dummy_req();
        let ctx = DebugRowContext {
            model_name: "gpt-test".to_string(),
            endpoint: "openai".to_string(),
            azure_deployment: None,
            cache_key: Some("cache-key-1".to_string()),
            cap_usd: 10.0,
        };
        let id = db.insert_request("test", &ctx, &req).await.unwrap();

        let resp = dummy_resp();
        db.update_response(id, &req, &resp).await.unwrap();

        let billing = ModelBilling::new(10.0);
        let usage = DebugUsage {
            input_without_cached_tokens: 7,
            cached_tokens: 3,
            output_without_reasoning_tokens: 5,
            reasoning_tokens: 1,
        };
        db.update_billing(id, &billing, &usage).await.unwrap();

        let row = db.get_row(id).await.unwrap().expect("row missing");
        assert_eq!(row.model_name, "gpt-test");
        assert_eq!(row.endpoint, "openai");
        assert_eq!(row.debug_prefix, "test");
        assert_eq!(row.cache_key.as_deref(), Some("cache-key-1"));
        assert_eq!(row.cached_tokens, Some(3));
        assert_eq!(row.reasoning_tokens, Some(1));
        assert_eq!(row.resp_text_content.as_deref(), Some("world"));
        assert!(row.full_conversation.contains("world"));
        assert_eq!(row.cap_usd, 10.0);

        let listed = db
            .list_by_client(db.client_id().expect("client_id"))
            .await
            .unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, id);
    }
}
