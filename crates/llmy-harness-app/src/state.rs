//! SQLite-backed run state: every tool call is persisted in full, while the
//! copy fed back into the model context is truncated to a configurable size.
//! The model can always page through the untruncated result via
//! [`ReadToolOutputTool`], so a huge tool output never floods the context yet
//! is never lost either.

use std::collections::BTreeSet;
use std::fmt;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;

use color_eyre::eyre::eyre;
use llmy_agent::tool::{ToolBox, ToolDyn, ToolEntry};
use llmy_types::error::LLMYError;
use schemars::JsonSchema;
use serde::Deserialize;
use sqlx::Row;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};

const STATE_SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    model TEXT NOT NULL,
    user_prompt TEXT NOT NULL,
    final_output TEXT
);
CREATE TABLE IF NOT EXISTS tool_call (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    arguments TEXT NOT NULL,
    result TEXT NOT NULL,
    is_error INTEGER NOT NULL DEFAULT 0,
    truncated INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_call_run ON tool_call(run_id);
CREATE TABLE IF NOT EXISTS background_task (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    command TEXT NOT NULL,
    working_directory TEXT NOT NULL,
    status TEXT NOT NULL,
    exit_code INTEGER,
    output TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_background_task_run ON background_task(run_id);
"#;

/// How tool results are cut down before they are fed back to the model.
#[derive(Debug, Clone)]
pub struct ToolResultPolicy {
    /// Maximum number of characters of a tool result kept in the model
    /// context. The full result always lands in SQLite regardless.
    pub max_result_chars: usize,
    /// Tools that enforce their own output cap and therefore bypass the
    /// generic truncation (e.g. the paging reader itself).
    pub exempt_tools: BTreeSet<String>,
}

impl Default for ToolResultPolicy {
    fn default() -> Self {
        Self {
            max_result_chars: 20_000,
            exempt_tools: BTreeSet::new(),
        }
    }
}

impl ToolResultPolicy {
    /// Cut `result` down to the policy size, referencing the stored row so
    /// the model knows how to page through the rest. `None` means the result
    /// fits and is passed through unchanged.
    fn truncate(&self, result: &str, output_id: i64) -> Option<String> {
        let total = result.chars().count();
        if total <= self.max_result_chars {
            return None;
        }
        let cut = result
            .char_indices()
            .nth(self.max_result_chars)
            .map(|(byte, _)| byte)
            .unwrap_or(result.len());
        let mut out = result[..cut].to_string();
        out.push_str(&format!(
            "\n[tool result truncated: showing the first {} of {} characters. \
             The full result is stored as tool output #{output_id}; call \
             `read_tool_output` with output_id={output_id} and an offset to read the rest.]",
            self.max_result_chars, total
        ));
        Some(out)
    }

    /// The largest chunk a self-capping reader tool should emit so that its
    /// own output (chunk plus a small header) never trips the generic
    /// truncation again.
    pub fn reader_chunk_chars(&self) -> usize {
        self.max_result_chars.saturating_sub(1_000).max(1_000)
    }
}

/// A fully persisted tool call row, as read back from the store.
#[derive(Debug, Clone)]
pub struct StoredToolOutput {
    pub id: i64,
    pub tool_name: String,
    pub result: String,
}

/// SQLite handle holding all core harness state. Clones share the pool.
#[derive(Debug, Clone)]
pub struct HarnessStateDB {
    pool: SqlitePool,
    path: String,
}

impl HarnessStateDB {
    /// Open (or create) the state database at `path` and ensure the schema.
    pub async fn open(path: &str) -> Result<Self, LLMYError> {
        let fs_path = Path::new(path);
        if fs_path.is_dir() {
            return Err(eyre!(
                "harness state db path {} is a directory; sqlite needs a file",
                path
            )
            .into());
        }
        if let Some(parent) = fs_path.parent()
            && !parent.as_os_str().is_empty()
        {
            tokio::fs::create_dir_all(parent).await?;
        }

        let opts = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal);
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(opts)
            .await
            .map_err(|e| eyre!("failed to open harness state db at {}: {}", path, e))?;

        for stmt in STATE_SCHEMA_SQL.split(';') {
            let trimmed = stmt.trim();
            if trimmed.is_empty() {
                continue;
            }
            sqlx::query(trimmed)
                .execute(&pool)
                .await
                .map_err(|e| eyre!("failed to apply harness state schema: {}", e))?;
        }

        Ok(Self {
            pool,
            path: path.to_string(),
        })
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub async fn begin_run(&self, model: &str, user_prompt: &str) -> Result<i64, LLMYError> {
        let row = sqlx::query(
            "INSERT INTO run (started_at, status, model, user_prompt) VALUES (?, 'running', ?, ?) RETURNING id",
        )
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(model)
        .bind(user_prompt)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| eyre!("failed to insert run row: {}", e))?;
        row.try_get::<i64, _>("id")
            .map_err(|e| eyre!("failed to read run id: {}", e).into())
    }

    pub async fn finish_run(
        &self,
        run_id: i64,
        status: &str,
        final_output: Option<&str>,
    ) -> Result<(), LLMYError> {
        sqlx::query("UPDATE run SET finished_at = ?, status = ?, final_output = ? WHERE id = ?")
            .bind(chrono::Utc::now().to_rfc3339())
            .bind(status)
            .bind(final_output)
            .bind(run_id)
            .execute(&self.pool)
            .await
            .map_err(|e| eyre!("failed to finish run row: {}", e))?;
        Ok(())
    }

    pub async fn record_tool_call(
        &self,
        run_id: i64,
        tool_name: &str,
        arguments: &str,
        result: &str,
        is_error: bool,
    ) -> Result<i64, LLMYError> {
        let row = sqlx::query(
            "INSERT INTO tool_call (run_id, tool_name, arguments, result, is_error, created_at) \
             VALUES (?, ?, ?, ?, ?, ?) RETURNING id",
        )
        .bind(run_id)
        .bind(tool_name)
        .bind(arguments)
        .bind(result)
        .bind(is_error)
        .bind(chrono::Utc::now().to_rfc3339())
        .fetch_one(&self.pool)
        .await
        .map_err(|e| eyre!("failed to record tool call: {}", e))?;
        row.try_get::<i64, _>("id")
            .map_err(|e| eyre!("failed to read tool call id: {}", e).into())
    }

    pub async fn mark_tool_call_truncated(&self, id: i64) -> Result<(), LLMYError> {
        sqlx::query("UPDATE tool_call SET truncated = 1 WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| eyre!("failed to mark tool call truncated: {}", e))?;
        Ok(())
    }

    pub async fn read_tool_output(
        &self,
        output_id: i64,
    ) -> Result<Option<StoredToolOutput>, LLMYError> {
        let row = sqlx::query("SELECT id, tool_name, result FROM tool_call WHERE id = ?")
            .bind(output_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| eyre!("failed to read tool output: {}", e))?;
        row.map(|row| {
            Ok(StoredToolOutput {
                id: row.try_get("id").map_err(|e| eyre!("{}", e))?,
                tool_name: row.try_get("tool_name").map_err(|e| eyre!("{}", e))?,
                result: row.try_get("result").map_err(|e| eyre!("{}", e))?,
            })
        })
        .transpose()
    }

    /// Number of calls to any of `tool_names` recorded for this run. Used by
    /// the runner to check gates such as "did the agent write any memory".
    pub async fn count_tool_calls(
        &self,
        run_id: i64,
        tool_names: &[&str],
    ) -> Result<u64, LLMYError> {
        let mut count = 0u64;
        for name in tool_names {
            let row = sqlx::query(
                "SELECT COUNT(*) AS n FROM tool_call WHERE run_id = ? AND tool_name = ? AND is_error = 0",
            )
            .bind(run_id)
            .bind(name)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| eyre!("failed to count tool calls: {}", e))?;
            count += row.try_get::<i64, _>("n").map_err(|e| eyre!("{}", e))? as u64;
        }
        Ok(count)
    }

    pub async fn insert_background_task(
        &self,
        run_id: i64,
        command: &str,
        working_directory: &str,
    ) -> Result<i64, LLMYError> {
        let row = sqlx::query(
            "INSERT INTO background_task (run_id, command, working_directory, status, started_at) \
             VALUES (?, ?, ?, 'running', ?) RETURNING id",
        )
        .bind(run_id)
        .bind(command)
        .bind(working_directory)
        .bind(chrono::Utc::now().to_rfc3339())
        .fetch_one(&self.pool)
        .await
        .map_err(|e| eyre!("failed to insert background task: {}", e))?;
        row.try_get::<i64, _>("id")
            .map_err(|e| eyre!("failed to read background task id: {}", e).into())
    }

    pub async fn finish_background_task(
        &self,
        task_id: i64,
        status: &str,
        exit_code: Option<i32>,
        output: &str,
    ) -> Result<(), LLMYError> {
        sqlx::query(
            "UPDATE background_task SET status = ?, exit_code = ?, output = ?, finished_at = ? WHERE id = ?",
        )
        .bind(status)
        .bind(exit_code)
        .bind(output)
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(task_id)
        .execute(&self.pool)
        .await
        .map_err(|e| eyre!("failed to finish background task: {}", e))?;
        Ok(())
    }

    /// Wrap every tool of `tools` in a recording adapter bound to `run_id`,
    /// so full results are persisted and context copies are truncated per
    /// `policy`.
    pub fn record_toolbox(
        &self,
        tools: &ToolBox,
        run_id: i64,
        policy: &ToolResultPolicy,
    ) -> Result<ToolBox, LLMYError> {
        let mut recorded = ToolBox::new();
        for (_, entry) in tools.entries() {
            recorded.add_dyn_tool(Box::new(RecordedTool {
                inner: entry.clone(),
                db: self.clone(),
                run_id,
                policy: policy.clone(),
            }))?;
        }
        Ok(recorded)
    }
}

/// A [`ToolDyn`] adapter that persists the full result of every invocation
/// into the state database and hands the model a truncated copy when the
/// result exceeds the policy size.
#[derive(Clone)]
pub struct RecordedTool {
    inner: ToolEntry,
    db: HarnessStateDB,
    run_id: i64,
    policy: ToolResultPolicy,
}

impl fmt::Debug for RecordedTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RecordedTool")
            .field("inner", &self.inner.tool().name())
            .field("run_id", &self.run_id)
            .finish()
    }
}

impl ToolDyn for RecordedTool {
    fn name(&self) -> String {
        self.inner.tool().name()
    }

    fn description(&self) -> Option<String> {
        self.inner.tool().description()
    }

    fn schema(&self) -> schemars::Schema {
        self.inner.tool().schema()
    }

    fn strict(&self) -> bool {
        self.inner.tool().strict()
    }

    fn validate(
        &self,
        arguments: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + '_>> {
        self.inner.tool().validate(arguments)
    }

    fn run(
        &self,
        arguments: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<String, LLMYError>> + Send + '_>> {
        Box::pin(async move {
            let rendered_args = arguments.to_string();
            let result = self.inner.tool().run(arguments).await;

            match result {
                Ok(output) => {
                    let output_id = self
                        .db
                        .record_tool_call(self.run_id, &self.name(), &rendered_args, &output, false)
                        .await?;
                    if self.policy.exempt_tools.contains(&self.name()) {
                        return Ok(output);
                    }
                    match self.policy.truncate(&output, output_id) {
                        Some(truncated) => {
                            self.db.mark_tool_call_truncated(output_id).await?;
                            tracing::info!(
                                "tool {} result truncated ({} chars), stored as output #{}",
                                self.name(),
                                output.chars().count(),
                                output_id
                            );
                            Ok(truncated)
                        }
                        None => Ok(output),
                    }
                }
                Err(error) => {
                    // Recording an error must not mask it: log any storage
                    // failure and surface the original error unchanged.
                    if let Err(record_error) = self
                        .db
                        .record_tool_call(
                            self.run_id,
                            &self.name(),
                            &rendered_args,
                            &error.to_string(),
                            true,
                        )
                        .await
                    {
                        tracing::warn!("failed to record erroring tool call: {}", record_error);
                    }
                    Err(error)
                }
            }
        })
    }
}

/// Arguments accepted by [`ReadToolOutputTool`].
#[derive(Deserialize, JsonSchema)]
pub struct ReadToolOutputArgs {
    /// Id of the stored tool output, as referenced in a truncation notice.
    pub output_id: i64,
    /// 0-based character offset to start reading from. Defaults to 0.
    #[serde(default)]
    pub offset: Option<usize>,
    /// Maximum number of characters to return. Defaults to (and is capped by)
    /// the harness tool-result limit.
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Pages through the full, untruncated result of any earlier tool call.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = ReadToolOutputArgs,
    invoke = read,
    name = "read_tool_output",
    description = "Read the full stored result of an earlier tool call. Whenever a tool result says it was truncated, it names an output_id; use this tool with that output_id and an optional character offset/limit to page through the untruncated result.",
)]
pub struct ReadToolOutputTool {
    db: HarnessStateDB,
    policy: ToolResultPolicy,
}

impl ReadToolOutputTool {
    pub fn new(db: HarnessStateDB, policy: ToolResultPolicy) -> Self {
        Self { db, policy }
    }

    async fn read(&self, args: ReadToolOutputArgs) -> Result<String, LLMYError> {
        let Some(stored) = self.db.read_tool_output(args.output_id).await? else {
            return Ok(format!("No stored tool output with id {}", args.output_id));
        };

        let total = stored.result.chars().count();
        let offset = args.offset.unwrap_or(0);
        let chunk_cap = self.policy.reader_chunk_chars();
        let limit = args.limit.unwrap_or(chunk_cap).min(chunk_cap);

        if offset >= total && total != 0 {
            return Ok(format!(
                "Tool output #{} ({}) has {} characters; offset {} is past the end.",
                stored.id, stored.tool_name, total, offset
            ));
        }

        let chunk: String = stored.result.chars().skip(offset).take(limit).collect();
        let end = offset + chunk.chars().count();
        let mut header = format!(
            "Tool output #{} ({}), total {} characters, showing [{}..{})",
            stored.id, stored.tool_name, total, offset, end
        );
        if end < total {
            header.push_str(&format!("; continue with offset={} to read the rest", end));
        }
        Ok(format!("{header}\n{chunk}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmy_agent::Tool;

    #[derive(Debug, Clone)]
    struct LongOutputTool {
        chars: usize,
    }

    impl Tool for LongOutputTool {
        type ARGUMENTS = ();
        const NAME: &str = "long_output_tool";
        const DESCRIPTION: Option<&str> = Some("test tool");

        async fn invoke(&self, _arguments: Self::ARGUMENTS) -> Result<String, LLMYError> {
            Ok("x".repeat(self.chars))
        }
    }

    async fn open_db() -> (tempfile::TempDir, HarnessStateDB, i64) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("state.sqlite3");
        let db = HarnessStateDB::open(&path.display().to_string())
            .await
            .expect("open state db");
        let run_id = db.begin_run("test-model", "prompt").await.expect("run row");
        (dir, db, run_id)
    }

    #[tokio::test]
    async fn long_results_are_truncated_and_pageable() {
        let (_dir, db, run_id) = open_db().await;
        let policy = ToolResultPolicy {
            max_result_chars: 100,
            exempt_tools: BTreeSet::from(["read_tool_output".to_string()]),
        };
        let mut tools = ToolBox::new();
        tools.add_tool(LongOutputTool { chars: 250 });
        tools.add_tool(ReadToolOutputTool::new(db.clone(), policy.clone()));
        let recorded = db
            .record_toolbox(&tools, run_id, &policy)
            .expect("record toolbox");

        let truncated = recorded
            .invoke("long_output_tool".to_string(), "null".to_string())
            .await
            .expect("tool exists")
            .expect("tool ran");
        assert!(truncated.contains("truncated"), "{truncated}");
        assert!(truncated.contains("read_tool_output"), "{truncated}");

        // The notice names the stored output id; the stored row holds all
        // 250 characters even though the context copy was cut at 100.
        let stored = db.read_tool_output(1).await.expect("query").expect("row");
        assert_eq!(stored.result.chars().count(), 250);

        let paged = recorded
            .invoke(
                "read_tool_output".to_string(),
                r#"{"output_id": 1, "offset": 240}"#.to_string(),
            )
            .await
            .expect("tool exists")
            .expect("tool ran");
        assert!(paged.contains("showing [240..250)"), "{paged}");
        assert!(!paged.contains("continue with offset"), "{paged}");
    }

    #[tokio::test]
    async fn short_results_pass_through_untouched() {
        let (_dir, db, run_id) = open_db().await;
        let policy = ToolResultPolicy::default();
        let mut tools = ToolBox::new();
        tools.add_tool(LongOutputTool { chars: 10 });
        let recorded = db
            .record_toolbox(&tools, run_id, &policy)
            .expect("record toolbox");

        let result = recorded
            .invoke("long_output_tool".to_string(), "null".to_string())
            .await
            .expect("tool exists")
            .expect("tool ran");
        assert_eq!(result, "x".repeat(10));
        assert_eq!(
            db.count_tool_calls(run_id, &["long_output_tool"])
                .await
                .expect("count"),
            1
        );
    }
}
