//! SQLite cache for prebuilt code graphs. The graph is stored as one
//! serialized snapshot per root keyed by an input fingerprint (relative
//! paths and sizes), so `llmy codegraph index` can prebuild and the harness
//! can load-or-rebuild cheaply. A stale fingerprint simply misses.

use std::path::Path;

use color_eyre::eyre::eyre;
use llmy_types::error::LLMYError;
use sqlx::Row;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};

use crate::builder::BuildResult;
use crate::model::CodeGraph;

const STORE_SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS codegraph_snapshot (
    root TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    graph_json TEXT NOT NULL,
    files INTEGER NOT NULL,
    parse_errors INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"#;

#[derive(Debug, Clone)]
pub struct CodeGraphStore {
    pool: SqlitePool,
    path: String,
}

impl CodeGraphStore {
    pub async fn open(path: &str) -> Result<Self, LLMYError> {
        let fs_path = Path::new(path);
        if fs_path.is_dir() {
            return Err(eyre!(
                "codegraph db path {} is a directory; sqlite needs a file",
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
            .max_connections(2)
            .connect_with(opts)
            .await
            .map_err(|e| eyre!("failed to open codegraph db at {}: {}", path, e))?;
        for stmt in STORE_SCHEMA_SQL.split(';') {
            let trimmed = stmt.trim();
            if trimmed.is_empty() {
                continue;
            }
            sqlx::query(trimmed)
                .execute(&pool)
                .await
                .map_err(|e| eyre!("failed to apply codegraph schema: {}", e))?;
        }
        Ok(Self {
            pool,
            path: path.to_string(),
        })
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub async fn save(&self, root: &str, result: &BuildResult) -> Result<(), LLMYError> {
        let graph_json = serde_json::to_string(&result.graph)
            .map_err(|e| eyre!("failed to serialize code graph: {}", e))?;
        sqlx::query(
            "INSERT INTO codegraph_snapshot (root, fingerprint, graph_json, files, parse_errors, created_at) \
             VALUES (?, ?, ?, ?, ?, ?) \
             ON CONFLICT(root) DO UPDATE SET \
             fingerprint = excluded.fingerprint, graph_json = excluded.graph_json, \
             files = excluded.files, parse_errors = excluded.parse_errors, created_at = excluded.created_at",
        )
        .bind(root)
        .bind(&result.fingerprint)
        .bind(graph_json)
        .bind(result.files.len() as i64)
        .bind(result.total_parse_errors() as i64)
        .bind(chrono::Utc::now().to_rfc3339())
        .execute(&self.pool)
        .await
        .map_err(|e| eyre!("failed to save codegraph snapshot: {}", e))?;
        Ok(())
    }

    /// The cached graph for `root`, but only when the stored fingerprint
    /// still matches the given one.
    pub async fn load_fresh(
        &self,
        root: &str,
        fingerprint: &str,
    ) -> Result<Option<CodeGraph>, LLMYError> {
        let row =
            sqlx::query("SELECT fingerprint, graph_json FROM codegraph_snapshot WHERE root = ?")
                .bind(root)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| eyre!("failed to load codegraph snapshot: {}", e))?;
        let Some(row) = row else {
            return Ok(None);
        };
        let stored_fingerprint: String = row.try_get("fingerprint").map_err(|e| eyre!("{}", e))?;
        if stored_fingerprint != fingerprint {
            tracing::info!("codegraph cache for {} is stale, rebuilding", root);
            return Ok(None);
        }
        let graph_json: String = row.try_get("graph_json").map_err(|e| eyre!("{}", e))?;
        let graph = serde_json::from_str::<CodeGraph>(&graph_json)
            .map_err(|e| eyre!("failed to deserialize cached code graph: {}", e))?;
        Ok(Some(graph))
    }
}
