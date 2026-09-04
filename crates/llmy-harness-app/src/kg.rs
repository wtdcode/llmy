//! Knowledge-graph memory: a SQLite-backed directed graph of named nodes and
//! typed edges that persists across runs. There is no embedding search on
//! purpose — retrieval is an index injected into the system prompt plus
//! explicit graph traversal and regex tools. The *meaning* of node levels is
//! not fixed here: the memory instruction (caller-provided or the default)
//! defines the hierarchy, this module only provides the graph.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use color_eyre::eyre::eyre;
use llmy_types::error::LLMYError;
use schemars::JsonSchema;
use serde::Deserialize;
use sqlx::Row;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};

const KG_SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS kg_node (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    level TEXT NOT NULL,
    summary TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kg_edge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_id INTEGER NOT NULL,
    to_id INTEGER NOT NULL,
    relation TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(from_id, to_id, relation)
);
CREATE INDEX IF NOT EXISTS idx_kg_edge_from ON kg_edge(from_id);
CREATE INDEX IF NOT EXISTS idx_kg_edge_to ON kg_edge(to_id);
"#;

#[derive(Debug, Clone)]
pub struct KgNode {
    pub id: i64,
    pub name: String,
    pub level: String,
    pub summary: String,
    pub content: String,
    pub updated_at: String,
}

impl KgNode {
    fn from_row(row: &sqlx::sqlite::SqliteRow) -> Result<Self, LLMYError> {
        Ok(Self {
            id: row.try_get("id").map_err(|e| eyre!("{}", e))?,
            name: row.try_get("name").map_err(|e| eyre!("{}", e))?,
            level: row.try_get("level").map_err(|e| eyre!("{}", e))?,
            summary: row.try_get("summary").map_err(|e| eyre!("{}", e))?,
            content: row.try_get("content").map_err(|e| eyre!("{}", e))?,
            updated_at: row.try_get("updated_at").map_err(|e| eyre!("{}", e))?,
        })
    }
}

/// An edge with endpoint names already joined in, ready for rendering.
#[derive(Debug, Clone)]
pub struct KgEdge {
    pub from_name: String,
    pub to_name: String,
    pub relation: String,
    pub description: Option<String>,
}

impl KgEdge {
    fn render(&self) -> String {
        match self.description.as_deref() {
            Some(description) => format!(
                "{} -[{}]-> {} ({})",
                self.from_name, self.relation, self.to_name, description
            ),
            None => format!("{} -[{}]-> {}", self.from_name, self.relation, self.to_name),
        }
    }
}

/// SQLite handle for the memory graph. Clones share the pool.
#[derive(Debug, Clone)]
pub struct KgMemoryDB {
    pool: SqlitePool,
    path: String,
}

impl KgMemoryDB {
    pub async fn open(path: &str) -> Result<Self, LLMYError> {
        let fs_path = Path::new(path);
        if fs_path.is_dir() {
            return Err(eyre!(
                "memory db path {} is a directory; sqlite needs a file",
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
            .map_err(|e| eyre!("failed to open memory db at {}: {}", path, e))?;
        for stmt in KG_SCHEMA_SQL.split(';') {
            let trimmed = stmt.trim();
            if trimmed.is_empty() {
                continue;
            }
            sqlx::query(trimmed)
                .execute(&pool)
                .await
                .map_err(|e| eyre!("failed to apply memory schema: {}", e))?;
        }
        Ok(Self {
            pool,
            path: path.to_string(),
        })
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub async fn node_by_name(&self, name: &str) -> Result<Option<KgNode>, LLMYError> {
        let row = sqlx::query("SELECT * FROM kg_node WHERE name = ?")
            .bind(name)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| eyre!("failed to read node: {}", e))?;
        row.map(|row| KgNode::from_row(&row)).transpose()
    }

    pub async fn all_nodes(&self) -> Result<Vec<KgNode>, LLMYError> {
        let rows = sqlx::query("SELECT * FROM kg_node ORDER BY level, name")
            .fetch_all(&self.pool)
            .await
            .map_err(|e| eyre!("failed to list nodes: {}", e))?;
        rows.iter().map(KgNode::from_row).collect()
    }

    pub async fn nodes_by_level(&self, level: &str) -> Result<Vec<KgNode>, LLMYError> {
        let rows = sqlx::query("SELECT * FROM kg_node WHERE level = ? ORDER BY name")
            .bind(level)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| eyre!("failed to list nodes by level: {}", e))?;
        rows.iter().map(KgNode::from_row).collect()
    }

    pub async fn insert_node(
        &self,
        name: &str,
        level: &str,
        summary: &str,
        content: &str,
    ) -> Result<i64, LLMYError> {
        let now = chrono::Utc::now().to_rfc3339();
        let row = sqlx::query(
            "INSERT INTO kg_node (name, level, summary, content, created_at, updated_at) \
             VALUES (?, ?, ?, ?, ?, ?) RETURNING id",
        )
        .bind(name)
        .bind(level)
        .bind(summary)
        .bind(content)
        .bind(&now)
        .bind(&now)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| eyre!("failed to insert node: {}", e))?;
        row.try_get::<i64, _>("id")
            .map_err(|e| eyre!("failed to read node id: {}", e).into())
    }

    pub async fn update_node(
        &self,
        id: i64,
        level: Option<&str>,
        summary: Option<&str>,
        content: Option<&str>,
    ) -> Result<(), LLMYError> {
        sqlx::query(
            "UPDATE kg_node SET \
             level = COALESCE(?, level), \
             summary = COALESCE(?, summary), \
             content = COALESCE(?, content), \
             updated_at = ? WHERE id = ?",
        )
        .bind(level)
        .bind(summary)
        .bind(content)
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(|e| eyre!("failed to update node: {}", e))?;
        Ok(())
    }

    pub async fn delete_node(&self, id: i64) -> Result<(), LLMYError> {
        sqlx::query("DELETE FROM kg_edge WHERE from_id = ? OR to_id = ?")
            .bind(id)
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| eyre!("failed to delete edges: {}", e))?;
        sqlx::query("DELETE FROM kg_node WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| eyre!("failed to delete node: {}", e))?;
        Ok(())
    }

    pub async fn upsert_edge(
        &self,
        from_id: i64,
        to_id: i64,
        relation: &str,
        description: Option<&str>,
    ) -> Result<(), LLMYError> {
        sqlx::query(
            "INSERT INTO kg_edge (from_id, to_id, relation, description, created_at) \
             VALUES (?, ?, ?, ?, ?) \
             ON CONFLICT(from_id, to_id, relation) DO UPDATE SET description = excluded.description",
        )
        .bind(from_id)
        .bind(to_id)
        .bind(relation)
        .bind(description)
        .bind(chrono::Utc::now().to_rfc3339())
        .execute(&self.pool)
        .await
        .map_err(|e| eyre!("failed to upsert edge: {}", e))?;
        Ok(())
    }

    pub async fn delete_edges(
        &self,
        from_id: i64,
        to_id: i64,
        relation: Option<&str>,
    ) -> Result<u64, LLMYError> {
        let result = match relation {
            Some(relation) => {
                sqlx::query("DELETE FROM kg_edge WHERE from_id = ? AND to_id = ? AND relation = ?")
                    .bind(from_id)
                    .bind(to_id)
                    .bind(relation)
                    .execute(&self.pool)
                    .await
            }
            None => {
                sqlx::query("DELETE FROM kg_edge WHERE from_id = ? AND to_id = ?")
                    .bind(from_id)
                    .bind(to_id)
                    .execute(&self.pool)
                    .await
            }
        }
        .map_err(|e| eyre!("failed to delete edges: {}", e))?;
        Ok(result.rows_affected())
    }

    /// All edges, endpoint names joined, ordered for stable rendering.
    pub async fn all_edges(&self) -> Result<Vec<KgEdge>, LLMYError> {
        let rows = sqlx::query(
            "SELECT a.name AS from_name, b.name AS to_name, e.relation, e.description \
             FROM kg_edge e \
             JOIN kg_node a ON a.id = e.from_id \
             JOIN kg_node b ON b.id = e.to_id \
             ORDER BY a.name, e.relation, b.name",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| eyre!("failed to list edges: {}", e))?;
        rows.iter()
            .map(|row| {
                Ok(KgEdge {
                    from_name: row.try_get("from_name").map_err(|e| eyre!("{}", e))?,
                    to_name: row.try_get("to_name").map_err(|e| eyre!("{}", e))?,
                    relation: row.try_get("relation").map_err(|e| eyre!("{}", e))?,
                    description: row.try_get("description").map_err(|e| eyre!("{}", e))?,
                })
            })
            .collect()
    }

    /// Edges touching one node, endpoint names joined.
    pub async fn edges_of(&self, node_id: i64) -> Result<Vec<KgEdge>, LLMYError> {
        let rows = sqlx::query(
            "SELECT a.name AS from_name, b.name AS to_name, e.relation, e.description \
             FROM kg_edge e \
             JOIN kg_node a ON a.id = e.from_id \
             JOIN kg_node b ON b.id = e.to_id \
             WHERE e.from_id = ? OR e.to_id = ? \
             ORDER BY a.name, e.relation, b.name",
        )
        .bind(node_id)
        .bind(node_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| eyre!("failed to list node edges: {}", e))?;
        rows.iter()
            .map(|row| {
                Ok(KgEdge {
                    from_name: row.try_get("from_name").map_err(|e| eyre!("{}", e))?,
                    to_name: row.try_get("to_name").map_err(|e| eyre!("{}", e))?,
                    relation: row.try_get("relation").map_err(|e| eyre!("{}", e))?,
                    description: row.try_get("description").map_err(|e| eyre!("{}", e))?,
                })
            })
            .collect()
    }
}

/// The memory graph plus the per-run mutation counter used by the
/// force-memory gate.
#[derive(Debug, Clone)]
pub struct KgMemoryContext {
    pub db: KgMemoryDB,
    writes: Arc<AtomicU64>,
}

impl KgMemoryContext {
    pub fn new(db: KgMemoryDB) -> Self {
        Self {
            db,
            writes: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn write_count(&self) -> u64 {
        self.writes.load(Ordering::SeqCst)
    }

    fn count_write(&self) {
        self.writes.fetch_add(1, Ordering::SeqCst);
    }

    /// Render the compact index injected into the system prompt: node names,
    /// levels and summaries plus the edge list — full contents stay behind
    /// the read tools.
    pub async fn render_index(&self, max_chars: usize) -> Result<String, LLMYError> {
        let nodes = self.db.all_nodes().await?;
        if nodes.is_empty() {
            return Ok("The memory graph is currently empty.".to_string());
        }

        let mut by_level: BTreeMap<String, Vec<&KgNode>> = BTreeMap::new();
        for node in &nodes {
            by_level.entry(node.level.clone()).or_default().push(node);
        }

        let mut sections = vec![format!("Memory graph index ({} nodes):", nodes.len())];
        for (level, level_nodes) in &by_level {
            let mut lines = vec![format!("[{level}]")];
            for node in level_nodes {
                lines.push(format!("- {}: {}", node.name, node.summary));
            }
            sections.push(lines.join("\n"));
        }

        let edges = self.db.all_edges().await?;
        if !edges.is_empty() {
            let mut lines = vec!["Edges:".to_string()];
            for edge in &edges {
                lines.push(format!("- {}", edge.render()));
            }
            sections.push(lines.join("\n"));
        }

        let mut rendered = sections.join("\n\n");
        if rendered.chars().count() > max_chars {
            let cut = rendered
                .char_indices()
                .nth(max_chars)
                .map(|(byte, _)| byte)
                .unwrap_or(rendered.len());
            rendered.truncate(cut);
            rendered.push_str(
                "\n[memory index truncated; use list_memory and grep_memory to explore the rest]",
            );
        }
        Ok(rendered)
    }

    pub fn tool_box(&self) -> llmy_agent::tool::ToolBox {
        let mut tools = llmy_agent::tool::ToolBox::new();
        tools.add_tool(WriteMemoryTool::new(self.clone()));
        tools.add_tool(UpdateMemoryTool::new(self.clone()));
        tools.add_tool(DeleteMemoryTool::new(self.clone()));
        tools.add_tool(LinkMemoryTool::new(self.clone()));
        tools.add_tool(UnlinkMemoryTool::new(self.clone()));
        tools.add_tool(ReadMemoryTool::new(self.clone()));
        tools.add_tool(ListMemoryTool::new(self.clone()));
        tools.add_tool(GrepMemoryTool::new(self.clone()));
        tools
    }

    /// Tool names whose successful use counts as "the agent wrote memory".
    pub fn write_tool_names() -> Vec<&'static str> {
        vec!["write_memory", "update_memory", "link_memory"]
    }
}

/// A link requested at node creation time.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct MemoryLinkSpec {
    /// Name of the target node (must exist).
    pub to: String,
    /// Relation label of the edge (free-form, e.g. "depends_on",
    /// "instance_of", "contradicts").
    pub relation: String,
    /// Optional description of the edge.
    #[serde(default)]
    pub description: Option<String>,
}

/// Arguments accepted by [`WriteMemoryTool`].
#[derive(Deserialize, JsonSchema)]
pub struct WriteMemoryArgs {
    /// Unique node name. Short, stable, slug-like.
    pub name: String,
    /// The node's level in the memory hierarchy, as defined by the memory
    /// instruction.
    pub level: String,
    /// One-line summary shown in the memory index.
    pub summary: String,
    /// Full content of the memory.
    pub content: String,
    /// Optional edges from this node to existing nodes.
    #[serde(default)]
    pub links: Option<Vec<MemoryLinkSpec>>,
}

/// Creates a memory node (and optionally its outgoing edges).
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = WriteMemoryArgs,
    invoke = write,
    name = "write_memory",
    description = "Create a new memory node in the knowledge graph: a unique name, a level (per the memory instruction), a one-line summary for the index, the full content, and optional typed links to existing nodes.",
)]
pub struct WriteMemoryTool {
    context: KgMemoryContext,
}

impl WriteMemoryTool {
    pub fn new(context: KgMemoryContext) -> Self {
        Self { context }
    }

    async fn write(&self, args: WriteMemoryArgs) -> Result<String, LLMYError> {
        if args.name.trim().is_empty() {
            return Ok("write_memory failed: name must not be empty".to_string());
        }
        if self.context.db.node_by_name(&args.name).await?.is_some() {
            return Ok(format!(
                "write_memory failed: a node named {:?} already exists; use update_memory to modify it or pick another name",
                args.name
            ));
        }
        let id = self
            .context
            .db
            .insert_node(&args.name, &args.level, &args.summary, &args.content)
            .await?;
        self.context.count_write();

        let mut linked = vec![];
        let mut link_errors = vec![];
        for link in args.links.unwrap_or_default() {
            match self.context.db.node_by_name(&link.to).await? {
                Some(target) => {
                    self.context
                        .db
                        .upsert_edge(id, target.id, &link.relation, link.description.as_deref())
                        .await?;
                    linked.push(format!("{} -[{}]-> {}", args.name, link.relation, link.to));
                }
                None => link_errors.push(format!(
                    "link to {:?} skipped: no such node (create it first, then link_memory)",
                    link.to
                )),
            }
        }

        let mut out = format!(
            "Memory node {:?} created at level {:?}.",
            args.name, args.level
        );
        if !linked.is_empty() {
            out.push_str(&format!("\nEdges: {}", linked.join(", ")));
        }
        if !link_errors.is_empty() {
            out.push_str(&format!("\n{}", link_errors.join("\n")));
        }
        Ok(out)
    }
}

/// Arguments accepted by [`UpdateMemoryTool`].
#[derive(Deserialize, JsonSchema)]
pub struct UpdateMemoryArgs {
    /// Name of the node to update.
    pub name: String,
    /// New level; omit to keep.
    #[serde(default)]
    pub level: Option<String>,
    /// New one-line summary; omit to keep.
    #[serde(default)]
    pub summary: Option<String>,
    /// New full content; omit to keep.
    #[serde(default)]
    pub content: Option<String>,
}

/// Updates fields of an existing memory node.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = UpdateMemoryArgs,
    invoke = update,
    name = "update_memory",
    description = "Update an existing memory node's level, summary and/or content. Omitted fields are kept.",
)]
pub struct UpdateMemoryTool {
    context: KgMemoryContext,
}

impl UpdateMemoryTool {
    pub fn new(context: KgMemoryContext) -> Self {
        Self { context }
    }

    async fn update(&self, args: UpdateMemoryArgs) -> Result<String, LLMYError> {
        let Some(node) = self.context.db.node_by_name(&args.name).await? else {
            return Ok(format!(
                "update_memory failed: no node named {:?}; use write_memory to create it",
                args.name
            ));
        };
        if args.level.is_none() && args.summary.is_none() && args.content.is_none() {
            return Ok(
                "update_memory failed: nothing to update (provide level, summary and/or content)"
                    .to_string(),
            );
        }
        self.context
            .db
            .update_node(
                node.id,
                args.level.as_deref(),
                args.summary.as_deref(),
                args.content.as_deref(),
            )
            .await?;
        self.context.count_write();
        Ok(format!("Memory node {:?} updated.", args.name))
    }
}

/// Arguments accepted by [`DeleteMemoryTool`].
#[derive(Deserialize, JsonSchema)]
pub struct DeleteMemoryArgs {
    /// Name of the node to delete.
    pub name: String,
}

/// Deletes a memory node and every edge touching it.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = DeleteMemoryArgs,
    invoke = delete,
    name = "delete_memory",
    description = "Delete a memory node and all of its edges. Use when a memory turned out to be wrong or obsolete.",
)]
pub struct DeleteMemoryTool {
    context: KgMemoryContext,
}

impl DeleteMemoryTool {
    pub fn new(context: KgMemoryContext) -> Self {
        Self { context }
    }

    async fn delete(&self, args: DeleteMemoryArgs) -> Result<String, LLMYError> {
        let Some(node) = self.context.db.node_by_name(&args.name).await? else {
            return Ok(format!(
                "delete_memory failed: no node named {:?}",
                args.name
            ));
        };
        self.context.db.delete_node(node.id).await?;
        Ok(format!("Memory node {:?} deleted.", args.name))
    }
}

/// Arguments accepted by [`LinkMemoryTool`].
#[derive(Deserialize, JsonSchema)]
pub struct LinkMemoryArgs {
    /// Name of the source node.
    pub from: String,
    /// Name of the target node.
    pub to: String,
    /// Relation label (free-form).
    pub relation: String,
    /// Optional description of the edge.
    #[serde(default)]
    pub description: Option<String>,
}

/// Creates (or re-describes) a directed edge between two nodes.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = LinkMemoryArgs,
    invoke = link,
    name = "link_memory",
    description = "Create a directed, typed edge between two existing memory nodes (from -[relation]-> to). Linking the same pair with the same relation again updates the description.",
)]
pub struct LinkMemoryTool {
    context: KgMemoryContext,
}

impl LinkMemoryTool {
    pub fn new(context: KgMemoryContext) -> Self {
        Self { context }
    }

    async fn link(&self, args: LinkMemoryArgs) -> Result<String, LLMYError> {
        let Some(from) = self.context.db.node_by_name(&args.from).await? else {
            return Ok(format!("link_memory failed: no node named {:?}", args.from));
        };
        let Some(to) = self.context.db.node_by_name(&args.to).await? else {
            return Ok(format!("link_memory failed: no node named {:?}", args.to));
        };
        self.context
            .db
            .upsert_edge(from.id, to.id, &args.relation, args.description.as_deref())
            .await?;
        self.context.count_write();
        Ok(format!(
            "Edge {} -[{}]-> {} recorded.",
            args.from, args.relation, args.to
        ))
    }
}

/// Arguments accepted by [`UnlinkMemoryTool`].
#[derive(Deserialize, JsonSchema)]
pub struct UnlinkMemoryArgs {
    /// Name of the source node.
    pub from: String,
    /// Name of the target node.
    pub to: String,
    /// Relation to remove; omit to remove every edge between the two nodes.
    #[serde(default)]
    pub relation: Option<String>,
}

/// Removes edges between two nodes.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = UnlinkMemoryArgs,
    invoke = unlink,
    name = "unlink_memory",
    description = "Remove the edge(s) from one memory node to another, optionally restricted to one relation label.",
)]
pub struct UnlinkMemoryTool {
    context: KgMemoryContext,
}

impl UnlinkMemoryTool {
    pub fn new(context: KgMemoryContext) -> Self {
        Self { context }
    }

    async fn unlink(&self, args: UnlinkMemoryArgs) -> Result<String, LLMYError> {
        let Some(from) = self.context.db.node_by_name(&args.from).await? else {
            return Ok(format!(
                "unlink_memory failed: no node named {:?}",
                args.from
            ));
        };
        let Some(to) = self.context.db.node_by_name(&args.to).await? else {
            return Ok(format!("unlink_memory failed: no node named {:?}", args.to));
        };
        let removed = self
            .context
            .db
            .delete_edges(from.id, to.id, args.relation.as_deref())
            .await?;
        Ok(format!("Removed {removed} edge(s)."))
    }
}

/// Arguments accepted by [`ReadMemoryTool`].
#[derive(Deserialize, JsonSchema)]
pub struct ReadMemoryArgs {
    /// Name of the node to read.
    pub name: String,
}

/// Reads a node in full, with its neighborhood.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = ReadMemoryArgs,
    invoke = read,
    name = "read_memory",
    description = "Read a memory node in full: level, summary, content, and every edge touching it (with neighbor summaries).",
)]
pub struct ReadMemoryTool {
    context: KgMemoryContext,
}

impl ReadMemoryTool {
    pub fn new(context: KgMemoryContext) -> Self {
        Self { context }
    }

    async fn read(&self, args: ReadMemoryArgs) -> Result<String, LLMYError> {
        let Some(node) = self.context.db.node_by_name(&args.name).await? else {
            return Ok(format!("read_memory failed: no node named {:?}", args.name));
        };
        let edges = self.context.db.edges_of(node.id).await?;
        let mut out = format!(
            "name: {}\nlevel: {}\nupdated_at: {}\nsummary: {}\ncontent:\n{}",
            node.name, node.level, node.updated_at, node.summary, node.content
        );
        if !edges.is_empty() {
            let rendered = edges
                .iter()
                .map(|edge| format!("- {}", edge.render()))
                .collect::<Vec<_>>()
                .join("\n");
            out.push_str(&format!("\nedges:\n{rendered}"));
        }
        Ok(out)
    }
}

/// Arguments accepted by [`ListMemoryTool`].
#[derive(Deserialize, JsonSchema)]
pub struct ListMemoryArgs {
    /// Only list nodes at this level; omit for all levels.
    #[serde(default)]
    pub level: Option<String>,
}

/// Lists the memory index (optionally one level).
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = ListMemoryArgs,
    invoke = list,
    name = "list_memory",
    description = "List memory nodes (name, level, one-line summary), optionally filtered to one level.",
)]
pub struct ListMemoryTool {
    context: KgMemoryContext,
}

impl ListMemoryTool {
    pub fn new(context: KgMemoryContext) -> Self {
        Self { context }
    }

    async fn list(&self, args: ListMemoryArgs) -> Result<String, LLMYError> {
        let nodes = match args.level.as_deref() {
            Some(level) => self.context.db.nodes_by_level(level).await?,
            None => self.context.db.all_nodes().await?,
        };
        if nodes.is_empty() {
            return Ok("No memory nodes found.".to_string());
        }
        let lines = nodes
            .iter()
            .map(|node| format!("- [{}] {}: {}", node.level, node.name, node.summary))
            .collect::<Vec<_>>();
        Ok(format!("{} node(s):\n{}", nodes.len(), lines.join("\n")))
    }
}

/// Arguments accepted by [`GrepMemoryTool`].
#[derive(Deserialize, JsonSchema)]
pub struct GrepMemoryArgs {
    /// Regular expression matched against node names, summaries and
    /// contents.
    pub pattern: String,
    /// Case-insensitive matching.
    #[serde(default)]
    pub case_insensitive: Option<bool>,
}

/// Regex search over the memory graph.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = GrepMemoryArgs,
    invoke = grep,
    name = "grep_memory",
    description = "Search memory nodes by regular expression over their names, summaries and contents. Returns matching nodes with their level and summary.",
)]
pub struct GrepMemoryTool {
    context: KgMemoryContext,
}

impl GrepMemoryTool {
    pub fn new(context: KgMemoryContext) -> Self {
        Self { context }
    }

    async fn grep(&self, args: GrepMemoryArgs) -> Result<String, LLMYError> {
        let pattern = if args.case_insensitive.unwrap_or(false) {
            format!("(?i){}", args.pattern)
        } else {
            args.pattern.clone()
        };
        let regex = match regex::Regex::new(&pattern) {
            Ok(regex) => regex,
            Err(error) => {
                return Ok(format!("grep_memory failed: invalid pattern: {error}"));
            }
        };
        let nodes = self.context.db.all_nodes().await?;
        let matches = nodes
            .iter()
            .filter(|node| {
                regex.is_match(&node.name)
                    || regex.is_match(&node.summary)
                    || regex.is_match(&node.content)
            })
            .map(|node| format!("- [{}] {}: {}", node.level, node.name, node.summary))
            .collect::<Vec<_>>();
        if matches.is_empty() {
            return Ok(format!("No memory nodes match {:?}.", args.pattern));
        }
        Ok(format!(
            "{} matching node(s) (use read_memory for full content):\n{}",
            matches.len(),
            matches.join("\n")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn open_context() -> (tempfile::TempDir, KgMemoryContext) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("memory.sqlite3");
        let db = KgMemoryDB::open(&path.display().to_string())
            .await
            .expect("open kg db");
        (dir, KgMemoryContext::new(db))
    }

    #[tokio::test]
    async fn write_link_read_roundtrip_counts_writes() {
        let (_dir, context) = open_context().await;
        let write = WriteMemoryTool::new(context.clone());
        let link = LinkMemoryTool::new(context.clone());
        let read = ReadMemoryTool::new(context.clone());

        for (name, level) in [("proj", "project"), ("bug-1", "finding")] {
            let out = write
                .write(WriteMemoryArgs {
                    name: name.to_string(),
                    level: level.to_string(),
                    summary: format!("{name} summary"),
                    content: format!("{name} content"),
                    links: None,
                })
                .await
                .expect("write");
            assert!(out.contains("created"), "{out}");
        }
        let out = link
            .link(LinkMemoryArgs {
                from: "bug-1".to_string(),
                to: "proj".to_string(),
                relation: "found_in".to_string(),
                description: None,
            })
            .await
            .expect("link");
        assert!(out.contains("recorded"), "{out}");
        assert_eq!(context.write_count(), 3);

        let rendered = read
            .read(ReadMemoryArgs {
                name: "bug-1".to_string(),
            })
            .await
            .expect("read");
        assert!(rendered.contains("bug-1 content"), "{rendered}");
        assert!(rendered.contains("bug-1 -[found_in]-> proj"), "{rendered}");
    }

    #[tokio::test]
    async fn duplicate_write_is_a_soft_failure_and_does_not_count() {
        let (_dir, context) = open_context().await;
        let write = WriteMemoryTool::new(context.clone());
        let args = || WriteMemoryArgs {
            name: "n".to_string(),
            level: "l".to_string(),
            summary: "s".to_string(),
            content: "c".to_string(),
            links: None,
        };
        write.write(args()).await.expect("first write");
        let out = write.write(args()).await.expect("second write");
        assert!(out.contains("already exists"), "{out}");
        assert_eq!(context.write_count(), 1);
    }

    #[tokio::test]
    async fn index_groups_by_level_and_renders_edges() {
        let (_dir, context) = open_context().await;
        let write = WriteMemoryTool::new(context.clone());
        write
            .write(WriteMemoryArgs {
                name: "a".to_string(),
                level: "core".to_string(),
                summary: "a sum".to_string(),
                content: "a".to_string(),
                links: None,
            })
            .await
            .expect("write a");
        write
            .write(WriteMemoryArgs {
                name: "b".to_string(),
                level: "detail".to_string(),
                summary: "b sum".to_string(),
                content: "b".to_string(),
                links: Some(vec![MemoryLinkSpec {
                    to: "a".to_string(),
                    relation: "part_of".to_string(),
                    description: None,
                }]),
            })
            .await
            .expect("write b");

        let index = context.render_index(10_000).await.expect("index");
        assert!(index.contains("[core]"), "{index}");
        assert!(index.contains("[detail]"), "{index}");
        assert!(index.contains("- a: a sum"), "{index}");
        assert!(index.contains("b -[part_of]-> a"), "{index}");

        let tiny = context.render_index(20).await.expect("tiny index");
        assert!(tiny.contains("truncated"), "{tiny}");
    }

    #[tokio::test]
    async fn delete_removes_node_and_edges() {
        let (_dir, context) = open_context().await;
        let write = WriteMemoryTool::new(context.clone());
        for name in ["x", "y"] {
            write
                .write(WriteMemoryArgs {
                    name: name.to_string(),
                    level: "l".to_string(),
                    summary: "s".to_string(),
                    content: "c".to_string(),
                    links: None,
                })
                .await
                .expect("write");
        }
        LinkMemoryTool::new(context.clone())
            .link(LinkMemoryArgs {
                from: "x".to_string(),
                to: "y".to_string(),
                relation: "r".to_string(),
                description: None,
            })
            .await
            .expect("link");
        DeleteMemoryTool::new(context.clone())
            .delete(DeleteMemoryArgs {
                name: "y".to_string(),
            })
            .await
            .expect("delete");
        assert!(context.db.node_by_name("y").await.expect("query").is_none());
        assert!(context.db.all_edges().await.expect("edges").is_empty());
    }
}
