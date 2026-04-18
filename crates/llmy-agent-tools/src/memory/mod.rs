pub mod embed;

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    fmt,
    ops::Deref,
    sync::Arc,
};

use color_eyre::eyre::eyre;
use llmy_agent::{LLMYError, tool::ToolBox};
use llmy_agent_derive::tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::memory::embed::{Embeding, SimilarityModel};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentMemoryContent {
    /// The title of the memory and should be concise, i.e., less than 128 characters in English.
    pub title: String,
    /// The related context of the memory.
    pub related_context: String,
    /// In which scenario an agent should read this memory? Useful for short-term memories.
    pub trigger_scenario: String,
    /// The actual content of memories. For long-term memory, the content should be concise and structured.
    /// For short-term memory, this could be long and short-term memories content could be folded during
    /// the conversation compaction.
    pub content: String,
    /// Raw content is the very verbose raw contents. In most cases, raw contents are not shown unless details
    /// are left out.
    /// During compaction, `content` is the summarized and `raw_content` is the raw conversation including original
    /// tool calls.
    pub raw_content: Option<String>,
}

impl AgentMemoryContent {
    pub fn render_full(&self) -> String {
        format!(
            "title: {}\nrelated_context: {}\ntrigger_scenario: {}\ncontent:\n{}",
            self.title, self.related_context, self.trigger_scenario, self.content
        )
    }

    pub fn render_short_term_memory_entry(&self) -> String {
        format!(
            "title: {}\nrelated_context: {}\ntrigger_scenario: {}",
            self.title, self.related_context, self.trigger_scenario
        )
    }

    fn render_raw_content(&self) -> Option<&str> {
        self.raw_content.as_deref()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AgentMemory {
    /// Long-term memory persists all its contents in the prompt
    pub long_term: BTreeMap<String, AgentMemoryContent>,
    /// Short-term memory only have their titles in the prompt.
    /// If needed, an agent needs to call tools to expand the memories.
    pub short_term: BTreeMap<String, AgentMemoryContent>,
}

pub struct AgentMemoryContextInner {
    pub memory: RwLock<AgentMemory>,
    pub embed: SimilarityModel,
    title_embed_cache: RwLock<BTreeMap<String, Embeding>>,
}

#[derive(Clone)]
pub struct AgentMemoryContext {
    inner: Arc<AgentMemoryContextInner>,
}

impl fmt::Debug for AgentMemoryContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentMemoryContext").finish_non_exhaustive()
    }
}

impl Deref for AgentMemoryContext {
    type Target = AgentMemoryContextInner;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl AgentMemoryContext {
    pub fn new(memory: AgentMemory, embed: SimilarityModel) -> Self {
        Self {
            inner: Arc::new(AgentMemoryContextInner {
                memory: RwLock::new(memory),
                embed,
                title_embed_cache: RwLock::new(BTreeMap::new()),
            }),
        }
    }

    pub fn tool_box(&self) -> ToolBox {
        let mut tools = ToolBox::new();
        tools.add_tool(ListMemoriesTool::new(self.clone()));
        tools.add_tool(ReadMemoryTool::new(self.clone()));
        tools.add_tool(ReadMemoryRawTool::new(self.clone()));
        tools.add_tool(SearchMemoryTool::new(self.clone()));
        tools.add_tool(DeleteMemoryTool::new(self.clone()));
        tools.add_tool(WriteMemoryTool::new(self.clone()));
        tools.add_tool(UpdateMemoryTool::new(self.clone()));
        tools
    }

    async fn write_memory(
        &self,
        memory_content: AgentMemoryContent,
        is_long_term: bool,
    ) -> MemoryWriteResult {
        let mut memory = self.memory.write().await;
        let title = memory_content.title.clone();

        if memory.short_term.contains_key(&title) || memory.long_term.contains_key(&title) {
            return MemoryWriteResult::AlreadyExists;
        }

        if is_long_term {
            memory
                .long_term
                .insert(memory_content.title.clone(), memory_content);
            MemoryWriteResult::Stored {
                scope: MemoryScope::LongTerm,
            }
        } else {
            memory
                .short_term
                .insert(memory_content.title.clone(), memory_content);
            MemoryWriteResult::Stored {
                scope: MemoryScope::ShortTerm,
            }
        }
    }

    pub async fn read_memory(&self, title: &str) -> Option<AgentMemoryContent> {
        let memory = self.memory.read().await;

        memory
            .short_term
            .get(title)
            .cloned()
            .or_else(|| memory.long_term.get(title).cloned())
    }

    async fn update_memory(
        &self,
        title: &str,
        related_context: Option<String>,
        trigger_scenario: Option<String>,
        content: Option<String>,
        raw_content: Option<String>,
    ) -> MemoryUpdateResult {
        let mut memory = self.memory.write().await;

        let target = if let Some(memory_content) = memory.short_term.get_mut(title) {
            Some((MemoryScope::ShortTerm, memory_content))
        } else {
            memory
                .long_term
                .get_mut(title)
                .map(|memory_content| (MemoryScope::LongTerm, memory_content))
        };

        let Some((scope, memory_content)) = target else {
            return MemoryUpdateResult::NotFound;
        };

        let mut updated_fields = Vec::new();

        if let Some(related_context) = related_context {
            memory_content.related_context = related_context;
            updated_fields.push("related_context");
        }

        if let Some(trigger_scenario) = trigger_scenario {
            memory_content.trigger_scenario = trigger_scenario;
            updated_fields.push("trigger_scenario");
        }

        if let Some(content) = content {
            memory_content.content = content;
            updated_fields.push("content");
        }

        if let Some(raw_content) = raw_content {
            memory_content.raw_content = Some(raw_content);
            updated_fields.push("raw_content");
        }

        if updated_fields.is_empty() {
            MemoryUpdateResult::NoChanges
        } else {
            MemoryUpdateResult::Updated {
                scope,
                updated_fields,
            }
        }
    }

    async fn delete_memory(&self, title: &str) -> MemoryDeleteResult {
        let deleted_scope = {
            let mut memory = self.memory.write().await;

            if memory.short_term.remove(title).is_some() {
                Some(MemoryScope::ShortTerm)
            } else if memory.long_term.remove(title).is_some() {
                Some(MemoryScope::LongTerm)
            } else {
                None
            }
        };

        match deleted_scope {
            Some(scope) => {
                self.title_embed_cache.write().await.remove(title);
                MemoryDeleteResult::Deleted { scope }
            }
            None => MemoryDeleteResult::NotFound,
        }
    }

    pub async fn list_memories(&self, memories_per_page: usize, page_index: usize) -> Vec<String> {
        if memories_per_page == 0 {
            return Vec::new();
        }

        let titles = {
            let memory = self.memory.read().await;
            collect_memory_titles(&memory)
        };

        let start = page_index.saturating_mul(memories_per_page);
        if start >= titles.len() {
            return Vec::new();
        }

        let end = (start + memories_per_page).min(titles.len());
        titles[start..end].to_vec()
    }

    pub async fn search_memory(&self, query: &str) -> Result<Vec<String>, LLMYError> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let titles = {
            let memory = self.memory.read().await;
            collect_memory_titles(&memory)
        };

        if titles.is_empty() {
            return Ok(Vec::new());
        }

        let query_embedding = self
            .embed
            .batch_embed(vec![query.to_string()])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| eyre!("local embedding model returned no query embedding"))?;
        let title_embeddings = self.cached_title_embeddings(&titles).await?;
        let scores = query_embedding.calculate_similarity(&title_embeddings);

        let mut ranked = titles.into_iter().zip(scores).collect::<Vec<_>>();
        ranked.sort_by(|(title_a, score_a), (title_b, score_b)| {
            score_b
                .partial_cmp(score_a)
                .unwrap_or(Ordering::Equal)
                .then_with(|| title_a.cmp(title_b))
        });

        Ok(ranked
            .into_iter()
            .map(|(title, _)| title)
            .collect::<Vec<_>>())
    }

    async fn cached_title_embeddings(&self, titles: &[String]) -> Result<Vec<Embeding>, LLMYError> {
        let missing_titles = {
            let cache = self.title_embed_cache.read().await;
            titles
                .iter()
                .filter(|title| !cache.contains_key(*title))
                .cloned()
                .collect::<Vec<_>>()
        };

        if !missing_titles.is_empty() {
            let embeddings = self.embed.batch_embed(missing_titles.clone()).await?;
            let mut cache = self.title_embed_cache.write().await;
            for (title, embedding) in missing_titles.into_iter().zip(embeddings) {
                cache.insert(title, embedding);
            }
        }

        let cache = self.title_embed_cache.read().await;
        titles
            .iter()
            .map(|title| {
                cache
                    .get(title)
                    .cloned()
                    .ok_or_else(|| eyre!("missing cached title embedding for {:?}", title).into())
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemoryScope {
    ShortTerm,
    LongTerm,
}

impl fmt::Display for MemoryScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShortTerm => f.write_str("short_term"),
            Self::LongTerm => f.write_str("long_term"),
        }
    }
}

enum MemoryWriteResult {
    Stored { scope: MemoryScope },
    AlreadyExists,
}

enum MemoryUpdateResult {
    Updated {
        scope: MemoryScope,
        updated_fields: Vec<&'static str>,
    },
    NoChanges,
    NotFound,
}

enum MemoryDeleteResult {
    Deleted { scope: MemoryScope },
    NotFound,
}

fn collect_memory_titles(memory: &AgentMemory) -> Vec<String> {
    let mut seen_titles = BTreeSet::new();
    let mut titles = Vec::with_capacity(memory.short_term.len() + memory.long_term.len());

    for title in memory.short_term.keys() {
        if seen_titles.insert(title.clone()) {
            titles.push(title.clone());
        }
    }

    for title in memory.long_term.keys() {
        if seen_titles.insert(title.clone()) {
            titles.push(title.clone());
        }
    }

    titles
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct WriteMemoryArgs {
    pub title: String,
    pub related_context: String,
    pub trigger_scenario: String,
    pub content: String,
    pub raw_content: Option<String>,
    pub is_long_term: bool,
}

#[derive(Debug, Clone)]
#[tool(
    arguments = WriteMemoryArgs,
    invoke = write_memory,
    name = "write_memory",
    description = "Write a memory entry into the shared agent memory context. `title` must be concise and ideally stay under 128 English characters. `related_context` should describe the surrounding context that makes this memory relevant. `trigger_scenario` should explain when the agent ought to read this memory later. `content` is the actual memory content; keep it concise and structured for long-term memory, while short-term memory may be longer and more detailed. `raw_content` is optional and can store the full detailed form, including original details and tool-call-heavy transcripts, so it may be very long. `is_long_term` controls where the memory is stored: true writes to long-term memory and false writes to short-term memory.",
)]
pub struct WriteMemoryTool {
    pub context: AgentMemoryContext,
}

impl WriteMemoryTool {
    pub fn new(context: AgentMemoryContext) -> Self {
        Self { context }
    }

    pub async fn write_memory(&self, args: WriteMemoryArgs) -> Result<String, LLMYError> {
        let memory_content = AgentMemoryContent {
            title: args.title,
            related_context: args.related_context,
            trigger_scenario: args.trigger_scenario,
            content: args.content,
            raw_content: args.raw_content,
        };
        let title = memory_content.title.clone();
        Ok(
            match self
                .context
                .write_memory(memory_content, args.is_long_term)
                .await
            {
                MemoryWriteResult::Stored { scope } => {
                    format!("Stored memory {:?} in {}", title, scope)
                }
                MemoryWriteResult::AlreadyExists => format!(
                    "Memory {:?} already exists. Do not replace it with write_memory; call update_memory instead.",
                    title
                ),
            },
        )
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UpdateMemoryArgs {
    pub title: String,
    pub related_context: Option<String>,
    pub trigger_scenario: Option<String>,
    pub content: Option<String>,
    pub raw_content: Option<String>,
}

#[derive(Debug, Clone)]
#[tool(
    arguments = UpdateMemoryArgs,
    invoke = update_memory,
    name = "update_memory",
    description = "Update an existing memory by exact `title` match in the shared agent memory context. `title` is required and is never changed. `related_context`, `trigger_scenario`, `content`, and `raw_content` are optional; only fields provided as Some values are updated, and omitted fields are left unchanged. `raw_content` is the full detailed form and may be very long. The lookup prefers short-term memory and falls back to long-term memory if the title does not exist in short-term memory.",
)]
pub struct UpdateMemoryTool {
    pub context: AgentMemoryContext,
}

impl UpdateMemoryTool {
    pub fn new(context: AgentMemoryContext) -> Self {
        Self { context }
    }

    pub async fn update_memory(&self, args: UpdateMemoryArgs) -> Result<String, LLMYError> {
        Ok(
            match self
                .context
                .update_memory(
                    &args.title,
                    args.related_context,
                    args.trigger_scenario,
                    args.content,
                    args.raw_content,
                )
                .await
            {
                MemoryUpdateResult::Updated {
                    scope,
                    updated_fields,
                } => format!(
                    "Updated memory {:?} in {}: {}",
                    args.title,
                    scope,
                    updated_fields.join(", ")
                ),
                MemoryUpdateResult::NoChanges => format!(
                    "Memory {:?} was found, but no fields were updated because all optional fields were None.",
                    args.title
                ),
                MemoryUpdateResult::NotFound => {
                    format!("No exact memory found with title {:?}", args.title)
                }
            },
        )
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReadMemoryArgs {
    pub title: String,
}

#[derive(Debug, Clone)]
#[tool(
    arguments = ReadMemoryArgs,
    invoke = read_memory,
    name = "read_memory",
    description = "Read a memory by exact `title` match from the shared agent memory context. This lookup is exact, not fuzzy. It searches short-term memory first and falls back to long-term memory only if no short-term entry has the same title. This tool returns the structured memory fields and does not include `raw_content`.",
)]
pub struct ReadMemoryTool {
    pub context: AgentMemoryContext,
}

impl ReadMemoryTool {
    pub fn new(context: AgentMemoryContext) -> Self {
        Self { context }
    }

    pub async fn read_memory(&self, args: ReadMemoryArgs) -> Result<String, LLMYError> {
        match self.context.read_memory(&args.title).await {
            Some(memory) => Ok(memory.render_full()),
            None => Ok(format!("No exact memory found with title {:?}", args.title)),
        }
    }
}

#[derive(Debug, Clone)]
#[tool(
    arguments = ReadMemoryArgs,
    invoke = read_memory_raw,
    name = "read_memory_raw",
    description = "Read only the `raw_content` for a memory by exact `title` match from the shared agent memory context. This lookup is exact, not fuzzy. It searches short-term memory first and falls back to long-term memory only if no short-term entry has the same title. `raw_content` is the full detailed form and may include all original details, so it can be very long.",
)]
pub struct ReadMemoryRawTool {
    pub context: AgentMemoryContext,
}

impl ReadMemoryRawTool {
    pub fn new(context: AgentMemoryContext) -> Self {
        Self { context }
    }

    pub async fn read_memory_raw(&self, args: ReadMemoryArgs) -> Result<String, LLMYError> {
        match self.context.read_memory(&args.title).await {
            Some(memory) => Ok(match memory.render_raw_content() {
                Some(raw_content) => raw_content.to_string(),
                None => format!("Memory {:?} has no raw content", args.title),
            }),
            None => Ok(format!("No exact memory found with title {:?}", args.title)),
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteMemoryArgs {
    pub title: String,
}

#[derive(Debug, Clone)]
#[tool(
    arguments = DeleteMemoryArgs,
    invoke = delete_memory,
    name = "delete_memory",
    description = "Delete a memory by exact `title` match from the shared agent memory context. The lookup prefers short-term memory and falls back to long-term memory if the title does not exist in short-term memory. If no exact title exists, the tool explicitly reports that nothing was deleted.",
)]
pub struct DeleteMemoryTool {
    pub context: AgentMemoryContext,
}

impl DeleteMemoryTool {
    pub fn new(context: AgentMemoryContext) -> Self {
        Self { context }
    }

    pub async fn delete_memory(&self, args: DeleteMemoryArgs) -> Result<String, LLMYError> {
        Ok(match self.context.delete_memory(&args.title).await {
            MemoryDeleteResult::Deleted { scope } => {
                format!("Deleted memory {:?} from {}", args.title, scope)
            }
            MemoryDeleteResult::NotFound => {
                format!("No exact memory found with title {:?}", args.title)
            }
        })
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListMemoriesArgs {
    pub memories_per_page: usize,
    pub page_index: usize,
}

#[derive(Debug, Clone)]
#[tool(
    arguments = ListMemoriesArgs,
    invoke = list_memories,
    name = "list_memories",
    description = "List memory titles from the shared agent memory context with pagination. `memories_per_page` is the number of titles to return per page and should be greater than zero. `page_index` is zero-based. The result contains titles only, one per line, with no memory content included.",
)]
pub struct ListMemoriesTool {
    pub context: AgentMemoryContext,
}

impl ListMemoriesTool {
    pub fn new(context: AgentMemoryContext) -> Self {
        Self { context }
    }

    pub async fn list_memories(&self, args: ListMemoriesArgs) -> Result<String, LLMYError> {
        if args.memories_per_page == 0 {
            return Ok("memories_per_page must be greater than 0".to_string());
        }

        let titles = self
            .context
            .list_memories(args.memories_per_page, args.page_index)
            .await;

        if titles.is_empty() {
            Ok(format!(
                "No memories found for page_index {} with memories_per_page {}",
                args.page_index, args.memories_per_page
            ))
        } else {
            Ok(titles.join("\n"))
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchMemoryArgs {
    pub query: String,
}

#[derive(Debug, Clone)]
#[tool(
    arguments = SearchMemoryArgs,
    invoke = search_memory,
    name = "search_memory",
    description = "Search the shared agent memory context semantically using embeddings. `query` should describe the concept or scenario you want to find. The result contains only matching memory titles, one title per line, so the caller can fetch full details later through `read_memory`.",
)]
pub struct SearchMemoryTool {
    pub context: AgentMemoryContext,
}

impl SearchMemoryTool {
    pub fn new(context: AgentMemoryContext) -> Self {
        Self { context }
    }

    pub async fn search_memory(&self, args: SearchMemoryArgs) -> Result<String, LLMYError> {
        Ok(self.context.search_memory(&args.query).await?.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embed::SimilarityModelConfig;
    use std::sync::LazyLock;
    use tempfile::TempDir;
    use tokio::sync::OnceCell;

    static CACHE_DIR: LazyLock<TempDir> =
        LazyLock::new(|| tempfile::tempdir().expect("failed to create temp cache dir"));
    static MODEL: OnceCell<SimilarityModel> = OnceCell::const_new();

    async fn shared_model() -> SimilarityModel {
        MODEL
            .get_or_init(|| async {
                SimilarityModel::new(SimilarityModelConfig {
                    cache_dir: Some(CACHE_DIR.path().to_path_buf()),
                    ..Default::default()
                })
                .await
                .expect("failed to initialize shared similarity model")
            })
            .await
            .clone()
    }

    async fn new_context() -> AgentMemoryContext {
        AgentMemoryContext::new(AgentMemory::default(), shared_model().await)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn read_memory_prefers_short_term_over_long_term() {
        let mut memory = AgentMemory::default();
        memory.long_term.insert(
            "rust async".to_string(),
            AgentMemoryContent {
                title: "rust async".to_string(),
                related_context: "long term".to_string(),
                trigger_scenario: "planning".to_string(),
                content: "long-term memory".to_string(),
                raw_content: None,
            },
        );
        memory.short_term.insert(
            "rust async".to_string(),
            AgentMemoryContent {
                title: "rust async".to_string(),
                related_context: "short term".to_string(),
                trigger_scenario: "current task".to_string(),
                content: "short-term memory".to_string(),
                raw_content: None,
            },
        );

        let context = AgentMemoryContext::new(memory, shared_model().await);
        let reader = ReadMemoryTool::new(context);

        let rendered = reader
            .read_memory(ReadMemoryArgs {
                title: "rust async".to_string(),
            })
            .await
            .unwrap();

        assert!(rendered.contains("related_context: short term"));
        assert!(rendered.contains("content:\nshort-term memory"));
        assert!(!rendered.contains("content:\nlong-term memory"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn read_memory_omits_raw_content_and_raw_tool_returns_it() {
        let context = new_context().await;
        let writer = WriteMemoryTool::new(context.clone());
        let reader = ReadMemoryTool::new(context.clone());
        let raw_reader = ReadMemoryRawTool::new(context);

        writer
            .write_memory(WriteMemoryArgs {
                title: "incident summary".to_string(),
                related_context: "prod outage".to_string(),
                trigger_scenario: "postmortem".to_string(),
                content: "summarized memory".to_string(),
                raw_content: Some("very long raw transcript".to_string()),
                is_long_term: false,
            })
            .await
            .unwrap();

        let rendered = reader
            .read_memory(ReadMemoryArgs {
                title: "incident summary".to_string(),
            })
            .await
            .unwrap();
        let raw_rendered = raw_reader
            .read_memory_raw(ReadMemoryArgs {
                title: "incident summary".to_string(),
            })
            .await
            .unwrap();

        assert!(rendered.contains("content:\nsummarized memory"));
        assert!(!rendered.contains("raw_content:"));
        assert!(!rendered.contains("very long raw transcript"));
        assert_eq!(raw_rendered, "very long raw transcript");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn read_memory_raw_reports_missing_raw_content() {
        let context = new_context().await;
        let writer = WriteMemoryTool::new(context.clone());
        let raw_reader = ReadMemoryRawTool::new(context);

        writer
            .write_memory(WriteMemoryArgs {
                title: "rust async runtime".to_string(),
                related_context: "systems".to_string(),
                trigger_scenario: "debugging".to_string(),
                content: "tokio notes".to_string(),
                raw_content: None,
                is_long_term: false,
            })
            .await
            .unwrap();

        let rendered = raw_reader
            .read_memory_raw(ReadMemoryArgs {
                title: "rust async runtime".to_string(),
            })
            .await
            .unwrap();

        assert_eq!(rendered, "Memory \"rust async runtime\" has no raw content");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn write_memory_refuses_to_replace_existing_memory() {
        let context = new_context().await;
        let writer = WriteMemoryTool::new(context.clone());
        let reader = ReadMemoryTool::new(context);

        let first_result = writer
            .write_memory(WriteMemoryArgs {
                title: "rust async".to_string(),
                related_context: "initial".to_string(),
                trigger_scenario: "planning".to_string(),
                content: "first version".to_string(),
                raw_content: None,
                is_long_term: false,
            })
            .await
            .unwrap();
        let second_result = writer
            .write_memory(WriteMemoryArgs {
                title: "rust async".to_string(),
                related_context: "replacement".to_string(),
                trigger_scenario: "planning".to_string(),
                content: "second version".to_string(),
                raw_content: None,
                is_long_term: true,
            })
            .await
            .unwrap();
        let rendered = reader
            .read_memory(ReadMemoryArgs {
                title: "rust async".to_string(),
            })
            .await
            .unwrap();

        assert!(first_result.contains("Stored memory"));
        assert!(second_result.contains("already exists"));
        assert!(second_result.contains("update_memory"));
        assert!(rendered.contains("content:\nfirst version"));
        assert!(!rendered.contains("content:\nsecond version"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn read_memory_uses_exact_title_match() {
        let context = new_context().await;
        let writer = WriteMemoryTool::new(context.clone());
        let reader = ReadMemoryTool::new(context);

        writer
            .write_memory(WriteMemoryArgs {
                title: "rust async runtime".to_string(),
                related_context: "systems".to_string(),
                trigger_scenario: "debugging".to_string(),
                content: "tokio notes".to_string(),
                raw_content: None,
                is_long_term: false,
            })
            .await
            .unwrap();

        let rendered = reader
            .read_memory(ReadMemoryArgs {
                title: "rust async".to_string(),
            })
            .await
            .unwrap();

        assert_eq!(rendered, "No exact memory found with title \"rust async\"");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn update_memory_changes_only_some_fields() {
        let context = new_context().await;
        let writer = WriteMemoryTool::new(context.clone());
        let updater = UpdateMemoryTool::new(context.clone());
        let reader = ReadMemoryTool::new(context.clone());
        let raw_reader = ReadMemoryRawTool::new(context);

        writer
            .write_memory(WriteMemoryArgs {
                title: "rust async runtime".to_string(),
                related_context: "systems".to_string(),
                trigger_scenario: "debugging".to_string(),
                content: "tokio notes".to_string(),
                raw_content: Some("original raw details".to_string()),
                is_long_term: false,
            })
            .await
            .unwrap();

        let update_result = updater
            .update_memory(UpdateMemoryArgs {
                title: "rust async runtime".to_string(),
                related_context: None,
                trigger_scenario: Some("incident response".to_string()),
                content: Some("updated tokio notes".to_string()),
                raw_content: Some("updated raw details".to_string()),
            })
            .await
            .unwrap();
        let rendered = reader
            .read_memory(ReadMemoryArgs {
                title: "rust async runtime".to_string(),
            })
            .await
            .unwrap();
        let raw_rendered = raw_reader
            .read_memory_raw(ReadMemoryArgs {
                title: "rust async runtime".to_string(),
            })
            .await
            .unwrap();

        assert!(update_result.contains("Updated memory"));
        assert!(update_result.contains("trigger_scenario, content, raw_content"));
        assert!(rendered.contains("related_context: systems"));
        assert!(rendered.contains("trigger_scenario: incident response"));
        assert!(rendered.contains("content:\nupdated tokio notes"));
        assert_eq!(raw_rendered, "updated raw details");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn list_memories_returns_titles_only_by_page() {
        let context = new_context().await;
        let writer = WriteMemoryTool::new(context.clone());
        let lister = ListMemoriesTool::new(context);

        for title in ["alpha", "beta", "gamma"] {
            writer
                .write_memory(WriteMemoryArgs {
                    title: title.to_string(),
                    related_context: "context".to_string(),
                    trigger_scenario: "trigger".to_string(),
                    content: "content".to_string(),
                    raw_content: None,
                    is_long_term: false,
                })
                .await
                .unwrap();
        }

        let first_page = lister
            .list_memories(ListMemoriesArgs {
                memories_per_page: 2,
                page_index: 0,
            })
            .await
            .unwrap();
        let second_page = lister
            .list_memories(ListMemoriesArgs {
                memories_per_page: 2,
                page_index: 1,
            })
            .await
            .unwrap();

        assert_eq!(first_page, "alpha\nbeta");
        assert_eq!(second_page, "gamma");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn search_memory_uses_title_embeddings_and_caches_them() {
        let context = new_context().await;
        let writer = WriteMemoryTool::new(context.clone());
        let searcher = SearchMemoryTool::new(context);

        writer
            .write_memory(WriteMemoryArgs {
                title: "Rust async notes".to_string(),
                related_context: "irrelevant".to_string(),
                trigger_scenario: "building async services".to_string(),
                content: "banana recipe content that should not help search".to_string(),
                raw_content: None,
                is_long_term: false,
            })
            .await
            .unwrap();
        writer
            .write_memory(WriteMemoryArgs {
                title: "Banana bread".to_string(),
                related_context: "tokio and futures".to_string(),
                trigger_scenario: "cooking".to_string(),
                content: "Use tokio runtime for async rust workloads.".to_string(),
                raw_content: None,
                is_long_term: true,
            })
            .await
            .unwrap();

        let titles = searcher
            .search_memory(SearchMemoryArgs {
                query: "tokio async runtime".to_string(),
            })
            .await
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect::<Vec<_>>();

        assert_eq!(titles.first().map(String::as_str), Some("Rust async notes"));
        assert_eq!(titles.len(), 2);
        assert!(titles.iter().all(|title| !title.contains("content:")));
        assert!(
            titles
                .iter()
                .all(|title| !title.contains("related_context:"))
        );

        let cached_titles = searcher
            .context
            .title_embed_cache
            .read()
            .await
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            cached_titles,
            vec!["Banana bread".to_string(), "Rust async notes".to_string()]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn delete_memory_removes_memory_and_cached_title_embedding() {
        let context = new_context().await;
        let writer = WriteMemoryTool::new(context.clone());
        let searcher = SearchMemoryTool::new(context.clone());
        let deleter = DeleteMemoryTool::new(context.clone());
        let reader = ReadMemoryTool::new(context.clone());

        writer
            .write_memory(WriteMemoryArgs {
                title: "Rust async notes".to_string(),
                related_context: "tokio".to_string(),
                trigger_scenario: "services".to_string(),
                content: "notes".to_string(),
                raw_content: None,
                is_long_term: false,
            })
            .await
            .unwrap();

        searcher
            .search_memory(SearchMemoryArgs {
                query: "rust async".to_string(),
            })
            .await
            .unwrap();
        assert!(
            searcher
                .context
                .title_embed_cache
                .read()
                .await
                .contains_key("Rust async notes")
        );

        let delete_result = deleter
            .delete_memory(DeleteMemoryArgs {
                title: "Rust async notes".to_string(),
            })
            .await
            .unwrap();
        let read_result = reader
            .read_memory(ReadMemoryArgs {
                title: "Rust async notes".to_string(),
            })
            .await
            .unwrap();

        assert!(delete_result.contains("Deleted memory"));
        assert_eq!(
            read_result,
            "No exact memory found with title \"Rust async notes\""
        );
        assert!(
            !searcher
                .context
                .title_embed_cache
                .read()
                .await
                .contains_key("Rust async notes")
        );
    }
}
