use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

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
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AgentMemory {
    /// Long-term memory persists all ites contents in the prompt
    pub long_term: BTreeMap<String, AgentMemoryContent>,
    /// Short-term memory only have their titles in the prompt.
    /// If needed, an agent needs to call tools to expand the memories.
    pub short_term: BTreeMap<String, AgentMemoryContent>,
}

impl AgentMemory {}
