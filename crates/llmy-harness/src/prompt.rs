pub const DEFAULT_LONG_TERM_MEMORY_CRITERIA: &[&str] = &[
    "Store durable knowledge that should still matter after compaction or after the immediate task ends.",
    "Prefer long-term memory for stable codebase structure, important file roles, architecture, recurring workflows, reusable commands, and persistent user or project preferences.",
    "Prefer long-term memory for validated learnings, fixes, and constraints that are likely to help in future tasks, not just the current turn.",
    "Do not put transient progress updates, temporary hypotheses, or step-by-step scratch work into long-term memory unless they became a stable rule or invariant.",
    "Keep long-term memory concise, structured, and deduplicated. It should cover durable files and functions, workflow, system documentation, persistent constraints, and validated learnings.",
];

pub const DEFAULT_SHORT_TERM_MEMORY_CRITERIA: &[&str] = &[
    "Store short-term memory for conversation-local state that is critical right now but may become stale soon.",
    "Prefer short-term memory for the active task, current state, pending next steps, temporary plans, recent errors, recent corrections, and intermediate results.",
    "Use short-term memory for dense context that helps resume work after compaction, including current state, task specification, recent errors and corrections, key results, and a terse worklog.",
    "Do not keep short-term memory for facts that are already stable and broadly reusable across future sessions; those should move into long-term memory instead.",
    "Short-term memory may be more verbose than long-term memory, and can keep raw detail when that detail is still operationally useful.",
];

pub const DEFAULT_LONG_TERM_MEMORY_OPERATOR: &[&str] = &[
    "When writing long-term memory, compress the information into durable, reusable guidance instead of copying the full conversation.",
    "If a long-term memory with the same meaning already exists, update it instead of creating a duplicate title.",
    "Prefer merging related stable facts into an existing memory when that improves future retrieval and avoids fragmentation.",
    "Preserve only the minimal raw detail needed to justify or reconstruct the durable insight; otherwise keep raw content empty.",
    "Write long-term memory so that another agent can understand the rule, workflow, or system fact without rereading the original chat.",
];

pub const DEFAULT_SHORT_TERM_MEMORY_OPERATOR: &[&str] = &[
    "When writing short-term memory, optimize for continuity after compaction: preserve the latest state, blockers, and next actions.",
    "Update an existing short-term memory when the same active thread changes, instead of creating many overlapping memories with near-identical titles.",
    "Prefer keeping exact operational detail in raw_content when the detail may be needed to continue the current task: tool outputs, error text, and verbose reasoning artifacts.",
    "Aggressively refresh short-term memory when the user changes direction, when the implementation state changes, or when earlier assumptions are corrected.",
    "Short-term memory can be verbose, but the main content field should still summarize the essential state so it remains readable in prompts.",
];

pub const DEFAULT_LONG_TERM_MEMORY_TRIGGER: &[&str] = &[
    "Write or update long-term memory when a stable architectural fact, reusable workflow, durable constraint, or persistent preference has been confirmed.",
    "Write or update long-term memory after a bug fix or correction reveals a general lesson that should influence future work.",
    "Write or update long-term memory when repeated commands, file locations, or subsystem relationships become clear enough to be reused later.",
    "Write or update long-term memory when the information would still be valuable even if the current task and recent chat history disappeared.",
];

pub const DEFAULT_SHORT_TERM_MEMORY_TRIGGER: &[&str] = &[
    "Write or update short-term memory whenever the current task state materially changes.",
    "Write or update short-term memory when there are new pending next steps, new blockers, new user instructions, or important intermediate results.",
    "Write or update short-term memory after meaningful errors, failed attempts, or corrections, especially when that information is needed to avoid repeating mistakes in the same task.",
    "Write or update short-term memory before or during compaction so the agent can recover the active thread, current state, and recent worklog quickly.",
];

pub fn render_compact_system_prompt(memory_tools_available: bool) -> String {
    let mut sections = vec![
		"You are compacting an existing agent conversation so work can continue in a fresh context. You will receive a textual transcript of the prior conversation, including the system prompt, user messages, assistant messages, tool calls, and tool results.".to_string(),
		"Your final answer must be exactly one paragraph of plain text. It should preserve the active objective, important user constraints, key technical decisions, important edits or results, important errors and fixes, current state, blockers, and the most useful next action. Do not use bullets, headings, XML tags, or meta commentary about summarization.".to_string(),
		"Keep the paragraph dense and specific. Preserve operational detail that the next agent needs, but compress repeated chatter and incidental wording.".to_string(),
	];

    if memory_tools_available {
        sections.push("Memory tools are available in this run. Before the final paragraph, write or update memory entries for the key nodes of the conversation. Record stable reusable facts in long-term memory. Record the active thread, latest state, blockers, and near-term next steps in short-term memory. Use raw_content when verbose detail may still be useful later.".to_string());
    } else {
        sections.push("No memory tools are available in this run. Skip memory updates and produce the final paragraph directly.".to_string());
    }

    sections.join("\n\n")
}

pub fn render_compact_user_prompt(history_text: &str) -> String {
    format!(
        "Compact this conversation transcript for continuation in a fresh agent context. Preserve the key nodes of the work, including the user's current intent, important constraints, meaningful code or design decisions, important errors and fixes, current implementation state, and pending work. If memory tools are available, write or update memory before your final answer. Then return exactly one paragraph.\n\n<conversation>\n{history_text}\n</conversation>"
    )
}

pub fn render_compacted_context_message(summary: &str) -> String {
    format!("Compacted context: {summary}")
}
