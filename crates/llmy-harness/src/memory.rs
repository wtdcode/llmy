use llmy_agent_tools::memory::{AgentMemory, AgentMemoryContent};

use crate::prompt::{
    DEFAULT_LONG_TERM_MEMORY_CRITERIA, DEFAULT_LONG_TERM_MEMORY_OPERATOR,
    DEFAULT_LONG_TERM_MEMORY_TRIGGER, DEFAULT_SHORT_TERM_MEMORY_CRITERIA,
    DEFAULT_SHORT_TERM_MEMORY_OPERATOR, DEFAULT_SHORT_TERM_MEMORY_TRIGGER,
};

#[derive(Debug, Clone)]
pub struct AgentMemorySystemPromptCriteria {
    // these two criteria define how to classify the two memories
    pub long_term_memory_criteria: Vec<String>,
    pub short_term_memory_criteria: Vec<String>,

    // these two operators define how to write/update existing memories
    pub long_term_memory_operator: Vec<String>,
    pub short_term_memory_operator: Vec<String>,

    // these two triggers define when to write/update new memories
    pub long_term_memory_trigger: Vec<String>,
    pub short_term_memory_trigger: Vec<String>,
}

impl AgentMemorySystemPromptCriteria {
    pub fn builder() -> AgentMemorySystemPromptCriteriaBuilder {
        AgentMemorySystemPromptCriteriaBuilder::default()
    }

    fn render_section(title: &str, values: &[String]) -> String {
        let items = if values.is_empty() {
            "- None.".to_string()
        } else {
            values
                .iter()
                .map(|value| format!("- {value}"))
                .collect::<Vec<_>>()
                .join("\n")
        };

        format!("{title}:\n{items}")
    }

    pub fn render_instruction(&self) -> String {
        [
            "You are equipped with a shared memory system. Use the memory tools deliberately to preserve durable knowledge and active task state for long-running work.".to_string(),
            Self::render_section(
                "Long-term memory criteria",
                &self.long_term_memory_criteria,
            ),
            Self::render_section(
                "Short-term memory criteria",
                &self.short_term_memory_criteria,
            ),
            Self::render_section(
                "Long-term memory operators",
                &self.long_term_memory_operator,
            ),
            Self::render_section(
                "Short-term memory operators",
                &self.short_term_memory_operator,
            ),
            Self::render_section(
                "Long-term memory triggers",
                &self.long_term_memory_trigger,
            ),
            Self::render_section(
                "Short-term memory triggers",
                &self.short_term_memory_trigger,
            ),
        ]
        .join("\n\n")
    }

    pub fn append_to_system_prompt(&self, system_prompt: &str) -> String {
        let memory_instruction = self.render_instruction();
        if system_prompt.trim().is_empty() {
            memory_instruction
        } else {
            format!("{system_prompt}\n\n{memory_instruction}")
        }
    }

    pub fn render_system_prompt(&self, system_prompt: &str, memory: &AgentMemory) -> String {
        let mut rendered = self.append_to_system_prompt(system_prompt);

        if let Some(long_term_memory_section) =
            render_long_term_memory_section(memory.long_term.values())
        {
            rendered.push_str("\n\n");
            rendered.push_str(&long_term_memory_section);
        }

        if let Some(short_term_memory_section) =
            render_short_term_memory_section(memory.short_term.values())
        {
            rendered.push_str("\n\n");
            rendered.push_str(&short_term_memory_section);
        }

        rendered
    }
}

#[derive(Debug, Clone)]
pub struct AgentMemorySystemPromptCriteriaBuilder {
    criteria: AgentMemorySystemPromptCriteria,
}

impl Default for AgentMemorySystemPromptCriteriaBuilder {
    fn default() -> Self {
        Self {
            criteria: AgentMemorySystemPromptCriteria::default(),
        }
    }
}

impl AgentMemorySystemPromptCriteriaBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn append_long_term_memory_criteria(mut self, value: String) -> Self {
        self.criteria.long_term_memory_criteria.push(value);
        self
    }

    pub fn append_short_term_memory_criteria(mut self, value: String) -> Self {
        self.criteria.short_term_memory_criteria.push(value);
        self
    }

    pub fn append_long_term_memory_operator(mut self, value: String) -> Self {
        self.criteria.long_term_memory_operator.push(value);
        self
    }

    pub fn append_short_term_memory_operator(mut self, value: String) -> Self {
        self.criteria.short_term_memory_operator.push(value);
        self
    }

    pub fn append_long_term_memory_trigger(mut self, value: String) -> Self {
        self.criteria.long_term_memory_trigger.push(value);
        self
    }

    pub fn append_short_term_memory_trigger(mut self, value: String) -> Self {
        self.criteria.short_term_memory_trigger.push(value);
        self
    }

    pub fn build(self) -> AgentMemorySystemPromptCriteria {
        self.criteria
    }
}

impl Default for AgentMemorySystemPromptCriteria {
    fn default() -> Self {
        Self {
            long_term_memory_criteria: DEFAULT_LONG_TERM_MEMORY_CRITERIA
                .iter()
                .map(|v| (*v).to_string())
                .collect(),
            short_term_memory_criteria: DEFAULT_SHORT_TERM_MEMORY_CRITERIA
                .iter()
                .map(|v| (*v).to_string())
                .collect(),
            long_term_memory_operator: DEFAULT_LONG_TERM_MEMORY_OPERATOR
                .iter()
                .map(|v| (*v).to_string())
                .collect(),
            short_term_memory_operator: DEFAULT_SHORT_TERM_MEMORY_OPERATOR
                .iter()
                .map(|v| (*v).to_string())
                .collect(),
            long_term_memory_trigger: DEFAULT_LONG_TERM_MEMORY_TRIGGER
                .iter()
                .map(|v| (*v).to_string())
                .collect(),
            short_term_memory_trigger: DEFAULT_SHORT_TERM_MEMORY_TRIGGER
                .iter()
                .map(|v| (*v).to_string())
                .collect(),
        }
    }
}

fn render_long_term_memory_section<'a>(
    memories: impl IntoIterator<Item = &'a AgentMemoryContent>,
) -> Option<String> {
    let rendered = memories
        .into_iter()
        .map(AgentMemoryContent::render_full)
        .collect::<Vec<_>>();

    if rendered.is_empty() {
        None
    } else {
        Some(format!(
            "Long-term memory entries:\n\n{}",
            rendered.join("\n\n---\n\n")
        ))
    }
}

fn render_short_term_memory_section<'a>(
    memories: impl IntoIterator<Item = &'a AgentMemoryContent>,
) -> Option<String> {
    let rendered = memories
        .into_iter()
        .map(AgentMemoryContent::render_short_term_memory_entry)
        .collect::<Vec<_>>();

    if rendered.is_empty() {
        None
    } else {
        Some(format!(
            "Short-term memory entries:\n\n{}",
            rendered.join("\n\n---\n\n")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn criteria_builder_appends_to_each_section() {
        let criteria = AgentMemorySystemPromptCriteria::builder()
            .append_long_term_memory_criteria("lt criteria".to_string())
            .append_short_term_memory_criteria("st criteria".to_string())
            .append_long_term_memory_operator("lt operator".to_string())
            .append_short_term_memory_operator("st operator".to_string())
            .append_long_term_memory_trigger("lt trigger".to_string())
            .append_short_term_memory_trigger("st trigger".to_string())
            .build();

        assert_eq!(
            criteria
                .long_term_memory_criteria
                .last()
                .map(String::as_str),
            Some("lt criteria")
        );
        assert_eq!(
            criteria
                .short_term_memory_criteria
                .last()
                .map(String::as_str),
            Some("st criteria")
        );
        assert_eq!(
            criteria
                .long_term_memory_operator
                .last()
                .map(String::as_str),
            Some("lt operator")
        );
        assert_eq!(
            criteria
                .short_term_memory_operator
                .last()
                .map(String::as_str),
            Some("st operator")
        );
        assert_eq!(
            criteria.long_term_memory_trigger.last().map(String::as_str),
            Some("lt trigger")
        );
        assert_eq!(
            criteria
                .short_term_memory_trigger
                .last()
                .map(String::as_str),
            Some("st trigger")
        );
    }

    #[test]
    fn criteria_append_to_system_prompt_renders_instruction_sections() {
        let criteria = AgentMemorySystemPromptCriteria::builder()
            .append_long_term_memory_criteria("extra long-term rule".to_string())
            .append_short_term_memory_trigger("extra short-term trigger".to_string())
            .build();

        let rendered = criteria.append_to_system_prompt("base system prompt");

        assert!(
            rendered
                .starts_with("base system prompt\n\nYou are equipped with a shared memory system.")
        );
        assert!(rendered.contains("Long-term memory criteria:"));
        assert!(rendered.contains("Short-term memory triggers:"));
        assert!(rendered.contains("- extra long-term rule"));
        assert!(rendered.contains("- extra short-term trigger"));
    }

    #[test]
    fn criteria_render_system_prompt_includes_long_and_short_term_memory() {
        let criteria = AgentMemorySystemPromptCriteria::default();
        let mut memory = AgentMemory::default();

        memory.long_term.insert(
            "architecture".to_string(),
            AgentMemoryContent {
                title: "architecture".to_string(),
                related_context: "backend".to_string(),
                trigger_scenario: "planning".to_string(),
                content: "service graph".to_string(),
                raw_content: None,
            },
        );
        memory.short_term.insert(
            "current task".to_string(),
            AgentMemoryContent {
                title: "current task".to_string(),
                related_context: "chat loop".to_string(),
                trigger_scenario: "resume after compact".to_string(),
                content: "do not render this".to_string(),
                raw_content: Some("do not render this either".to_string()),
            },
        );

        let rendered = criteria.render_system_prompt("base system prompt", &memory);

        assert!(
            rendered
                .starts_with("base system prompt\n\nYou are equipped with a shared memory system.")
        );
        assert!(rendered.contains("Long-term memory entries:\n\n"));
        assert!(rendered.contains("title: architecture"));
        assert!(rendered.contains("content:\nservice graph"));
        assert!(rendered.contains("Short-term memory entries:\n\n"));
        assert!(rendered.contains("title: current task"));
        assert!(rendered.contains("trigger_scenario: resume after compact"));
        assert!(!rendered.contains("do not render this either"));
    }

    #[test]
    fn render_long_term_memory_section_renders_full_entries() {
        let memories = vec![
            AgentMemoryContent {
                title: "architecture".to_string(),
                related_context: "backend".to_string(),
                trigger_scenario: "planning".to_string(),
                content: "service graph".to_string(),
                raw_content: None,
            },
            AgentMemoryContent {
                title: "workflow".to_string(),
                related_context: "dev loop".to_string(),
                trigger_scenario: "implementation".to_string(),
                content: "run cargo check".to_string(),
                raw_content: None,
            },
        ];

        let rendered = render_long_term_memory_section(memories.iter()).unwrap();

        assert!(rendered.starts_with("Long-term memory entries:\n\n"));
        assert!(rendered.contains("title: architecture"));
        assert!(rendered.contains("related_context: backend"));
        assert!(rendered.contains("content:\nservice graph"));
        assert!(rendered.contains("title: workflow"));
        assert!(rendered.contains("content:\nrun cargo check"));
        assert!(rendered.contains("\n\n---\n\n"));
    }

    #[test]
    fn render_short_term_memory_section_omits_content_and_raw_content() {
        let memories = vec![AgentMemoryContent {
            title: "current task".to_string(),
            related_context: "chat loop".to_string(),
            trigger_scenario: "resume after compact".to_string(),
            content: "do not render this".to_string(),
            raw_content: Some("do not render this either".to_string()),
        }];

        let rendered = render_short_term_memory_section(memories.iter()).unwrap();

        assert!(rendered.starts_with("Short-term memory entries:\n\n"));
        assert!(rendered.contains("title: current task"));
        assert!(rendered.contains("related_context: chat loop"));
        assert!(rendered.contains("trigger_scenario: resume after compact"));
        assert!(!rendered.contains("content:\n"));
        assert!(!rendered.contains("do not render this either"));
    }
}
