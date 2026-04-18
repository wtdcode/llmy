#[derive(Debug, Clone)]
pub enum StepResult {
    Stop(String),
    Toolcalled(Option<String>),
}

impl StepResult {
    pub fn assistant_message(&self) -> Option<&String> {
        match self {
            Self::Stop(v) => Some(v),
            Self::Toolcalled(v) => v.as_ref(),
        }
    }

    pub fn did_tool_call(&self) -> bool {
        matches!(self, Self::Toolcalled(_))
    }

    pub fn did_stop(&self) -> bool {
        matches!(self, Self::Stop(_))
    }
}
