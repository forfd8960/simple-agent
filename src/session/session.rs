use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;

/// Represents a conversation session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique identifier for the session
    pub id: String,
    /// The messages in the conversation
    pub messages: Vec<super::Message>,
    /// The system prompt for the agent
    pub system_prompt: String,
    /// The model configuration
    pub model: ModelConfig,
    /// The current status of the session
    pub status: SessionStatus,
}

/// The status of a session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionStatus {
    /// Idle, waiting for input
    Idle,
    /// Currently running
    Running,
    /// Completed
    Completed,
    /// Error occurred
    Error,
}

/// Configuration for the LLM model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// The model name (e.g., "gpt-4", "claude-sonnet-4-20250514")
    pub name: String,
    /// Maximum number of tokens to generate
    pub max_tokens: u32,
    /// Temperature for sampling (0.0 to 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Additional model-specific parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<HashMap<String, serde_json::Value>>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "gpt-4o".to_string(),
            max_tokens: 4096,
            temperature: None,
            extra: None,
        }
    }
}

impl Session {
    /// Creates a new session with the given model configuration and system prompt.
    pub fn new(model: ModelConfig, system_prompt: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            messages: Vec::new(),
            system_prompt: system_prompt.into(),
            model,
            status: SessionStatus::Idle,
        }
    }

    /// Creates a new session with default model configuration.
    pub fn with_default_model(system_prompt: impl Into<String>) -> Self {
        Self::new(ModelConfig::default(), system_prompt)
    }

    /// Adds a message to the session.
    pub fn add_message(&mut self, message: super::Message) {
        self.messages.push(message);
    }

    /// Returns the number of messages in the session.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Clears all messages from the session.
    pub fn clear_messages(&mut self) {
        self.messages.clear();
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::with_default_model("")
    }
}
