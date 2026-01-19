use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Represents a message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Unique identifier for the message
    pub id: String,
    /// The role of the message sender
    pub role: MessageRole,
    /// The content of the message
    pub content: Vec<MessageContent>,
    /// Timestamp when the message was created
    pub created_at: DateTime<Utc>,
}

/// The role of the message sender.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// User message
    User,
    /// Assistant message (from the LLM)
    Assistant,
    /// Tool result message
    Tool,
}

/// The content of a message, which can be text or a tool call/result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContent {
    /// Plain text content
    Text {
        /// The text content
        text: String,
    },
    /// A tool call request
    ToolCall {
        /// Unique identifier for the tool call
        id: String,
        /// The name of the tool to call
        name: String,
        /// The arguments to pass to the tool
        arguments: serde_json::Value,
    },
    /// The result of a tool execution
    ToolResult {
        /// The ID of the tool call this result is for
        tool_call_id: String,
        /// The result returned by the tool
        result: String,
        /// Whether the tool execution resulted in an error
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

impl Message {
    /// Creates a new user message.
    pub fn new_user(text: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role: MessageRole::User,
            content: vec![MessageContent::Text {
                text: text.into(),
            }],
            created_at: Utc::now(),
        }
    }

    /// Creates a new assistant message.
    pub fn new_assistant(content: Vec<MessageContent>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role: MessageRole::Assistant,
            content,
            created_at: Utc::now(),
        }
    }

    /// Creates a new tool result message.
    pub fn new_tool_result(results: Vec<MessageContent>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role: MessageRole::Tool,
            content: results,
            created_at: Utc::now(),
        }
    }
}
