//! Error types for the simple-agent library.

use thiserror::Error;

/// Unified error type for the agent SDK.
#[derive(Debug, Error)]
pub enum AgentError {
    /// LLM-related error
    #[error("LLM error: {0}")]
    LLM(#[from] crate::llm::LLMError),

    /// Tool-related error
    #[error("Tool error: {0}")]
    Tool(#[from] crate::tool::ToolError),

    /// MCP-related error
    #[error("MCP error: {0}")]
    MCP(#[from] crate::mcp::MCPError),

    /// Session-related error
    #[error("Session error: {0}")]
    Session(String),

    /// Permission denied
    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    /// Maximum steps exceeded
    #[error("Max steps exceeded")]
    MaxStepsExceeded,

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
