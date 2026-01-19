//! # Simple Agent SDK
//!
//! A lightweight, type-safe Rust SDK for building multi-turn agents with tool calling support.
//!
//! ## Features
//!
//! - **Core Agent**: Multi-turn agent loop with streaming support
//! - **Tool System**: Easy-to-use trait for custom tools
//! - **OpenAI Integration**: Built-in support for OpenAI's API
//! - **MCP Support**: Connect to Model Context Protocol servers
//! - **Permission System**: Configure tool execution permissions
//!
//! ## Quick Start
//!
//! ```rust
//! use simple_agent::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let api_key = std::env::var("OPENAI_API_KEY")?;
//!
//!     // Create OpenAI client
//!     let llm_client = LLMClientBuilder::new()
//!         .with_api_key(api_key)
//!         .build_openai()?;
//!
//!     // Create tool registry
//!     let registry = Arc::new(Mutex::new(ToolRegistry::new()));
//!
//!     // Create session
//!     let session = Session::new(
//!         ModelConfig::default(),
//!         "You are a helpful assistant."
//!     );
//!
//!     // Create and run agent
//!     let agent = Agent::with_defaults(session, llm_client, registry);
//!     let messages = agent.run("Hello!").await?;
//!
//!     for message in messages {
//!         println!("{:?}", message);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!

pub mod agent;
pub mod llm;
pub mod session;
pub mod tool;
pub mod mcp;
pub mod permission;

// Re-exports for convenient usage
pub use agent::{Agent, AgentConfig, AgentEvent};
pub use llm::{LLMClient, LLMInput, LLMOutput, LLMEvent, OpenAIClient};
pub use llm::client::LLMClientBuilder;
pub use session::{Session, Message, MessageContent, MessageRole, ModelConfig};
pub use tool::{Tool, ToolRegistry, ToolExecutor, ToolDefinition, ToolResult, ToolError, DynTool};
pub use mcp::{MCPClient, MCPClientBuilder, MCPConfig, MCPTransport, MCToolInfo};
pub use permission::{PermissionManager, Permission, PermissionAction, PermissionResult};

/// Prelude module with commonly used types.
pub mod prelude {
    pub use crate::agent::{Agent, AgentConfig};
    pub use crate::llm::{LLMClient, OpenAIClient};
    pub use crate::session::{Session, Message, ModelConfig};
    pub use crate::tool::{Tool, ToolRegistry, ToolResult, ToolError, DynTool};
    pub use crate::LLMClientBuilder;
}
