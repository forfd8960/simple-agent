pub mod client;
pub mod adapter;

pub use client::{MCPClient, MCPConfig, MCPTransport, MCPError, MCPClientBuilder, MCToolInfo, ToolsListResponse};
pub use adapter::{MCPToolAdapter, adapt_mcp_tools};
