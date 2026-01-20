//! # MCP Client Example
//!
//! This example demonstrates how to connect to an MCP (Model Context Protocol) server
//! and use its tools with the agent.
//!
//! ## Prerequisites
//!
//! You need an MCP server running. For testing, you can use the example MCP server
//! that comes with the MCP SDK, or any MCP-compatible server.
//!
//! ## Usage
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key"
//!
//! # Option 1: Run with a command-based MCP server (filesystem example)
//! cargo run --example mcp_client -- --command "npx" --args "@modelcontextprotocol/server-filesystem" "/path/to/dir"
//!
//! # Option 2: Connect to an HTTP MCP server
//! cargo run --example mcp_client -- --url "http://localhost:3000/mcp"
//! ```

use async_trait::async_trait;
use clap::Parser;
use serde_json::Value;
use simple_agent::mcp::MCPClient;
use simple_agent::prelude::*;
use simple_agent::{MCToolInfo, MessageContent, MessageRole, ToolError, ToolResult};
use std::sync::Arc;
use tokio::sync::Mutex;

/// MCP Client Example
#[derive(Parser, Debug)]
#[command(name = "mcp-example")]
struct Args {
    /// MCP server name
    #[arg(long, default_value = "filesystem")]
    name: String,

    /// Connect via command (stdio transport)
    #[arg(long, conflicts_with = "url")]
    command: Option<String>,

    /// Arguments for the command
    #[arg(long, requires = "command")]
    args: Option<Vec<String>>,

    /// Connect via HTTP URL
    #[arg(long, conflicts_with_all = ["command", "args"])]
    url: Option<String>,
}

/// A wrapper tool that calls MCP tools
#[derive(Debug, Clone)]
struct MCPWrappedTool {
    client: Arc<Mutex<MCPClient>>,
    name: String,
    description: String,
    schema: Value,
}

impl MCPWrappedTool {
    fn new(client: Arc<Mutex<MCPClient>>, tool_info: MCToolInfo) -> Self {
        Self {
            client,
            name: tool_info.name,
            description: tool_info.description,
            schema: tool_info.input_schema,
        }
    }
}

#[async_trait]
impl Tool for MCPWrappedTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> Value {
        self.schema.clone()
    }

    async fn execute(&self, args: Value) -> Result<ToolResult, ToolError> {
        let mut client = self.client.lock().await;

        let result = client
            .call_tool(&self.name, args)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        Ok(ToolResult::ok(result))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    // Parse command line arguments
    let args = Args::parse();

    // Get API key
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("Please set OPENAI_API_KEY environment variable");
    let base_url: String = std::env::var("OPENAI_API_BASE_URL")
        .expect("Please set OPENAI_API_BASE_URL environment variable")
        .into();

    // Create OpenAI client
    let llm_client = LLMClientBuilder::new()
        .with_api_key(api_key)
        .with_base_url(base_url)
        .build_openai()?;

    // Create MCP client - build first, then wrap in Arc
    let mcp_client = MCPClient::builder()
        .with_name(&args.name)
        .with_sse_transport("http://127.0.0.1:8000/sse")
        // .with_stdio_transport(
        //     args.command.unwrap_or_else(|| "npx".to_string()),
        //     args.args.unwrap_or_else(|| {
        //         vec![
        //             "@modelcontextprotocol/server-filesystem".to_string(),
        //             ".".to_string(),
        //         ]
        //     }),
        // )
        .build()?;

    // Wrap in Arc<Mutex> before connecting since MCPClient has mutable methods
    let mcp_client_arc: Arc<Mutex<MCPClient>> = Arc::new(mcp_client.into());

    println!("Connecting to MCP server...");
    mcp_client_arc.lock().await.connect().await?;
    println!("Connected to MCP server!");

    // List available tools
    let tools: Vec<MCToolInfo> = mcp_client_arc.lock().await.list_tools().await?;
    println!("\nAvailable MCP tools:");
    for tool in &tools {
        println!("  - {}: {}", tool.name, tool.description);
    }

    // Create tool registry
    let mut registry = ToolRegistry::new();

    // Wrap tools
    let wrapped_tools: Vec<Arc<dyn Tool>> = tools
        .into_iter()
        .map(|tool_info| {
            let client = mcp_client_arc.clone();
            Arc::new(MCPWrappedTool::new(client, tool_info)) as Arc<dyn Tool>
        })
        .collect();

    for tool in &wrapped_tools {
        registry.register(tool.clone() as DynTool);
    }

    println!(
        "\nRegistered {} MCP tools in agent registry",
        registry.len()
    );

    // Create session
    let session = Session::new(
        ModelConfig {
            name: "MiniMax-M2.1".to_string(),
            max_tokens: 4096,
            temperature: Some(0.7),
            extra: None,
        },
        "You are a helpful assistant with access to postgres mcp tools. \
         You can send sql query to the database and get results back.",
    );

    // Create agent
    let registry = Arc::new(Mutex::new(registry));
    let mut agent_config = AgentConfig::default();
    agent_config.model = "MiniMax-M2.1".to_string();

    let agent = Agent::new(session, llm_client, registry, agent_config);

    // Example queries that use MCP tools
    let queries = vec![
        "有多少用户？",
        "列出所有产品名称和价格",
    ];

    for query in queries {
        println!("\n{}", "=".repeat(60));
        println!("Query: {}", query);
        println!("{}", "=".repeat(60));

        match agent.run(query).await {
            Ok(messages) => {
                for message in messages {
                    match message.role {
                        MessageRole::User => {
                            println!("\nUser: {}", content_to_text(&message.content));
                        }
                        MessageRole::Assistant => {
                            println!("\nAssistant: {}", content_to_text(&message.content));
                        }
                        MessageRole::Tool => {
                            println!("\n[Tool Result]");
                        }
                    }
                }
            }
            Err(e) => {
                println!("\nError: {}", e);
            }
        }
    }

    // Cleanup
    mcp_client_arc.lock().await.disconnect().await?;

    Ok(())
}

fn content_to_text(content: &[MessageContent]) -> String {
    content
        .iter()
        .filter_map(|c| {
            if let MessageContent::Text { text } = c {
                Some(text.clone())
            } else {
                None
            }
        })
        .collect()
}
