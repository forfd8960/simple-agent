use std::sync::Arc;
use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex;

use crate::tool::{Tool, ToolDefinition, ToolResult, ToolError};
use crate::mcp::client::MCPClient;

/// Adapter that wraps an MCP client tool as a local Tool.
#[derive(Debug, Clone)]
pub struct MCPToolAdapter {
    client: Arc<Mutex<MCPClient>>,
    definition: ToolDefinition,
}

impl MCPToolAdapter {
    /// Creates a new MCP tool adapter.
    pub fn new(client: Arc<Mutex<MCPClient>>, definition: ToolDefinition) -> Self {
        Self {
            client,
            definition,
        }
    }
}

#[async_trait]
impl Tool for MCPToolAdapter {
    fn name(&self) -> &str {
        &self.definition.name
    }

    fn description(&self) -> &str {
        &self.definition.description
    }

    fn parameters_schema(&self) -> Value {
        self.definition.input_schema.clone()
    }

    async fn execute(&self, args: Value) -> Result<ToolResult, ToolError> {
        let mut client = self.client.lock().await;
        let output = client
            .call_tool(&self.definition.name, args)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        Ok(ToolResult {
            output,
            metadata: None,
            error: None,
        })
    }
}

/// Converts a list of MCP tool definitions to local tools.
pub fn adapt_mcp_tools(
    client: Arc<Mutex<MCPClient>>,
    tools: Vec<ToolDefinition>,
) -> Vec<Arc<dyn Tool>> {
    tools
        .into_iter()
        .map(|def| {
            Arc::new(MCPToolAdapter::new(client.clone(), def)) as Arc<dyn Tool>
        })
        .collect()
}
