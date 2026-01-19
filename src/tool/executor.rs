use std::sync::Arc;
use tokio::sync::Mutex;
use crate::tool::{ToolRegistry, ToolDefinition};
use crate::session::MessageContent;

/// Context for tool execution.
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// The session ID
    pub session_id: String,
    /// The message ID
    pub message_id: String,
}

/// The result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// The output from the tool
    pub output: String,
    /// Optional metadata from the tool execution
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
    /// Optional error message if the tool execution failed
    pub error: Option<String>,
}

/// Executes tool calls from the agent.
#[derive(Debug, Clone)]
pub struct ToolExecutor {
    registry: Arc<Mutex<ToolRegistry>>,
}

impl ToolExecutor {
    /// Creates a new tool executor with the given registry.
    pub fn new(registry: Arc<Mutex<ToolRegistry>>) -> Self {
        Self { registry }
    }

    /// Returns all tool definitions for passing to the LLM.
    pub async fn get_tool_definitions(&self) -> Vec<ToolDefinition> {
        let registry = self.registry.lock().await;
        registry.to_tool_definitions()
    }

    /// Executes a single tool call.
    pub async fn execute(
        &self,
        call: &MessageContent,
        _ctx: ExecutionContext,
    ) -> MessageContent {
        let (id, name, arguments) = match call {
            MessageContent::ToolCall {
                id,
                name,
                arguments,
            } => (id.clone(), name.clone(), arguments.clone()),
            _ => {
                return MessageContent::ToolResult {
                    tool_call_id: String::new(),
                    result: "Invalid tool call content".to_string(),
                    is_error: Some(true),
                }
            }
        };

        let registry = self.registry.lock().await;
        let tool = match registry.get(&name) {
            Some(tool) => tool.clone(),
            None => {
                return MessageContent::ToolResult {
                    tool_call_id: id,
                    result: format!("Tool not found: {}", name),
                    is_error: Some(true),
                }
            }
        };
        drop(registry);

        match tool.execute(arguments).await {
            Ok(result) => MessageContent::ToolResult {
                tool_call_id: id,
                result: result.output,
                is_error: result.error.as_ref().map(|_| true),
            },
            Err(error) => MessageContent::ToolResult {
                tool_call_id: id,
                result: error.to_string(),
                is_error: Some(true),
            },
        }
    }

    /// Executes multiple tool calls in parallel.
    pub async fn execute_all(
        &self,
        calls: Vec<MessageContent>,
        ctx: ExecutionContext,
    ) -> Vec<MessageContent> {
        let mut results = Vec::new();

        for call in calls {
            let result = self.execute(&call, ctx.clone()).await;
            results.push(result);
        }

        results
    }
}
