pub mod registry;
pub mod executor;

pub use registry::ToolRegistry;
pub use executor::{ToolExecutor, ExecutionContext};
pub use tool_types::{ToolDefinition, ToolResult, ToolError};
pub use tool_trait::Tool;
pub use tool_trait::DynTool;

mod tool_types {
    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    /// Definition of a tool that can be called by the agent.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ToolDefinition {
        /// The name of the tool
        pub name: String,
        /// A description of what the tool does
        pub description: String,
        /// JSON Schema for the tool's input parameters
        pub input_schema: Value,
    }

    /// The result of executing a tool.
    #[derive(Debug, Clone)]
    pub struct ToolResult {
        /// The output from the tool
        pub output: String,
        /// Optional metadata from the tool execution
        #[allow(dead_code)]
        pub metadata: Option<serde_json::Map<String, Value>>,
        /// Optional error message if the tool execution failed
        #[allow(dead_code)]
        pub error: Option<String>,
    }

    impl ToolResult {
        /// Creates a successful result.
        pub fn ok(output: impl Into<String>) -> Self {
            Self {
                output: output.into(),
                metadata: None,
                error: None,
            }
        }

        /// Creates a result with an error.
        pub fn error(error: impl Into<String>) -> Self {
            Self {
                output: String::new(),
                metadata: None,
                error: Some(error.into()),
            }
        }
    }

    /// Errors that can occur when executing a tool.
    #[derive(Debug, thiserror::Error)]
    pub enum ToolError {
        #[error("Invalid arguments: {0}")]
        InvalidArguments(String),
        #[error("Execution failed: {0}")]
        ExecutionFailed(String),
        #[error("Tool not found: {0}")]
        NotFound(String),
    }
}

mod tool_trait {
    use super::tool_types::{ToolDefinition, ToolResult, ToolError};
    use async_trait::async_trait;
    use serde_json::Value;
    use std::sync::Arc;

    /// Trait representing a tool that can be called by the agent.
    #[async_trait]
    pub trait Tool: Send + Sync {
        /// Returns the name of the tool.
        fn name(&self) -> &str;
        /// Returns a description of what the tool does.
        fn description(&self) -> &str;
        /// Returns the JSON Schema for the tool's input parameters.
        fn parameters_schema(&self) -> Value;

        /// Executes the tool with the given arguments.
        async fn execute(&self, args: Value) -> Result<ToolResult, ToolError>;

        /// Converts the tool to its definition.
        fn to_definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.name().to_string(),
                description: self.description().to_string(),
                input_schema: self.parameters_schema(),
            }
        }
    }

    /// A type alias for a dynamic tool reference.
    pub type DynTool = Arc<dyn Tool>;
}
