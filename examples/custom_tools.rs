//! # Custom Tools Example
//!
//! This example demonstrates how to create custom tools and register them with the agent.
//!
//! ## Usage
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key"
//! cargo run --example custom_tools
//! ```

use simple_agent::prelude::*;
use simple_agent::{MessageRole, MessageContent, ToolError, ToolResult};
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

/// A custom tool that gets the current weather.
#[derive(Debug)]
struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> &str {
        "get_weather"
    }

    fn description(&self) -> &str {
        "Get the current weather for a location"
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        })
    }

    async fn execute(&self, args: Value) -> Result<ToolResult, ToolError> {
        let location = args["location"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("location is required".to_string()))?;

        let unit = args["unit"].as_str().unwrap_or("celsius");

        // Simulate weather API call
        let temperature = match unit {
            "fahrenheit" => 72.0,
            _ => 22.0,
        };

        let condition = match unit {
            "fahrenheit" => "sunny and warm",
            _ => "晴天",
        };

        Ok(ToolResult::ok(format!(
            "The weather in {} is {}°{} - {}",
            location, temperature, unit, condition
        )))
    }
}

/// A custom tool that calculates something.
#[derive(Debug)]
struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculate"
    }

    fn description(&self) -> &str {
        "Perform basic calculations"
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate"
                }
            },
            "required": ["expression"]
        })
    }

    async fn execute(&self, args: Value) -> Result<ToolResult, ToolError> {
        let expression = args["expression"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("expression is required".to_string()))?;

        // Simple calculator using eval (for demo purposes)
        // In production, use a proper math parser!
        let result = match expression {
            e if e.contains('+') => {
                let parts: Vec<&str> = e.split('+').collect();
                if parts.len() == 2 {
                    let a: f64 = parts[0].trim().parse().unwrap_or(0.0);
                    let b: f64 = parts[1].trim().parse().unwrap_or(0.0);
                    a + b
                } else {
                    return Err(ToolError::InvalidArguments(
                        "Invalid expression format".to_string(),
                    ));
                }
            }
            e if e.contains('-') => {
                let parts: Vec<&str> = e.split('-').collect();
                if parts.len() == 2 {
                    let a: f64 = parts[0].trim().parse().unwrap_or(0.0);
                    let b: f64 = parts[1].trim().parse().unwrap_or(0.0);
                    a - b
                } else {
                    return Err(ToolError::InvalidArguments(
                        "Invalid expression format".to_string(),
                    ));
                }
            }
            e if e.contains('*') => {
                let parts: Vec<&str> = e.split('*').collect();
                if parts.len() == 2 {
                    let a: f64 = parts[0].trim().parse().unwrap_or(0.0);
                    let b: f64 = parts[1].trim().parse().unwrap_or(0.0);
                    a * b
                } else {
                    return Err(ToolError::InvalidArguments(
                        "Invalid expression format".to_string(),
                    ));
                }
            }
            e if e.contains('/') => {
                let parts: Vec<&str> = e.split('/').collect();
                if parts.len() == 2 {
                    let a: f64 = parts[0].trim().parse().unwrap_or(0.0);
                    let b: f64 = parts[1].trim().parse().unwrap_or(1.0);
                    if b == 0.0 {
                        return Err(ToolError::ExecutionFailed(
                            "Division by zero".to_string(),
                        ));
                    }
                    a / b
                } else {
                    return Err(ToolError::InvalidArguments(
                        "Invalid expression format".to_string(),
                    ));
                }
            }
            _ => {
                return Err(ToolError::InvalidArguments(
                    "Unknown operator. Supported: +, -, *, /".to_string(),
                ));
            }
        };

        Ok(ToolResult::ok(format!("{} = {}", expression, result)))
    }
}

/// A tool that searches for information.
#[derive(Debug)]
struct SearchTool;

#[async_trait]
impl Tool for SearchTool {
    fn name(&self) -> &str {
        "search"
    }

    fn description(&self) -> &str {
        "Search for information on a topic"
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: Value) -> Result<ToolResult, ToolError> {
        let query = args["query"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("query is required".to_string()))?;

        // Simulated search results
        let results = format!(
            "Search results for '{}':\n\
             1. Result A - First matching item\n\
             2. Result B - Second matching item\n\
             3. Result C - Third matching item",
            query
        );

        Ok(ToolResult::ok(results))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    // Get API key
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("Please set OPENAI_API_KEY environment variable");
    let base_url: String = std::env::var("OPENAI_API_BASE_URL").expect("Please set OPENAI_API_BASE_URL environment variable").into();

    // Create OpenAI client
    let llm_client = LLMClientBuilder::new()
        .with_api_key(api_key)
        .with_base_url(base_url)
        .build_openai()?;

    // Create and populate tool registry
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(WeatherTool));
    registry.register(Arc::new(CalculatorTool));
    registry.register(Arc::new(SearchTool));

    println!("Registered tools: {:?}", registry.list().len());

    // Wrap in Arc<Mutex>
    let registry = Arc::new(Mutex::new(registry));

    // Create session
    let session = Session::new(
        ModelConfig {
            name: "MiniMax-M2.1".to_string(),
            max_tokens: 1024,
            temperature: Some(0.7),
            extra: None,
        },
        "You are a helpful assistant with access to tools. \
         Use the tools when appropriate to provide accurate information."
    );

    // Create agent
    let agent = Agent::with_defaults(session, llm_client, registry);

    // Run with different queries
    let queries = vec![
        "What's the weather in Chengdu?",
        "Calculate 15 + 27",
        "What is borrow checking in Rust Programming",
    ];

    for query in queries {
        println!("\n{}", "=".repeat(50));
        println!("Query: {}", query);
        println!("{}", "=".repeat(50));

        let messages = agent.run(query).await?;

        for message in messages {
            match message.role {
                MessageRole::User => {
                    println!("\nUser: {}", content_to_text(&message.content));
                }
                MessageRole::Assistant => {
                    println!("\nAssistant: {}", content_to_text(&message.content));
                }
                MessageRole::Tool => {
                    println!("\n[Tool executed]");
                }
            }
        }
    }

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
