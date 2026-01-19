//! # Basic Agent Example
//!
//! This example demonstrates how to create a simple agent with OpenAI and run it.
//!
//! ## Usage
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key"
//! cargo run --example basic_agent
//! ```

use simple_agent::prelude::*;
use simple_agent::{MessageRole, MessageContent};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Get API key from environment
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("Please set OPENAI_API_KEY environment variable");
    let base_url = std::env::var("OPENAI_API_BASE_URL").expect("Please set OPENAI_API_BASE_URL environment variable").into();

    println!("base url: {:?}", base_url);
    println!("Simple Agent Example");

    // Create OpenAI client
    let mut builder = LLMClientBuilder::new()
        .with_api_key(api_key);
    if let Some(url) = base_url {
        builder = builder.with_base_url(url);
    }
    let llm_client = builder.build_openai()?;

    // Create tool registry
    let registry = Arc::new(Mutex::new(ToolRegistry::new()));

    // Create session with system prompt
    let session = Session::new(
        ModelConfig {
            name: "MiniMax-M2.1".to_string(),
            max_tokens: 1024,
            temperature: Some(0.7),
            extra: None,
        },
        "You are a helpful assistant. Be concise and friendly."
    );

    // Create agent configuration
    let config = AgentConfig {
        model: "MiniMax-M2.1".to_string(),
        system_prompt: "You are a helpful assistant. Be concise and friendly.".to_string(),
        max_steps: 10,
        max_tokens: 1024,
        temperature: Some(0.7),
    };

    // Create agent
    let agent = Agent::new(session, llm_client, registry, config);

    // Run the agent
    let messages = agent.run("Hello! Can you help me with some questions?").await?;

    // Print responses
    println!("\n=== Agent Response ===");
    for message in messages {
        match message.role {
            MessageRole::User => {
                println!("\nUser: {}", content_to_text(&message.content));
            }
            MessageRole::Assistant => {
                println!("\nAssistant: {}", content_to_text(&message.content));
            }
            MessageRole::Tool => {
                println!("\nTool Result: {}", content_to_text(&message.content));
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
