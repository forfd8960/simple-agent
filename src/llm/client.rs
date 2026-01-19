use async_trait::async_trait;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use crate::session::Message;
use crate::tool::ToolDefinition;
use super::openai::OpenAIClient;

/// Input for an LLM request.
#[derive(Debug, Clone)]
pub struct LLMInput {
    /// The model to use
    pub model: String,
    /// The messages to send
    pub messages: Vec<Message>,
    /// The system prompt
    pub system_prompt: String,
    /// Available tools for the LLM
    pub tools: Vec<ToolDefinition>,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Optional temperature (0.0 to 1.0)
    pub temperature: Option<f32>,
}

/// Output from an LLM response.
#[derive(Debug, Clone)]
pub struct LLMOutput {
    /// The content of the response
    pub content: Vec<super::super::session::MessageContent>,
    /// The reason the response finished
    pub finish_reason: FinishReason,
    /// Token usage statistics
    pub usage: Usage,
}

/// The reason the LLM finished generating.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural stop point reached
    Stop,
    /// Stopped due to tool calls
    ToolCalls,
    /// Maximum tokens reached
    MaxTokens,
    /// Stopped due to an error
    Error,
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Number of input tokens
    pub input_tokens: u32,
    /// Number of output tokens
    pub output_tokens: u32,
}

/// Events from a streaming LLM response.
#[derive(Debug, Clone)]
pub enum LLMEvent {
    /// A text chunk was received
    TextDelta {
        text: String,
    },
    /// A tool call has started
    ToolCallStart {
        id: String,
        name: String,
    },
    /// A tool call argument chunk was received
    ToolCallDelta {
        id: String,
        arguments: String,
    },
    /// A tool call has been completed
    ToolCallEnd {
        id: String,
    },
    /// The response has finished
    Finish {
        reason: FinishReason,
        usage: Usage,
    },
    /// An error occurred
    Error {
        error: String,
    },
}

/// A stream of LLM events.
pub type LLMStream = Pin<Box<dyn Stream<Item = Result<LLMEvent, LLMError>> + Send>>;

/// Errors that can occur when communicating with an LLM.
#[derive(Debug, thiserror::Error)]
pub enum LLMError {
    /// An API error occurred
    #[error("API error: {0}")]
    ApiError(String),
    /// A network error occurred
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),
    /// The response from the LLM was invalid
    #[error("Invalid response: {0}")]
    InvalidResponse(String),
    /// Authentication failed
    #[error("Authentication failed: {0}")]
    AuthError(String),
    /// Rate limit exceeded
    #[error("Rate limit exceeded: {0}")]
    RateLimitError(String),
}

/// Trait for LLM clients.
#[async_trait]
pub trait LLMClient: Send + Sync {
    /// Sends a request and returns a streaming response.
    async fn stream(&self, input: LLMInput) -> Result<LLMStream, LLMError>;
    /// Sends a request and returns a complete response.
    async fn complete(&self, input: LLMInput) -> Result<LLMOutput, LLMError>;
}

/// A builder for creating LLM clients.
#[derive(Debug, Default)]
pub struct LLMClientBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    timeout: Option<std::time::Duration>,
}

impl LLMClientBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the API key.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Sets the base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Sets the timeout.
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Creates an OpenAI client.
    pub fn build_openai(self) -> Result<Arc<dyn LLMClient>, LLMError> {
        Ok(Arc::new(OpenAIClient::new(
            self.api_key
                .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                .ok_or(LLMError::AuthError("OpenAI API key not provided".to_string()))?,
            self.base_url,
            self.timeout,
        )))
    }
}
