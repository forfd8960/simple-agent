use async_trait::async_trait;
use async_stream::stream;
use futures::stream::StreamExt;
use reqwest::{Client, RequestBuilder};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use tracing::debug;

use super::{LLMClient, LLMInput, LLMOutput, LLMStream, LLMEvent, FinishReason, Usage, LLMError};
use crate::session::{MessageContent, MessageRole};

/// OpenAI API response for chat completions.
#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    #[serde(default)]
    id: String,
    #[serde(default)]
    object: String,
    #[serde(default)]
    created: u64,
    #[serde(default)]
    model: String,
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    usage: UsageInfo,
}

#[derive(Debug, Deserialize)]
struct Choice {
    #[serde(default)]
    index: u32,
    message: MessageResponse,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageResponse {
    #[serde(default)]
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize)]
struct ToolCall {
    #[serde(default)]
    id: String,
    #[serde(rename = "type", default)]
    call_type: String,
    function: FunctionCall,
}

#[derive(Debug, Deserialize)]
struct FunctionCall {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Debug, Default, Deserialize)]
struct UsageInfo {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
}

/// Streaming response chunk.
#[derive(Debug, Deserialize)]
struct ChatCompletionChunk {
    #[serde(default)]
    id: String,
    #[serde(default)]
    object: String,
    #[serde(default)]
    created: u64,
    #[serde(default)]
    model: String,
    choices: Vec<ChunkChoice>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    index: u32,
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Delta {
    #[serde(default)]
    role: Option<String>,
    content: Option<String>,
    tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Debug, Deserialize)]
struct ChunkToolCall {
    #[serde(default)]
    id: String,
    #[serde(rename = "type", default)]
    call_type: String,
    function: ChunkFunctionCall,
}

#[derive(Debug, Deserialize)]
struct ChunkFunctionCall {
    name: Option<String>,
    arguments: Option<String>,
}

/// An LLM client for OpenAI's API.
#[derive(Debug, Clone)]
pub struct OpenAIClient {
    client: Client,
    base_url: String,
}

impl OpenAIClient {
    /// Creates a new OpenAI client.
    pub fn new(
        api_key: String,
        base_url: Option<String>,
        timeout: Option<Duration>,
    ) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!("Bearer {}", api_key))
                .expect("Failed to create authorization header"),
        );
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );

        let mut client_builder = reqwest::Client::builder()
            .default_headers(headers)
            .http1_title_case_headers();

        if let Some(timeout) = timeout {
            client_builder = client_builder.timeout(timeout);
        }

        let client = client_builder.build().expect("Failed to build HTTP client");

        Self {
            client,
            base_url: base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
        }
    }

    /// Creates a request builder for chat completions.
    fn chat_completions_request(&self, input: &LLMInput) -> RequestBuilder {
        // Note: MiniMax API does not support the OpenAI tool format
        // Tools will be skipped for now - this can be extended for APIs that support tools
        let tools: Vec<Value> = Vec::new();

        let body = ChatRequest {
            model: input.model.clone(),
            messages: Self::build_messages(input),
            tools: if tools.is_empty() { None } else { Some(tools) },
            max_tokens: Some(input.max_tokens),
            temperature: input.temperature,
            stream: false,
        };

        debug!(model = %input.model, "Sending request to OpenAI");

        self.client
            .post(&format!("{}/chat/completions", self.base_url))
            .json(&body)
    }

    /// Builds messages for the API request.
    fn build_messages(input: &LLMInput) -> Vec<Value> {
        let mut messages = Vec::new();

        // Add system prompt
        if !input.system_prompt.is_empty() {
            messages.push(serde_json::json!({
                "role": "system",
                "content": input.system_prompt
            }));
        }

        // Add conversation messages
        for msg in &input.messages {
            match msg.role {
                MessageRole::User => {
                    messages.push(serde_json::json!({
                        "role": "user",
                        "content": Self::content_to_string(&msg.content)
                    }));
                }
                MessageRole::Assistant => {
                    let tool_calls = msg.content.iter().filter_map(|c| {
                        if let MessageContent::ToolCall { id, name, arguments } = c {
                            Some(serde_json::json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments.to_string()
                                }
                            }))
                        } else {
                            None
                        }
                    }).collect::<Vec<_>>();

                    if !tool_calls.is_empty() {
                        messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": null,
                            "tool_calls": tool_calls
                        }));
                    } else {
                        messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": Self::content_to_string(&msg.content)
                        }));
                    }
                }
                MessageRole::Tool => {
                    for content in &msg.content {
                        if let MessageContent::ToolResult {
                            tool_call_id,
                            result,
                            is_error: _,
                        } = content
                        {
                            messages.push(serde_json::json!({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": result
                            }));
                        }
                    }
                }
            }
        }

        messages
    }

    /// Converts message content to a string.
    fn content_to_string(content: &[MessageContent]) -> String {
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
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Value>,
    tools: Option<Vec<Value>>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    stream: bool,
}

#[async_trait]
impl LLMClient for OpenAIClient {
    async fn stream(&self, input: LLMInput) -> Result<LLMStream, LLMError> {
        let client = self.client.clone();
        let base_url = self.base_url.clone();

        // Note: MiniMax API does not support the OpenAI tool format
        let tools: Vec<Value> = Vec::new();

        let body = ChatRequest {
            model: input.model.clone(),
            messages: Self::build_messages(&input),
            tools: if tools.is_empty() { None } else { Some(tools) },
            max_tokens: Some(input.max_tokens),
            temperature: input.temperature,
            stream: true,
        };

        debug!(model = %input.model, "Starting streaming request to OpenAI");

        let response = client
            .post(&format!("{}/chat/completions", base_url))
            .json(&body)
            .send()
            .await
            .map_err(LLMError::NetworkError)?;

        if !response.status().is_success() {
            let error_text = response.text().await.map_err(LLMError::NetworkError)?;
            return Err(LLMError::ApiError(error_text));
        }

        let mut stream = response.bytes_stream();

        let s = stream! {
            let mut buffer = String::new();
            let mut current_tool_id: Option<String> = None;
            let mut current_tool_name: Option<String> = None;

            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(LLMError::NetworkError(e));
                        return;
                    }
                };

                let line = String::from_utf8_lossy(&chunk);

                if line.starts_with("data: ") {
                    let data = &line[6..];
                    if data == "[DONE]" {
                        break;
                    }

                    match serde_json::from_str::<ChatCompletionChunk>(data) {
                        Ok(chunk) => {
                            for choice in chunk.choices {
                                if let Some(ref delta) = choice.delta.content {
                                    yield Ok(LLMEvent::TextDelta {
                                        text: delta.clone()
                                    });
                                    buffer.clear();
                                }

                                if let Some(ref tool_calls) = choice.delta.tool_calls {
                                    for tool_call in tool_calls {
                                        if let Some(ref name) = tool_call.function.name {
                                            current_tool_id = Some(tool_call.id.clone());
                                            current_tool_name = Some(name.clone());
                                            yield Ok(LLMEvent::ToolCallStart {
                                                id: tool_call.id.clone(),
                                                name: name.clone(),
                                            });
                                        }

                                        if let Some(ref args) = tool_call.function.arguments {
                                            buffer.push_str(args);
                                            yield Ok(LLMEvent::ToolCallDelta {
                                                id: tool_call.id.clone(),
                                                arguments: args.clone(),
                                            });
                                        }
                                    }
                                }

                                if let Some(ref reason) = choice.finish_reason {
                                    if reason == "tool_calls" {
                                        if let (Some(_id), Some(_name)) = (current_tool_id.take(), current_tool_name.take()) {
                                            // Tool call ended
                                        }
                                    }

                                    let finish_reason = match reason.as_str() {
                                        "stop" => FinishReason::Stop,
                                        "tool_calls" => FinishReason::ToolCalls,
                                        "length" => FinishReason::MaxTokens,
                                        _ => FinishReason::Error,
                                    };

                                    yield Ok(LLMEvent::Finish {
                                        reason: finish_reason,
                                        usage: Usage {
                                            input_tokens: 0,
                                            output_tokens: 0,
                                        },
                                    });
                                }
                            }
                        }
                        Err(e) => {
                            tracing::debug!("Failed to parse chunk: {:?}", e);
                        }
                    }
                }
            }
        };

        Ok(Box::pin(s))
    }

    async fn complete(&self, input: LLMInput) -> Result<LLMOutput, LLMError> {
        let request = self.chat_completions_request(&input);

        let response_text = request
            .send()
            .await
            .map_err(LLMError::NetworkError)?
            .text()
            .await
            .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

        tracing::debug!("LLM response: {}", response_text);

        let response: ChatCompletionResponse = serde_json::from_str(&response_text)
            .map_err(|e| LLMError::InvalidResponse(format!("{}: {}", e, response_text)))?;

        if response.choices.is_empty() {
            return Err(LLMError::InvalidResponse(
                format!("No choices in response. Response: {}", response_text)
            ));
        }

        let mut content = Vec::new();

        for choice in response.choices {
            if let Some(ref tool_calls) = choice.message.tool_calls {
                for tool_call in tool_calls {
                    let arguments: Value = if tool_call.function.arguments.is_empty() {
                        serde_json::json!({})
                    } else {
                        serde_json::from_str(&tool_call.function.arguments)
                            .unwrap_or(serde_json::json!({}))
                    };

                    content.push(MessageContent::ToolCall {
                        id: tool_call.id.clone(),
                        name: tool_call.function.name.clone(),
                        arguments,
                    });
                }
            }

            if let Some(ref text) = choice.message.content {
                if !text.is_empty() {
                    content.push(MessageContent::Text {
                        text: text.clone(),
                    });
                }
            }

            let finish_reason = match choice.finish_reason.as_deref() {
                Some("stop") => FinishReason::Stop,
                Some("tool_calls") => FinishReason::ToolCalls,
                Some("length") => FinishReason::MaxTokens,
                _ => FinishReason::Error,
            };

            return Ok(LLMOutput {
                content,
                finish_reason,
                usage: Usage {
                    input_tokens: response.usage.prompt_tokens,
                    output_tokens: response.usage.completion_tokens,
                },
            });
        }

        // If we get here without returning, something went wrong
        Err(LLMError::InvalidResponse("No choices in response".to_string()))
    }
}
