use std::sync::Arc;
use tokio::sync::Mutex;
use futures::stream::{Stream, StreamExt};
use std::pin::Pin;
use tracing::debug;

use crate::session::{Message, MessageContent, MessageRole, Session, SessionStatus};
use crate::llm::{LLMClient, LLMInput, LLMEvent, FinishReason};
use crate::tool::{ToolExecutor, ToolRegistry, ExecutionContext};

/// Configuration for the agent.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// The model to use
    pub model: String,
    /// The system prompt
    pub system_prompt: String,
    /// Maximum number of steps in the agent loop
    pub max_steps: usize,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Optional temperature
    pub temperature: Option<f32>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: "MiniMax-M2.1".to_string(),
            system_prompt: String::new(),
            max_steps: 100,
            max_tokens: 4096,
            temperature: None,
        }
    }
}

/// Events from the agent during execution.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// A new message is starting
    MessageStart {
        role: MessageRole,
    },
    /// Text content was received
    Text {
        text: String,
    },
    /// A tool is being called
    ToolCall {
        name: String,
        args: serde_json::Value,
    },
    /// A tool result was received
    ToolResult {
        name: String,
        result: String,
    },
    /// The message is complete
    MessageEnd {
        finish_reason: FinishReason,
    },
    /// An error occurred
    Error {
        error: String,
    },
}

/// A stream of agent events.
pub type AgentStream = Pin<Box<dyn Stream<Item = AgentEvent> + Send>>;

/// Errors from the agent.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// An LLM error occurred
    #[error("LLM error: {0}")]
    LLMError(#[from] crate::llm::LLMError),
    /// Maximum steps exceeded
    #[error("Max steps exceeded")]
    MaxStepsExceeded,
    /// A tool error occurred
    #[error("Tool error: {0}")]
    ToolError(#[from] crate::tool::ToolError),
}

/// The agent that can run conversations with tools.
#[derive(Clone)]
pub struct Agent {
    session: Arc<Mutex<Session>>,
    llm_client: Arc<dyn LLMClient>,
    tool_executor: Arc<ToolExecutor>,
    config: AgentConfig,
}

impl Agent {
    /// Creates a new agent.
    pub fn new(
        session: Session,
        llm_client: Arc<dyn LLMClient>,
        registry: Arc<Mutex<ToolRegistry>>,
        config: AgentConfig,
    ) -> Self {
        let tool_executor = Arc::new(ToolExecutor::new(registry));
        Self {
            session: Arc::new(Mutex::new(session)),
            llm_client,
            tool_executor,
            config,
        }
    }

    /// Creates a new agent with default configuration.
    pub fn with_defaults(
        session: Session,
        llm_client: Arc<dyn LLMClient>,
        registry: Arc<Mutex<ToolRegistry>>,
    ) -> Self {
        Self::new(session, llm_client, registry, AgentConfig::default())
    }

    /// Adds a user message to the session and runs the agent.
    pub async fn run(&self, user_input: &str) -> Result<Vec<Message>, AgentError> {
        let mut session = self.session.lock().await;
        session.add_message(Message::new_user(user_input));
        session.status = SessionStatus::Running;
        drop(session);

        let messages = self.run_loop().await?;

        let mut session = self.session.lock().await;
        session.status = SessionStatus::Completed;

        Ok(messages)
    }

    /// Runs the agent loop until completion.
    async fn run_loop(&self) -> Result<Vec<Message>, AgentError> {
        let mut step = 0;

        while step < self.config.max_steps {
            step += 1;

            // Get tool definitions from the registry
            let tool_defs = self.tool_executor.get_tool_definitions().await;
            debug!(count = tool_defs.len(), "Tool definitions loaded");

            // Prepare LLM input
            let session = self.session.lock().await;
            let input = LLMInput {
                model: self.config.model.clone(),
                messages: session.messages.clone(),
                system_prompt: self.config.system_prompt.clone(),
                tools: tool_defs,
                max_tokens: session.model.max_tokens,
                temperature: self.config.temperature,
            };
            drop(session);

            debug!(step, "Calling LLM");

            // Call LLM
            let response = self.llm_client.complete(input).await?;

            // Create assistant message
            let assistant_message = Message::new_assistant(response.content.clone());
            let message_id = assistant_message.id.clone();

            {
                let mut session = self.session.lock().await;
                session.add_message(assistant_message);
            }

            // Check for tool calls
            let tool_calls: Vec<MessageContent> = response
                .content
                .iter()
                .filter(|c| matches!(c, MessageContent::ToolCall { .. }))
                .cloned()
                .collect();

            if tool_calls.is_empty() {
                // No tool calls, loop ends
                break;
            }

            debug!(count = tool_calls.len(), "Executing tool calls");

            // Execute tool calls
            let session_id = {
                let session = self.session.lock().await;
                session.id.clone()
            };

            let ctx = ExecutionContext {
                session_id,
                message_id,
            };

            let results = self.tool_executor.execute_all(tool_calls, ctx).await;

            // Save tool results
            let tool_message = Message::new_tool_result(results);
            {
                let mut session = self.session.lock().await;
                session.add_message(tool_message);
            }
        }

        let session = self.session.lock().await;
        Ok(session.messages.clone())
    }

    /// Runs the agent with streaming output.
    pub async fn stream(&self) -> Result<AgentStream, AgentError> {
        let session = self.session.clone();
        let llm_client = self.llm_client.clone();
        let tool_executor = self.tool_executor.clone();
        let config = self.config.clone();

        let stream = async_stream::stream! {
            let mut step = 0;

            while step < config.max_steps {
                step += 1;

                yield AgentEvent::MessageStart {
                    role: MessageRole::Assistant
                };

                // Get tool definitions from the registry
                let tool_defs = tool_executor.get_tool_definitions().await;

                // Prepare LLM input
                let session_guard = session.lock().await;
                let input = LLMInput {
                    model: config.model.clone(),
                    messages: session_guard.messages.clone(),
                    system_prompt: config.system_prompt.clone(),
                    tools: tool_defs,
                    max_tokens: session_guard.model.max_tokens,
                    temperature: config.temperature,
                };
                drop(session_guard);

                // Stream LLM response
                let mut llm_stream = match llm_client.stream(input).await {
                    Ok(stream) => stream,
                    Err(e) => {
                        yield AgentEvent::Error {
                            error: e.to_string()
                        };
                        return;
                    }
                };

                let mut content = Vec::new();
                let mut tool_calls = Vec::new();
                let _finish_reason = FinishReason::Stop;
                let _current_tool_id: Option<String> = None;

                while let Some(event_result) = llm_stream.next().await {
                    match event_result {
                        Ok(LLMEvent::TextDelta { text }) => {
                            yield AgentEvent::Text { text: text.clone() };
                            content.push(MessageContent::Text { text });
                        }
                        Ok(LLMEvent::ToolCallStart { id, name }) => {
                            content.push(MessageContent::ToolCall {
                                id: id.clone(),
                                name: name.clone(),
                                arguments: serde_json::json!({}),
                            });
                        }
                        Ok(LLMEvent::ToolCallDelta { id: _, arguments }) => {
                            // Update the arguments in the last tool call
                            if let Some(last) = content.last_mut() {
                                if let MessageContent::ToolCall { arguments: args, .. } = last {
                                    *args = serde_json::from_str(&arguments)
                                        .unwrap_or(serde_json::json!({}));
                                }
                            }
                        }
                        Ok(LLMEvent::ToolCallEnd { id }) => {
                            // Collect the completed tool call
                            if let Some(pos) = content.iter().position(|c| {
                                if let MessageContent::ToolCall { id: tool_id, .. } = c {
                                    tool_id == &id
                                } else {
                                    false
                                }
                            }) {
                                let call = content.remove(pos);
                                tool_calls.push(call);
                            }
                        }
                        Ok(LLMEvent::Finish { reason, .. }) => {
                            yield AgentEvent::MessageEnd { finish_reason: reason.clone() };
                        }
                        Err(e) => {
                            yield AgentEvent::Error {
                                error: e.to_string()
                            };
                            return;
                        }
                        _ => {}
                    }
                }

                // Save assistant message
                let assistant_msg = Message::new_assistant(content);
                let msg_id = assistant_msg.id.clone();
                {
                    let mut session_guard = session.lock().await;
                    session_guard.add_message(assistant_msg);
                }

                // No tool calls, loop ends
                if tool_calls.is_empty() {
                    break;
                }

                // Execute tools
                let session_id = {
                    let session_guard = session.lock().await;
                    session_guard.id.clone()
                };

                let ctx = ExecutionContext {
                    session_id,
                    message_id: msg_id,
                };

                let results = tool_executor.execute_all(tool_calls, ctx).await;

                // Output tool results
                for result in &results {
                    if let MessageContent::ToolResult {
                        tool_call_id,
                        result: res,
                        ..
                    } = result
                    {
                        yield AgentEvent::ToolResult {
                            name: tool_call_id.clone(),
                            result: res.clone(),
                        };
                    }
                }

                // Save tool results
                let tool_msg = Message::new_tool_result(results);
                {
                    let mut session_guard = session.lock().await;
                    session_guard.add_message(tool_msg);
                }
            }
        };

        Ok(Box::pin(stream))
    }

    /// Gets the session ID.
    pub async fn session_id(&self) -> String {
        let session = self.session.lock().await;
        session.id.clone()
    }

    /// Gets the current messages.
    pub async fn messages(&self) -> Vec<Message> {
        let session = self.session.lock().await;
        session.messages.clone()
    }
}
