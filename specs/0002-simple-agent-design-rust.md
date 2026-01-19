# Simple Multi-turn Agent with Tool Calling - Rust Design Specification

## 1. 核心概念

### 1.1 Agent 定义

Agent 是一个能够：

1. 接收用户消息
2. 调用 LLM 生成响应
3. 识别并执行工具调用
4. 将工具结果返回给 LLM
5. 循环直到任务完成

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Loop                            │
│                                                          │
│  User Input ──► LLM ──► Tool Calls? ──► Execute Tools   │
│       ▲                     │                  │         │
│       │                     ▼                  ▼         │
│       └──────── NO ◄── Continue? ◄─── Results ──┘        │
│                     │                                    │
│                     ▼ YES                                │
│                  Response                                │
└─────────────────────────────────────────────────────────┘
```

## 2. 核心数据结构

### 2.1 消息 (Message)

```rust
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub role: MessageRole,
    pub content: Vec<MessageContent>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContent {
    Text { text: String },
    ToolCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },
    ToolResult {
        tool_call_id: String,
        result: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

impl Message {
    pub fn new_user(text: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role: MessageRole::User,
            content: vec![MessageContent::Text { text: text.into() }],
            created_at: Utc::now(),
        }
    }

    pub fn new_assistant(content: Vec<MessageContent>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role: MessageRole::Assistant,
            content,
            created_at: Utc::now(),
        }
    }

    pub fn new_tool_result(results: Vec<MessageContent>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role: MessageRole::Tool,
            content: results,
            created_at: Utc::now(),
        }
    }
}
```

### 2.2 工具定义 (Tool)

```rust
use async_trait::async_trait;
use schemars::JsonSchema;
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value, // JSON Schema
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> Value;
    
    async fn execute(&self, args: Value) -> Result<ToolResult, ToolError>;
    
    fn to_definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.parameters_schema(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Tool not found: {0}")]
    NotFound(String),
}

pub type DynTool = Arc<dyn Tool>;
```

### 2.3 会话 (Session)

```rust
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub messages: Vec<Message>,
    pub system_prompt: String,
    pub model: ModelConfig,
    pub status: SessionStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionStatus {
    Idle,
    Running,
    Completed,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<HashMap<String, Value>>,
}

impl Session {
    pub fn new(model: ModelConfig, system_prompt: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            messages: Vec::new(),
            system_prompt: system_prompt.into(),
            model,
            status: SessionStatus::Idle,
        }
    }

    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }
}
```

## 3. 核心模块

### 3.1 LLM 模块

负责与 LLM 通信，支持流式响应。

```rust
use futures::stream::Stream;
use std::pin::Pin;

pub struct LLMInput {
    pub model: String,
    pub messages: Vec<Message>,
    pub system_prompt: String,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: u32,
}

pub struct LLMOutput {
    pub content: Vec<MessageContent>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    ToolCalls,
    MaxTokens,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Clone)]
pub enum LLMEvent {
    TextDelta { text: String },
    ToolCallStart { id: String, name: String },
    ToolCallDelta { id: String, arguments: String },
    ToolCallEnd { id: String },
    Finish { reason: FinishReason, usage: Usage },
    Error { error: String },
}

pub type LLMStream = Pin<Box<dyn Stream<Item = Result<LLMEvent, LLMError>> + Send>>;

#[async_trait]
pub trait LLMClient: Send + Sync {
    async fn stream(&self, input: LLMInput) -> Result<LLMStream, LLMError>;
    async fn complete(&self, input: LLMInput) -> Result<LLMOutput, LLMError>;
}

#[derive(Debug, thiserror::Error)]
pub enum LLMError {
    #[error("API error: {0}")]
    ApiError(String),
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),
    #[error("Invalid response: {0}")]
    InvalidResponse(String),
}
```

### 3.2 工具注册表 (Tool Registry)

管理所有可用工具。

```rust
use std::collections::HashMap;

pub struct ToolRegistry {
    tools: HashMap<String, DynTool>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: DynTool) {
        let name = tool.name().to_string();
        self.tools.insert(name, tool);
    }

    pub fn unregister(&mut self, name: &str) -> Option<DynTool> {
        self.tools.remove(name)
    }

    pub fn get(&self, name: &str) -> Option<&DynTool> {
        self.tools.get(name)
    }

    pub fn list(&self) -> Vec<&DynTool> {
        self.tools.values().collect()
    }

    pub fn to_tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|tool| tool.to_definition())
            .collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}
```

### 3.3 工具执行器 (Tool Executor)

执行工具调用并处理结果。

```rust
use tokio::sync::Mutex;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub session_id: String,
    pub message_id: String,
}

pub struct ToolExecutor {
    registry: Arc<Mutex<ToolRegistry>>,
}

impl ToolExecutor {
    pub fn new(registry: Arc<Mutex<ToolRegistry>>) -> Self {
        Self { registry }
    }

    pub async fn execute(
        &self,
        call: &MessageContent,
        ctx: ExecutionContext,
    ) -> MessageContent {
        let (id, name, arguments) = match call {
            MessageContent::ToolCall { id, name, arguments } => {
                (id.clone(), name.clone(), arguments.clone())
            }
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
```

### 3.4 Agent Loop

核心循环逻辑。

```rust
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct AgentConfig {
    pub model: String,
    pub system_prompt: String,
    pub max_steps: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: "claude-sonnet-4-20250514".to_string(),
            system_prompt: String::new(),
            max_steps: 200,
        }
    }
}

#[derive(Debug, Clone)]
pub enum AgentEvent {
    MessageStart { role: MessageRole },
    Text { text: String },
    ToolCall { name: String, args: Value },
    ToolResult { name: String, result: String },
    MessageEnd { finish_reason: FinishReason },
    Error { error: String },
}

pub struct Agent {
    session: Arc<Mutex<Session>>,
    llm_client: Arc<dyn LLMClient>,
    tool_executor: Arc<ToolExecutor>,
    config: AgentConfig,
}

impl Agent {
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

    pub async fn run(&self) -> Result<Vec<Message>, AgentError> {
        let mut step = 0;

        while step < self.config.max_steps {
            step += 1;

            // 1. 准备 LLM 输入
            let session = self.session.lock().await;
            let input = LLMInput {
                model: self.config.model.clone(),
                messages: session.messages.clone(),
                system_prompt: self.config.system_prompt.clone(),
                tools: vec![], // 从 registry 获取
                max_tokens: session.model.max_tokens,
            };
            drop(session);

            // 2. 调用 LLM
            let response = self.llm_client.complete(input).await?;

            // 3. 创建助手消息
            let assistant_message = Message::new_assistant(response.content.clone());
            let message_id = assistant_message.id.clone();
            
            {
                let mut session = self.session.lock().await;
                session.add_message(assistant_message);
            }

            // 4. 检查工具调用
            let tool_calls: Vec<MessageContent> = response
                .content
                .iter()
                .filter(|c| matches!(c, MessageContent::ToolCall { .. }))
                .cloned()
                .collect();

            if tool_calls.is_empty() {
                // 没有工具调用，循环结束
                break;
            }

            // 5. 执行工具调用
            let session_guard = self.session.lock().await;
            let session_id = session_guard.id.clone();
            drop(session_guard);

            let ctx = ExecutionContext {
                session_id,
                message_id,
            };

            let results = self.tool_executor.execute_all(tool_calls, ctx).await;

            // 6. 保存工具结果
            let tool_message = Message::new_tool_result(results);
            {
                let mut session = self.session.lock().await;
                session.add_message(tool_message);
            }
        }

        let session = self.session.lock().await;
        Ok(session.messages.clone())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("LLM error: {0}")]
    LLMError(#[from] LLMError),
    #[error("Max steps exceeded")]
    MaxStepsExceeded,
    #[error("Tool error: {0}")]
    ToolError(#[from] ToolError),
}
```

## 4. MCP 集成

### 4.1 MCP 客户端

支持从 MCP 服务器动态加载工具。

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPConfig {
    pub name: String,
    pub transport: MCPTransport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum MCPTransport {
    Stdio {
        command: String,
        args: Vec<String>,
    },
    Http {
        url: String,
    },
    Sse {
        url: String,
    },
}

pub struct MCPClient {
    config: MCPConfig,
    // MCP client implementation
}

impl MCPClient {
    pub fn new(config: MCPConfig) -> Self {
        Self { config }
    }

    pub async fn connect(&mut self) -> Result<(), MCPError> {
        // Connect to MCP server
        todo!()
    }

    pub async fn disconnect(&mut self) -> Result<(), MCPError> {
        // Disconnect from MCP server
        todo!()
    }

    pub async fn list_tools(&self) -> Result<Vec<ToolDefinition>, MCPError> {
        // Get tools from MCP server
        todo!()
    }

    pub async fn call_tool(
        &self,
        name: &str,
        args: Value,
    ) -> Result<ToolResult, MCPError> {
        // Call MCP tool
        todo!()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MCPError {
    #[error("Connection error: {0}")]
    ConnectionError(String),
    #[error("Protocol error: {0}")]
    ProtocolError(String),
}
```

### 4.2 MCP 工具适配

将 MCP 工具转换为本地 Tool 接口。

```rust
use std::sync::Arc;

pub struct MCPToolAdapter {
    client: Arc<MCPClient>,
    definition: ToolDefinition,
}

impl MCPToolAdapter {
    pub fn new(client: Arc<MCPClient>, definition: ToolDefinition) -> Self {
        Self { client, definition }
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
        self.client
            .call_tool(&self.definition.name, args)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))
    }
}

pub fn adapt_mcp_tools(
    client: Arc<MCPClient>,
    tools: Vec<ToolDefinition>,
) -> Vec<DynTool> {
    tools
        .into_iter()
        .map(|def| {
            Arc::new(MCPToolAdapter::new(client.clone(), def)) as DynTool
        })
        .collect()
}
```

## 5. 权限系统

### 5.1 权限检查

```rust
use regex::Regex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    pub tool: String, // 工具名称或通配符
    pub action: PermissionAction,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub patterns: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PermissionAction {
    Allow,
    Deny,
    Ask,
}

pub struct PermissionContext {
    pub tool: String,
    pub args: Value,
    pub session_id: String,
}

pub struct PermissionManager {
    rules: Vec<Permission>,
}

impl PermissionManager {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add_rule(&mut self, rule: Permission) {
        self.rules.push(rule);
    }

    pub async fn check(&self, ctx: &PermissionContext) -> PermissionResult {
        for rule in &self.rules {
            if self.matches(rule, ctx) {
                return match rule.action {
                    PermissionAction::Allow => PermissionResult::Allow,
                    PermissionAction::Deny => PermissionResult::Deny,
                    PermissionAction::Ask => self.ask_user(ctx).await,
                };
            }
        }
        PermissionResult::Deny // 默认拒绝
    }

    fn matches(&self, rule: &Permission, ctx: &PermissionContext) -> bool {
        // 检查工具名是否匹配（支持通配符）
        if rule.tool != "*" && rule.tool != ctx.tool {
            return false;
        }

        // 检查参数模式
        if let Some(patterns) = &rule.patterns {
            // 实现参数模式匹配逻辑
            true
        } else {
            true
        }
    }

    async fn ask_user(&self, ctx: &PermissionContext) -> PermissionResult {
        // 询问用户是否允许此操作
        // 可以通过回调、事件等方式实现
        PermissionResult::Deny
    }
}

#[derive(Debug, Clone)]
pub enum PermissionResult {
    Allow,
    Deny,
}

impl Default for PermissionManager {
    fn default() -> Self {
        Self::new()
    }
}
```

## 6. 流式处理

### 6.1 流式 Agent Loop

支持实时输出的流式处理。

```rust
use futures::stream::{Stream, StreamExt};
use std::pin::Pin;

pub type AgentStream = Pin<Box<dyn Stream<Item = AgentEvent> + Send>>;

impl Agent {
    pub async fn stream(&self) -> Result<AgentStream, AgentError> {
        let session = self.session.clone();
        let llm_client = self.llm_client.clone();
        let tool_executor = self.tool_executor.clone();
        let config = self.config.clone();
        let registry = Arc::clone(&self.tool_executor.registry);

        let stream = async_stream::stream! {
            let mut step = 0;

            while step < config.max_steps {
                step += 1;
                yield AgentEvent::MessageStart { 
                    role: MessageRole::Assistant 
                };

                // 准备 LLM 输入
                let session_guard = session.lock().await;
                let tool_defs = {
                    let reg = registry.lock().await;
                    reg.to_tool_definitions()
                };
                
                let input = LLMInput {
                    model: config.model.clone(),
                    messages: session_guard.messages.clone(),
                    system_prompt: config.system_prompt.clone(),
                    tools: tool_defs,
                    max_tokens: session_guard.model.max_tokens,
                };
                drop(session_guard);

                // 流式处理 LLM 响应
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
                let mut finish_reason = FinishReason::Stop;

                while let Some(event_result) = llm_stream.next().await {
                    match event_result {
                        Ok(LLMEvent::TextDelta { text }) => {
                            yield AgentEvent::Text { text: text.clone() };
                            content.push(MessageContent::Text { text });
                        }
                        Ok(LLMEvent::ToolCallEnd { id }) => {
                            // 构建完整的工具调用
                            if let Some(call) = build_tool_call(&id, &content) {
                                if let MessageContent::ToolCall { name, args, .. } = &call {
                                    yield AgentEvent::ToolCall {
                                        name: name.clone(),
                                        args: args.clone(),
                                    };
                                }
                                tool_calls.push(call);
                            }
                        }
                        Ok(LLMEvent::Finish { reason, .. }) => {
                            finish_reason = reason.clone();
                            yield AgentEvent::MessageEnd { finish_reason };
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

                // 保存助手消息
                let assistant_msg = Message::new_assistant(content);
                let msg_id = assistant_msg.id.clone();
                {
                    let mut session_guard = session.lock().await;
                    session_guard.add_message(assistant_msg);
                }

                // 无工具调用则结束
                if tool_calls.is_empty() {
                    break;
                }

                // 执行工具
                let session_id = {
                    let session_guard = session.lock().await;
                    session_guard.id.clone()
                };

                let ctx = ExecutionContext {
                    session_id,
                    message_id: msg_id,
                };

                let results = tool_executor.execute_all(tool_calls, ctx).await;

                // 输出工具结果
                for result in &results {
                    if let MessageContent::ToolResult { tool_call_id, result: res, .. } = result {
                        yield AgentEvent::ToolResult {
                            name: tool_call_id.clone(),
                            result: res.clone(),
                        };
                    }
                }

                // 保存工具结果
                let tool_msg = Message::new_tool_result(results);
                {
                    let mut session_guard = session.lock().await;
                    session_guard.add_message(tool_msg);
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

fn build_tool_call(id: &str, content: &[MessageContent]) -> Option<MessageContent> {
    // 从内容中构建完整的工具调用
    // 实现细节取决于如何累积工具调用片段
    None
}
```

## 7. 错误处理与重试

```rust
use tokio::time::{sleep, Duration};

#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub retryable_errors: Vec<String>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            retryable_errors: vec![
                "rate_limit".to_string(),
                "timeout".to_string(),
            ],
        }
    }
}

pub async fn with_retry<F, T, E>(
    f: F,
    config: &RetryConfig,
) -> Result<T, E>
where
    F: Fn() -> futures::future::BoxFuture<'static, Result<T, E>>,
    E: std::fmt::Display,
{
    let mut last_error = None;

    for attempt in 0..=config.max_retries {
        match f().await {
            Ok(result) => return Ok(result),
            Err(error) => {
                let error_str = error.to_string();
                
                if !is_retryable(&error_str, &config.retryable_errors) {
                    return Err(error);
                }

                last_error = Some(error);

                if attempt < config.max_retries {
                    let delay = config
                        .base_delay
                        .mul_f32(2_f32.powi(attempt as i32))
                        .min(config.max_delay);
                    sleep(delay).await;
                }
            }
        }
    }

    Err(last_error.unwrap())
}

fn is_retryable(error: &str, retryable_errors: &[String]) -> bool {
    retryable_errors
        .iter()
        .any(|pattern| error.contains(pattern))
}
```

## 8. 最小实现示例

### 8.1 完整的最小 Agent

```rust
use anthropic::{Client, messages::*};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化客户端
    let client = Client::from_env()?;

    // 工具定义
    let tools = vec![
        ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get current weather for a location".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }),
        }
    ];

    // 初始消息
    let mut messages = vec![
        Message::new_user("What's the weather in Tokyo?")
    ];

    // Agent 循环
    loop {
        // 1. 调用 LLM
        let response = client
            .messages()
            .create(MessagesRequest {
                model: "claude-sonnet-4-20250514".to_string(),
                max_tokens: 4096,
                messages: messages.clone(),
                tools: Some(tools.clone()),
                ..Default::default()
            })
            .await?;

        // 2. 处理响应
        let mut has_tool_use = false;
        let mut tool_results = Vec::new();

        for block in &response.content {
            match block {
                ContentBlock::Text { text } => {
                    println!("Assistant: {}", text);
                }
                ContentBlock::ToolUse { id, name, input } => {
                    has_tool_use = true;
                    println!("Calling tool: {}", name);

                    // 执行工具
                    let result = execute_tool(name, input).await?;
                    tool_results.push(MessageContent::ToolResult {
                        tool_call_id: id.clone(),
                        result,
                        is_error: None,
                    });
                }
            }
        }

        // 3. 添加助手消息
        messages.push(Message::new_assistant(response.content));

        // 4. 如果有工具调用，添加结果并继续
        if has_tool_use {
            messages.push(Message::new_tool_result(tool_results));
            continue;
        }

        // 5. 没有工具调用，结束循环
        if matches!(response.stop_reason, Some(StopReason::EndTurn)) {
            break;
        }
    }

    Ok(())
}

async fn execute_tool(
    name: &str,
    args: &serde_json::Value,
) -> Result<String, Box<dyn std::error::Error>> {
    match name {
        "get_weather" => {
            Ok(json!({
                "temp": 22,
                "condition": "sunny"
            }).to_string())
        }
        _ => Ok("Unknown tool".to_string()),
    }
}
```

## 9. 核心设计原则

### 9.1 Rust 特定的设计要点

1. **类型安全**：充分利用 Rust 的类型系统，使用枚举和结构体确保数据正确性
2. **所有权与生命周期**：合理使用 `Arc`、`Mutex` 管理共享状态
3. **异步编程**：使用 `tokio` 和 `async-trait` 实现异步操作
4. **错误处理**：使用 `Result` 和 `thiserror` 进行清晰的错误处理
5. **流式处理**：使用 `futures::Stream` 和 `async_stream` 实现流式 API
6. **并发安全**：使用 `Send + Sync` trait 确保线程安全

### 9.2 简化版设计要点

| 组件         | 必须 | 可选     | 主要 Crates |
|--------------|------|----------|-------------|
| Message 结构 | ✅    | -        | serde, chrono, uuid |
| Tool 定义    | ✅    | -        | async-trait, serde_json |
| LLM 调用     | ✅    | 流式处理 | tokio, futures |
| Agent Loop   | ✅    | -        | tokio, async-stream |
| 权限系统     | -    | ✅        | regex |
| MCP 集成     | -    | ✅        | - |
| 消息压缩     | -    | ✅        | - |
| 重试机制     | -    | ✅        | tokio::time |

## 10. 文件结构建议

```
src/
├── agent/
│   ├── mod.rs            # Agent 配置和主结构
│   └── loop.rs           # Agent 循环逻辑
├── llm/
│   ├── mod.rs            # LLM 模块入口
│   ├── client.rs         # LLM 客户端 trait
│   ├── anthropic.rs      # Anthropic 实现
│   └── stream.rs         # 流式处理
├── tool/
│   ├── mod.rs            # 工具模块入口
│   ├── registry.rs       # 工具注册表
│   ├── executor.rs       # 工具执行器
│   └── builtin/          # 内置工具
│       ├── mod.rs
│       ├── bash.rs
│       ├── read.rs
│       └── write.rs
├── mcp/
│   ├── mod.rs            # MCP 模块入口
│   ├── client.rs         # MCP 客户端
│   └── adapter.rs        # 工具适配器
├── session/
│   ├── mod.rs            # 会话模块入口
│   ├── session.rs        # 会话管理
│   └── message.rs        # 消息处理
├── permission/
│   ├── mod.rs            # 权限模块入口
│   └── manager.rs        # 权限管理器
├── error.rs              # 统一错误类型
├── lib.rs                # 库入口
└── main.rs               # CLI 入口
```

## 11. 依赖建议 (Cargo.toml)

```toml
[package]
name = "simple-agent"
version = "0.1.0"
edition = "2021"

[dependencies]
# 异步运行时
tokio = { version = "1", features = ["full"] }
futures = "0.3"
async-trait = "0.1"
async-stream = "0.3"

# 序列化
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# HTTP 客户端
reqwest = { version = "0.11", features = ["json", "stream"] }

# 时间
chrono = { version = "0.4", features = ["serde"] }

# UUID
uuid = { version = "1", features = ["v4", "serde"] }

# 错误处理
thiserror = "1"
anyhow = "1"

# JSON Schema
schemars = "0.8"

# 正则表达式
regex = "1"

# Anthropic SDK (可选)
anthropic = "0.1"

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

[dev-dependencies]
tokio-test = "0.4"
mockall = "0.12"
```

## 12. 参考资源

- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Tokio Async Runtime](https://tokio.rs/)
- [async-trait](https://docs.rs/async-trait/)
- [Rust Async Book](https://rust-lang.github.io/async-book/)
