# Simple Agent

## 项目结构

```sh
  simple-agent/
  ├── Cargo.toml
  ├── src/
  │   ├── lib.rs                    # 主入口和 re-exports
  │   ├── agent/                    # Agent 核心
  │   │   └── loop_.rs             # Agent 循环和配置
  │   ├── llm/                      # LLM 客户端
  │   │   ├── client.rs             # LLM trait 和 builder
  │   │   └── openai.rs             # OpenAI 客户端实现
  │   ├── session/                  # 会话管理
  │   │   ├── message.rs            # Message 类型
  │   │   └── session.rs            # Session 类型
  │   ├── tool/                     # 工具系统
  │   │   ├── mod.rs                # Tool trait
  │   │   ├── registry.rs           # ToolRegistry
  │   │   └── executor.rs           # ToolExecutor
  │   ├── mcp/                      # MCP 支持
  │   │   ├── client.rs             # MCP 客户端
  │   │   └── adapter.rs            # MCP 工具适配器
  │   └── permission/               # 权限管理
  │       └── manager.rs            # PermissionManager
  └── examples/
      ├── basic_agent.rs            # 基础 agent 示例
      ├── custom_tools.rs           # 自定义工具示例
      └── mcp_client.rs             # MCP 客户端示例
```

## 核心功能实现

  1. Session 模块 (src/session/)
    - Message: 消息类型，支持 User/Assistant/Tool 角色
    - MessageContent: 文本、工具调用、工具结果
    - Session: 会话管理
    - ModelConfig: 模型配置
  2. Tool 模块 (src/tool/)
    - Tool trait: 自定义工具接口
    - ToolRegistry: 工具注册表
    - ToolExecutor: 工具执行器
  3. LLM 模块 (src/llm/)
    - LLMClient trait: LLM 客户端接口
    - OpenAIClient: OpenAI API 实现
    - 支持流式响应
  4. Agent 模块 (src/agent/)
    - Agent: 主 agent 类型
    - AgentConfig: 配置
    - 支持多轮对话和工具调用
    - 支持流式输出
  5. MCP 模块 (src/mcp/)
    - MCPClient: MCP 客户端
    - 支持 stdio、HTTP、SSE 传输
    - MCPToolAdapter: MCP 工具适配器
  6. Permission 模块 (src/permission/)
    - PermissionManager: 权限管理
    - 支持通配符匹配


## 示例用法

  1. 基础 Agent (examples/basic_agent.rs)

  let llm_client = LLMClientBuilder::new()
      .with_api_key(api_key)
      .build_openai()?;

  let registry = Arc::new(Mutex::new(ToolRegistry::new()));
  let session = Session::new(ModelConfig::default(), "You are a helpful assistant.");
  let agent = Agent::with_defaults(session, llm_client, registry);

  let messages = agent.run("Hello!").await?;

  2. 自定义工具 (examples/custom_tools.rs)

  #[async_trait]
  impl Tool for WeatherTool {
      fn name(&self) -> &str { "get_weather" }
      fn description(&self) -> &str { "Get weather for a location" }
      fn parameters_schema(&self) -> Value { ... }
      async fn execute(&self, args: Value) -> Result<ToolResult, ToolError> { ... }
  }

  3. MCP 客户端 (examples/mcp_client.rs)

  let mut mcp_client = MCPClient::builder()
      .with_name("filesystem")
      .with_stdio_transport("npx", vec!["@modelcontextprotocol/server-filesystem", "/path"])
      .build()?;

  mcp_client.connect().await?;

## 运行示例：

```sh
  export OPENAI_API_KEY="your-key"
  cargo run --example basic_agent
  cargo run --example custom_tools
  cargo run --example mcp_client
```