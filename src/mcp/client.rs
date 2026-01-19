use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::process::{Command, Stdio, Child};
use std::io::{BufReader, Write, BufRead};
use std::sync::{atomic::AtomicU64, Mutex, Arc};
use std::time::Duration;
use tracing::debug;
use tokio::task;

/// Configuration for connecting to an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPConfig {
    /// Name of the MCP server
    pub name: String,
    /// Transport type and configuration
    pub transport: MCPTransport,
    /// Connection timeout
    #[serde(default = "default_timeout")]
    pub timeout: Duration,
}

fn default_timeout() -> Duration {
    Duration::from_secs(30)
}

/// Transport type for MCP connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum MCPTransport {
    /// Connect via stdin/stdout of a process
    Stdio {
        /// The command to run
        command: String,
        /// Command arguments
        args: Vec<String>,
        /// Environment variables
        #[serde(skip_serializing_if = "Option::is_none")]
        env: Option<std::collections::HashMap<String, String>>,
    },
    /// Connect via HTTP
    Http {
        /// The URL of the MCP server
        url: String,
    },
    /// Connect via Server-Sent Events
    Sse {
        /// The URL of the SSE endpoint
        url: String,
        /// Optional authentication header
        #[serde(skip_serializing_if = "Option::is_none")]
        auth: Option<String>,
    },
}

/// Errors from MCP operations.
#[derive(Debug, thiserror::Error)]
pub enum MCPError {
    /// Connection error
    #[error("Connection error: {0}")]
    ConnectionError(String),
    /// Protocol error
    #[error("Protocol error: {0}")]
    ProtocolError(String),
    /// Tool not found
    #[error("Tool not found: {0}")]
    ToolNotFound(String),
    /// Execution error
    #[error("Execution error: {0}")]
    ExecutionError(String),
    /// Timeout
    #[error("Timeout")]
    Timeout,
    /// HTTP error
    #[error("HTTP error: {0}")]
    HttpError(String),
}

/// Builder for MCP client.
#[derive(Debug, Default)]
pub struct MCPClientBuilder {
    name: Option<String>,
    transport: Option<MCPTransport>,
    timeout: Option<Duration>,
}

impl MCPClientBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the server name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Configures stdio transport.
    pub fn with_stdio_transport(
        mut self,
        command: impl Into<String>,
        args: Vec<String>,
    ) -> Self {
        self.transport = Some(MCPTransport::Stdio {
            command: command.into(),
            args,
            env: None,
        });
        self
    }

    /// Configures HTTP transport.
    pub fn with_http_transport(mut self, url: impl Into<String>) -> Self {
        self.transport = Some(MCPTransport::Http {
            url: url.into(),
        });
        self
    }

    /// Configures SSE transport.
    pub fn with_sse_transport(mut self, url: impl Into<String>) -> Self {
        self.transport = Some(MCPTransport::Sse {
            url: url.into(),
            auth: None,
        });
        self
    }

    /// Sets the timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Builds the MCP client.
    pub fn build(self) -> Result<MCPClient, MCPError> {
        let name = self.name.ok_or_else(|| MCPError::ConnectionError(
            "MCP server name is required".to_string(),
        ))?;
        let transport = self.transport.ok_or_else(|| MCPError::ConnectionError(
            "MCP transport is required".to_string(),
        ))?;
        let timeout = self.timeout.unwrap_or_else(default_timeout);

        Ok(MCPClient {
            config: MCPConfig { name, transport, timeout },
            process: None,
            stdin: None,
            stdout_reader: None,
            http_client: None,
            sse_url: None,
            message_id: AtomicU64::new(0),
        })
    }
}

/// A client for connecting to MCP servers.
#[derive(Debug)]
pub struct MCPClient {
    config: MCPConfig,
    // Stdio transport fields
    process: Option<Child>,
    stdin: Option<std::process::ChildStdin>,
    stdout_reader: Option<Arc<Mutex<BufReader<std::process::ChildStdout>>>>,
    // HTTP/SSE transport fields
    http_client: Option<reqwest::Client>,
    sse_url: Option<String>,
    // Message ID counter for JSON-RPC
    message_id: AtomicU64,
}

impl MCPClient {
    /// Creates a new builder.
    pub fn builder() -> MCPClientBuilder {
        MCPClientBuilder::new()
    }

    /// Connects to the MCP server.
    pub async fn connect(&mut self) -> Result<(), MCPError> {
        // Clone the transport config so we can use it while keeping self borrowed
        let transport = self.config.transport.clone();

        match transport {
            MCPTransport::Stdio { command, args, env } => {
                self.connect_stdio(&command, &args, &env).await
            }
            MCPTransport::Http { url } => {
                self.connect_http(&url).await
            }
            MCPTransport::Sse { url, .. } => {
                self.connect_sse(&url).await
            }
        }
    }

    /// Connects via stdio.
    async fn connect_stdio(
        &mut self,
        command: &str,
        args: &[String],
        env: &Option<HashMap<String, String>>,
    ) -> Result<(), MCPError> {
        debug!("Starting MCP server: {} {:?}", command, args);

        let mut cmd = Command::new(command);
        cmd.args(args);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());

        if let Some(env_vars) = env {
            for (key, value) in env_vars {
                cmd.env(key, value);
            }
        }

        let mut process = cmd.spawn().map_err(|e| {
            MCPError::ConnectionError(format!("Failed to start MCP server: {}", e))
        })?;

        let stdin = process.stdin.take().ok_or_else(|| {
            MCPError::ConnectionError("Failed to get stdin".to_string())
        })?;

        let stdout = process.stdout.take().ok_or_else(|| {
            MCPError::ConnectionError("Failed to get stdout".to_string())
        })?;

        self.process = Some(process);
        self.stdin = Some(stdin);
        self.stdout_reader = Some(Arc::new(Mutex::new(BufReader::new(stdout))));

        // Send initialize message and read response
        self.send_initialize().await?;
        let _init_response = self.read_json_response().await?;

        debug!("MCP server initialized successfully");

        Ok(())
    }

    /// Connects via HTTP.
    async fn connect_http(&mut self, url: &str) -> Result<(), MCPError> {
        debug!("Connecting to MCP server via HTTP: {}", url);

        let client = reqwest::Client::builder()
            .timeout(self.config.timeout)
            .build()
            .map_err(|e| MCPError::ConnectionError(e.to_string()))?;

        // Verify the connection by sending an initialize request
        let response = client
            .post(&format!("{}/rpc", url))
            .header("Content-Type", "application/json")
            .json(&self.create_initialize_request())
            .send()
            .await
            .map_err(|e| MCPError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(MCPError::HttpError(format!(
                "HTTP error: {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
            )));
        }

        self.http_client = Some(client);
        debug!("Successfully connected to MCP server via HTTP");
        Ok(())
    }

    /// Connects via SSE.
    async fn connect_sse(&mut self, url: &str) -> Result<(), MCPError> {
        debug!("Connecting to MCP server via SSE: {}", url);

        let client = reqwest::Client::builder()
            .timeout(self.config.timeout)
            .build()
            .map_err(|e| MCPError::ConnectionError(e.to_string()))?;

        self.http_client = Some(client);
        self.sse_url = Some(url.to_string());

        debug!("Successfully connected to MCP server via SSE");
        Ok(())
    }

    /// Creates the initialize request.
    fn create_initialize_request(&self) -> Value {
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "simple-agent",
                    "version": "0.1.0"
                }
            }
        })
    }

    /// Sends the initialize message.
    async fn send_initialize(&mut self) -> Result<(), MCPError> {
        let message = self.create_initialize_request();
        self.send_message(message).await
    }

    /// Sends a JSON-RPC message.
    async fn send_message(&mut self, message: Value) -> Result<(), MCPError> {
        match &self.config.transport {
            MCPTransport::Stdio { .. } => {
                self.send_message_stdio(message).await
            }
            MCPTransport::Http { url } => {
                self.send_message_http(message, url).await
            }
            MCPTransport::Sse { url, .. } => {
                // For SSE, we use the same HTTP endpoint for requests
                self.send_message_http(message, url).await
            }
        }
    }

    /// Sends a message via stdio.
    async fn send_message_stdio(&mut self, message: Value) -> Result<(), MCPError> {
        let message_str = serde_json::to_string(&message)
            .map_err(|e| MCPError::ProtocolError(e.to_string()))?;

        let stdin = self.stdin.as_mut().ok_or_else(|| {
            MCPError::ConnectionError("Not connected".to_string())
        })?;

        stdin.write_all(message_str.as_bytes()).map_err(|e| {
            MCPError::ConnectionError(format!("Failed to write to stdin: {}", e))
        })?;
        stdin.write_all(b"\n").map_err(|e| {
            MCPError::ConnectionError(format!("Failed to write newline: {}", e))
        })?;

        Ok(())
    }

    /// Reads a JSON-RPC response line from stdout, skipping non-JSON lines.
    async fn read_json_response(&self) -> Result<Value, MCPError> {
        let reader_arc = self.stdout_reader.as_ref().ok_or_else(|| {
            MCPError::ConnectionError("Not connected".to_string())
        })?;

        // Keep reading until we get valid JSON
        loop {
            // Clone the Arc for each iteration
            let reader_arc_clone = reader_arc.clone();

            let line = task::spawn_blocking(move || {
                let mut reader = reader_arc_clone.lock().map_err(|e| {
                    MCPError::ProtocolError(format!("Failed to lock reader: {}", e))
                })?;

                let mut line = String::new();
                reader.read_line(&mut line).map_err(|e| {
                    MCPError::ProtocolError(format!("Failed to read response: {}", e))
                })?;

                // Trim the line
                let trimmed = line.trim();

                // Skip empty lines
                if trimmed.is_empty() {
                    return Ok::<Option<String>, MCPError>(None);
                }

                // Try to parse as JSON
                match serde_json::from_str::<Value>(trimmed) {
                    Ok(json) => Ok(Some(serde_json::to_string(&json).unwrap_or(trimmed.to_string()))),
                    Err(_) => {
                        // Not JSON, might be a log message - print it for debugging
                        tracing::debug!("Skipping non-JSON line: {}", trimmed);
                        Ok(None)
                    }
                }
            }).await.map_err(|e| {
                MCPError::ProtocolError(format!("Task error: {}", e))
            })?;

            if let Ok(Some(json_line)) = line {
                return serde_json::from_str(&json_line)
                    .map_err(|e| MCPError::ProtocolError(format!("Failed to parse response: {}", e)));
            }
            // If line was None (empty or non-JSON), continue the loop
        }
    }

    /// Sends a message via HTTP.
    async fn send_message_http(&self, message: Value, url: &str) -> Result<(), MCPError> {
        let client = self.http_client.as_ref().ok_or_else(|| {
            MCPError::ConnectionError("Not connected".to_string())
        })?;

        let response = client
            .post(&format!("{}/rpc", url))
            .header("Content-Type", "application/json")
            .json(&message)
            .send()
            .await
            .map_err(|e| MCPError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(MCPError::HttpError(format!(
                "HTTP error: {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
            )));
        }

        Ok(())
    }

    /// Disconnects from the MCP server.
    pub async fn disconnect(&mut self) -> Result<(), MCPError> {
        // Clean up stdio transport
        if let Some(mut process) = self.process.take() {
            process.wait().map_err(|e| {
                MCPError::ConnectionError(format!("Failed to wait for process: {}", e))
            })?;
        }

        self.stdin = None;
        self.stdout_reader = None;

        // Clean up HTTP/SSE transport
        self.http_client = None;
        self.sse_url = None;

        Ok(())
    }

    /// Lists available tools from the MCP server.
    pub async fn list_tools(&mut self) -> Result<Vec<MCToolInfo>, MCPError> {
        let request = self.create_json_rpc_request("tools/list", Value::Object(serde_json::Map::new()));

        match &self.config.transport {
            MCPTransport::Stdio { .. } => {
                // Send the tools/list request
                self.send_message_stdio(request).await?;

                // Read the JSON response (skips non-JSON lines)
                let response = self.read_json_response().await?;

                // Check for JSON-RPC error
                if let Some(error) = response.get("error") {
                    return Err(MCPError::ExecutionError(
                        error.to_string()
                    ));
                }

                // Extract tools from result
                let result = response.get("result")
                    .ok_or_else(|| MCPError::ProtocolError("No result in response".to_string()))?;

                let tools: Vec<MCToolInfo> = serde_json::from_value(result.get("tools")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!([])))
                    .map_err(|e| MCPError::ProtocolError(e.to_string()))?;

                Ok(tools)
            }
            MCPTransport::Http { url } => {
                let response: ToolsListResponse = self.call_json_rpc_method(request, url).await?;
                Ok(response.tools)
            }
            MCPTransport::Sse { url, .. } => {
                let response: ToolsListResponse = self.call_json_rpc_method(request, url).await?;
                Ok(response.tools)
            }
        }
    }

    /// Calls a tool on the MCP server.
    pub async fn call_tool(
        &mut self,
        name: &str,
        arguments: Value,
    ) -> Result<String, MCPError> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments
        });

        let request = self.create_json_rpc_request("tools/call", params);

        match &self.config.transport {
            MCPTransport::Stdio { .. } => {
                // Send the tool call request
                self.send_message_stdio(request).await?;

                // Read the JSON response (skips non-JSON lines)
                let response = self.read_json_response().await?;

                // Check for JSON-RPC error
                if let Some(error) = response.get("error") {
                    return Err(MCPError::ExecutionError(
                        error.to_string()
                    ));
                }

                // Extract result
                let result = response.get("result")
                    .ok_or_else(|| MCPError::ProtocolError("No result in response".to_string()))?;

                // Format the tool result
                self.extract_tool_result(result.clone())
            }
            MCPTransport::Http { url } => {
                let response: Value = self.call_json_rpc_method(request, url).await?;
                self.extract_tool_result(response)
            }
            MCPTransport::Sse { url, .. } => {
                let response: Value = self.call_json_rpc_method(request, url).await?;
                self.extract_tool_result(response)
            }
        }
    }

    /// Creates a JSON-RPC request.
    fn create_json_rpc_request(&self, method: &str, params: Value) -> Value {
        let id = self.message_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        })
    }

    /// Calls a JSON-RPC method and returns the result.
    async fn call_json_rpc_method<T: serde::de::DeserializeOwned>(
        &self,
        request: Value,
        url: &str,
    ) -> Result<T, MCPError> {
        let client = self.http_client.as_ref().ok_or_else(|| {
            MCPError::ConnectionError("Not connected".to_string())
        })?;

        let response = client
            .post(&format!("{}/rpc", url))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| MCPError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(MCPError::HttpError(format!(
                "HTTP error: {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
            )));
        }

        let response_value: Value = response
            .json()
            .await
            .map_err(|e| MCPError::ProtocolError(e.to_string()))?;

        // Check for JSON-RPC error
        if let Some(error) = response_value.get("error") {
            return Err(MCPError::ExecutionError(
                error.to_string()
            ));
        }

        // Extract result
        let result = response_value.get("result")
            .ok_or_else(|| MCPError::ProtocolError("No result in response".to_string()))?
            .clone();

        serde_json::from_value(result)
            .map_err(|e| MCPError::ProtocolError(e.to_string()))
    }

    /// Extracts the tool result from a JSON-RPC response.
    fn extract_tool_result(&self, response: Value) -> Result<String, MCPError> {
        // Handle the MCP tool call response format
        // The response may contain structured content
        Ok(response.to_string())
    }
}

/// Information about a tool from the MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCToolInfo {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub input_schema: Value,
}

/// Response from tools/list method.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolsListResponse {
    pub tools: Vec<MCToolInfo>,
}

impl Drop for MCPClient {
    fn drop(&mut self) {
        if let Some(mut process) = self.process.take() {
            let _ = process.kill();
        }
    }
}
