use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Permission action types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PermissionAction {
    /// Allow the action
    Allow,
    /// Deny the action
    Deny,
    /// Ask user for permission
    Ask,
}

/// A permission rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    /// Tool name pattern (supports wildcards)
    pub tool: String,
    /// The action for this permission
    pub action: PermissionAction,
    /// Optional regex patterns for arguments
    #[serde(skip_serializing_if = "Option::is_none")]
    pub patterns: Option<Vec<String>>,
}

/// Context for permission checking.
#[derive(Debug, Clone)]
pub struct PermissionContext {
    /// The tool being called
    pub tool: String,
    /// The arguments being passed
    pub args: Value,
    /// The session ID
    pub session_id: String,
}

/// Result of a permission check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionResult {
    /// Action is allowed
    Allow,
    /// Action is denied
    Deny,
    /// User needs to be asked
    Ask,
}

/// Manages permissions for tool execution.
#[derive(Debug, Clone)]
pub struct PermissionManager {
    rules: Vec<Permission>,
}

impl PermissionManager {
    /// Creates a new permission manager.
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Adds a permission rule.
    pub fn add_rule(&mut self, rule: Permission) {
        self.rules.push(rule);
    }

    /// Checks if an action is permitted.
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
        PermissionResult::Deny // Default deny
    }

    /// Checks if a rule matches the context.
    fn matches(&self, rule: &Permission, ctx: &PermissionContext) -> bool {
        // Check tool name match (supports wildcards)
        if rule.tool != "*" && !self.tool_matches(&rule.tool, &ctx.tool) {
            return false;
        }

        // Check argument patterns if specified
        if let Some(patterns) = &rule.patterns {
            if !self.args_match(patterns, &ctx.args) {
                return false;
            }
        }

        true
    }

    /// Checks if a tool name matches a pattern (supports * wildcards).
    fn tool_matches(&self, pattern: &str, tool: &str) -> bool {
        if pattern == "*" {
            return true;
        }

        // Convert glob pattern to regex
        let regex_pattern = pattern
            .replace("**", ".*")  // ** matches any characters including /
            .replace("*", "[^/]*"); // * matches any characters except /

        Regex::new(&format!("^{}$", regex_pattern))
            .ok()
            .and_then(|re| Some(re.is_match(tool)))
            .unwrap_or(false)
    }

    /// Checks if arguments match the given patterns.
    fn args_match(&self, patterns: &[String], args: &Value) -> bool {
        // Simple pattern matching - check if arguments contain the pattern
        for pattern in patterns {
            if let Some(args_str) = args.as_str() {
                if args_str.contains(pattern) {
                    return true;
                }
            } else {
                // For object args, check if any value contains the pattern
                if let Some(obj) = args.as_object() {
                    for value in obj.values() {
                        if let Some(s) = value.as_str() {
                            if s.contains(pattern) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        patterns.is_empty()
    }

    /// Asks the user for permission (placeholder).
    async fn ask_user(&self, _ctx: &PermissionContext) -> PermissionResult {
        // In a real implementation, this would:
        // - Emit an event to ask the user
        // - Wait for user response
        // - Return the user's decision
        //
        // For now, we default to deny
        PermissionResult::Deny
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_matches() {
        let manager = PermissionManager::new();

        // Exact match
        assert!(manager.tool_matches("bash", "bash"));
        assert!(!manager.tool_matches("bash", "read"));

        // Wildcard match
        assert!(manager.tool_matches("*", "bash"));
        assert!(manager.tool_matches("*", "read"));

        // Glob patterns
        assert!(manager.tool_matches("file_*", "file_read"));
        assert!(!manager.tool_matches("file_*", "bash_read"));

        // Multiple wildcards
        assert!(manager.tool_matches("file_*.write", "file_test.write"));
        assert!(!manager.tool_matches("file_*.write", "other_test.write"));
    }
}
