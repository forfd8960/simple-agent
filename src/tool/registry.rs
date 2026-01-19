use std::collections::HashMap;
use std::fmt;
use crate::tool::DynTool;

/// A registry for managing tools available to the agent.
#[derive(Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, DynTool>,
}

impl ToolRegistry {
    /// Creates a new empty tool registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Registers a tool with the registry.
    pub fn register(&mut self, tool: DynTool) {
        let name = tool.name().to_string();
        self.tools.insert(name, tool);
    }

    /// Unregisters a tool from the registry.
    pub fn unregister(&mut self, name: &str) -> Option<DynTool> {
        self.tools.remove(name)
    }

    /// Gets a tool by name.
    pub fn get(&self, name: &str) -> Option<&DynTool> {
        self.tools.get(name)
    }

    /// Returns a list of all registered tools.
    pub fn list(&self) -> Vec<&DynTool> {
        self.tools.values().collect()
    }

    /// Returns the number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Converts all tools to their definitions.
    pub fn to_tool_definitions(&self) -> Vec<crate::tool::ToolDefinition> {
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

impl fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools_count", &self.tools.len())
            .finish()
    }
}

impl IntoIterator for ToolRegistry {
    type Item = (String, DynTool);
    type IntoIter = std::collections::hash_map::IntoIter<String, DynTool>;

    fn into_iter(self) -> Self::IntoIter {
        self.tools.into_iter()
    }
}

impl<'a> IntoIterator for &'a ToolRegistry {
    type Item = (&'a String, &'a DynTool);
    type IntoIter = std::collections::hash_map::Iter<'a, String, DynTool>;

    fn into_iter(self) -> Self::IntoIter {
        self.tools.iter()
    }
}
