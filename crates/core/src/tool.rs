use async_trait::async_trait;

use crate::LlmError;

#[derive(Clone, Debug)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters_schema: serde_json::Value,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    async fn call(&self, arguments: serde_json::Value) -> Result<String, LlmError>;
}
