use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;

use crate::{Message, ToolDefinition};

pub type LlmError = Box<dyn std::error::Error + Send + Sync>;
pub type LlmStream = Pin<Box<dyn Stream<Item = Result<LlmEvent, LlmError>> + Send>>;

#[derive(Debug)]
pub enum LlmEvent {
    TextDelta(String),
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: String,
    },
}

#[async_trait]
pub trait Llm: Send + Sync {
    async fn send(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
    ) -> Result<LlmStream, LlmError>;
}
