use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures_util::Stream;
use std::pin::Pin;
use std::sync::Arc;
use telesema_core::{ChannelUri, Event, EventPayload, MessageUri, SenderUri};

use crate::LlmError;

pub type AgentStream = Pin<Box<dyn Stream<Item = Result<AgentEvent, LlmError>> + Send>>;

pub enum AgentEvent {
    TextDelta(String),
    ToolCallStarted { id: String, name: String },
    ToolCallReady { id: String, name: String, arguments: String },
    ToolCallFinished { id: String, name: String, summary: String },
    ToolCallFailed { id: String, name: String, error: String },
    /// A side-channel event the agent wants forwarded onto the bus verbatim
    /// (e.g. an extension event that mutates persistent state).
    Custom(Arc<dyn EventPayload>),
}

impl std::fmt::Debug for AgentEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TextDelta(text) => f.debug_tuple("TextDelta").field(text).finish(),
            Self::ToolCallStarted { id, name } => f
                .debug_struct("ToolCallStarted")
                .field("id", id)
                .field("name", name)
                .finish(),
            Self::ToolCallReady { id, name, arguments } => f
                .debug_struct("ToolCallReady")
                .field("id", id)
                .field("name", name)
                .field("arguments", arguments)
                .finish(),
            Self::ToolCallFinished { id, name, summary } => f
                .debug_struct("ToolCallFinished")
                .field("id", id)
                .field("name", name)
                .field("summary", summary)
                .finish(),
            Self::ToolCallFailed { id, name, error } => f
                .debug_struct("ToolCallFailed")
                .field("id", id)
                .field("name", name)
                .field("error", error)
                .finish(),
            Self::Custom(_) => f.debug_tuple("Custom").field(&"<opaque payload>").finish(),
        }
    }
}

pub struct Channel {
    pub uri: ChannelUri,
    pub name: String,
}

impl Channel {
    pub fn display_name(&self) -> &str {
        if self.name.is_empty() {
            self.uri.as_str()
        } else {
            self.name.as_str()
        }
    }
}

pub struct Sender {
    pub uri: SenderUri,
    pub name: String,
}

impl Sender {
    pub fn display_name(&self) -> &str {
        if self.name.is_empty() {
            self.uri.as_str()
        } else {
            self.name.as_str()
        }
    }
}

#[async_trait]
pub trait Agent: Send + Sync {
    async fn send(
        &self,
        channel: &Channel,
        sender: &Sender,
        timestamp: DateTime<Utc>,
        input: &str,
        response_uri: &MessageUri,
    ) -> Result<AgentStream, LlmError>;

    /// Called by the participant runtime for each historical message
    /// reconstructed during catch-up. Default: ignore. `is_self` is true when
    /// the replayed message was originally emitted by this agent.
    fn on_replay_message(
        &self,
        _channel: &Channel,
        _sender: &Sender,
        _is_self: bool,
        _timestamp: DateTime<Utc>,
        _content: &str,
    ) {
    }

    /// Called by the application runner once per historical [`Event`]
    /// (regardless of payload type) so the agent can rebuild its own state
    /// from any custom extension events it previously emitted. Default:
    /// ignore.
    fn on_replay_event(&self, _event: &Event) {}
}
