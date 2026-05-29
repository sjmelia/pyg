use anyhow::Result;
use chrono::Utc;
use futures_channel::mpsc;
use futures_util::SinkExt;
use futures_util::StreamExt;
use futures_util::future;
use std::collections::HashMap;
use telesema_core::CompleteMessage;
use telesema_core::prelude::*;

use pyg::{Agent, AgentEvent, Channel, Sender};

struct AgentState<A: Agent> {
    participant_id: ParticipantId,
    agent: A,
    messages: MessagesProjection,
    senders: HashMap<SenderUri, String>,
    channels: HashMap<ChannelUri, String>,
}

impl<A: Agent> AgentState<A> {
    fn new(participant_id: ParticipantId, agent: A) -> Self {
        Self {
            participant_id,
            agent,
            messages: MessagesProjection::new(),
            senders: HashMap::new(),
            channels: HashMap::new(),
        }
    }

    fn channel_for(&self, channel_uri: &ChannelUri) -> Channel {
        Channel {
            uri: channel_uri.clone(),
            name: self
                .channels
                .get(channel_uri)
                .cloned()
                .unwrap_or_else(|| channel_uri.to_string()),
        }
    }

    fn sender_for(&self, sender_uri: &SenderUri) -> Sender {
        Sender {
            uri: sender_uri.clone(),
            name: self
                .senders
                .get(sender_uri)
                .cloned()
                .unwrap_or_else(|| sender_uri.to_string()),
        }
    }

    /// Updates internal state from any [`CoreEvent`] and, if the event
    /// completed a message stream, returns the assembled message.
    fn project(&mut self, event: &CoreEvent) -> Option<CompleteMessage> {
        match event {
            CoreEvent::MessageStreamChunk(chunk) => {
                self.messages.apply(chunk.clone());
                None
            }
            CoreEvent::MessageStreamCompleted(completed) => self.messages.apply(completed.clone()),
            CoreEvent::SenderDiscovered(d) => {
                self.senders
                    .insert(d.sender_uri.clone(), d.display_name.clone());
                None
            }
            CoreEvent::ChannelDiscovered(d) => {
                self.channels
                    .insert(d.channel_uri.clone(), d.display_name.clone());
                None
            }
            _ => None,
        }
    }
}

struct EventEmitter<'a> {
    tx: &'a mut mpsc::UnboundedSender<Event>,
    participant_id: ParticipantId,
}

impl EventEmitter<'_> {
    async fn send(&mut self, event: CoreEvent) -> Result<()> {
        self.tx
            .send(Event::new(self.participant_id.clone(), event))
            .await?;
        Ok(())
    }

    async fn send_event(&mut self, event: Event) -> Result<()> {
        self.tx.send(event).await?;
        Ok(())
    }

    async fn send_chunk(&mut self, header: MessageHeader, content: String) -> Result<()> {
        self.send(CoreEvent::MessageStreamChunk(MessageStreamChunk {
            message_header: header,
            content,
            metadata: None,
        }))
        .await
    }

    async fn send_completed(&mut self, header: MessageHeader) -> Result<()> {
        self.send(CoreEvent::MessageStreamCompleted(MessageStreamCompleted {
            message_header: header,
            metadata: None,
        }))
        .await
    }

    async fn send_failed(&mut self, header: MessageHeader) -> Result<()> {
        self.send(CoreEvent::MessageDeliveryFailed(MessageDeliveryFailed {
            message_header: header,
        }))
        .await
    }

    async fn send_tool_chunk(
        &mut self,
        response_header: &MessageHeader,
        counter: u64,
        content: String,
    ) -> Result<()> {
        let header = tool_header(response_header, counter);
        self.send_chunk(header.clone(), content).await?;
        self.send_completed(header).await
    }
}

fn tool_header(response_header: &MessageHeader, counter: u64) -> MessageHeader {
    MessageHeader {
        channel_uri: response_header.channel_uri.clone(),
        sender_uri: response_header.sender_uri.clone(),
        date_time: Utc::now(),
        message_uri: MessageUri::new(format!(
            "{}:tool:{}",
            response_header.message_uri.as_str(),
            counter
        )),
    }
}

fn format_tool_event(event: &AgentEvent) -> Option<String> {
    match event {
        AgentEvent::TextDelta(_) | AgentEvent::Custom(_) => None,
        AgentEvent::ToolCallStarted { name, .. } => Some(format!("[tool] {name}")),
        AgentEvent::ToolCallReady {
            name, arguments, ..
        } => Some(format!("[tool] running {name} with {arguments}")),
        AgentEvent::ToolCallFinished { name, summary, .. } => {
            Some(format!("[tool] {name} returned {summary}"))
        }
        AgentEvent::ToolCallFailed { name, error, .. } => {
            Some(format!("[tool] {name} failed: {error}"))
        }
    }
}

async fn handle_inbound_message<A: Agent>(
    state: &AgentState<A>,
    message: CompleteMessage,
    emitter: &mut EventEmitter<'_>,
) -> Result<()> {
    let sender = state.sender_for(&message.message_header.sender_uri);
    let channel = state.channel_for(&message.message_header.channel_uri);
    let response_header = MessageHeader {
        channel_uri: message.message_header.channel_uri.clone(),
        sender_uri: SenderUri::new(state.participant_id.as_str()),
        date_time: Utc::now(),
        message_uri: MessageUri::new(format!(
            "{}:{}",
            state.participant_id.as_str(),
            message.message_header.message_uri.as_str()
        )),
    };

    let response = state
        .agent
        .send(
            &channel,
            &sender,
            message.message_header.date_time,
            &message.content,
            &response_header.message_uri,
        )
        .await;

    let mut stream = match response {
        Ok(stream) => stream,
        Err(_) => return emitter.send_failed(response_header).await,
    };

    let mut tool_counter = 0_u64;
    while let Some(result) = stream.next().await {
        let event = match result {
            Ok(event) => event,
            Err(_) => return emitter.send_failed(response_header).await,
        };

        match event {
            AgentEvent::TextDelta(content) => {
                emitter.send_chunk(response_header.clone(), content).await?;
            }
            AgentEvent::Custom(payload) => {
                emitter
                    .send_event(Event::from_payload_arc(
                        emitter.participant_id.clone(),
                        payload,
                    ))
                    .await?;
            }
            _ => {
                if let Some(content) = format_tool_event(&event) {
                    tool_counter += 1;
                    emitter
                        .send_tool_chunk(&response_header, tool_counter, content)
                        .await?;
                }
            }
        }
    }

    emitter.send_completed(response_header).await
}

struct AgentParticipant<A: Agent> {
    state: AgentState<A>,
}

impl<A: Agent> AgentParticipant<A> {
    async fn run(
        &mut self,
        rx: mpsc::UnboundedReceiver<Event>,
        mut tx: mpsc::UnboundedSender<Event>,
    ) -> Result<()> {
        let mut mapped_events = rx.filter_map(|ev| {
            let origin = ev.origin.clone();
            let core = ev.payload::<CoreEvent>().cloned();
            future::ready(core.map(|c| (origin, c)))
        });

        while let Some((origin, core_event)) = mapped_events.next().await {
            let completed = self.state.project(&core_event);
            let Some(message) = completed else {
                continue;
            };
            if origin == self.state.participant_id {
                continue;
            }
            let mut emitter = EventEmitter {
                tx: &mut tx,
                participant_id: self.state.participant_id.clone(),
            };
            handle_inbound_message(&self.state, message, &mut emitter).await?;
        }

        Ok(())
    }
}

pub struct AgentParticipantBuilder<A: Agent> {
    state: AgentState<A>,
}

impl<A: Agent + 'static> AgentParticipantBuilder<A> {
    pub fn new(participant_id: ParticipantId, agent: A) -> Self {
        Self {
            state: AgentState::new(participant_id, agent),
        }
    }

    pub async fn catch_up(
        mut self,
        replay: &mut (dyn futures_util::stream::Stream<Item = Event> + Send + Unpin),
    ) -> Result<Self> {
        while let Some(event) = replay.next().await {
            let Some(core_event) = event.payload::<CoreEvent>() else {
                continue;
            };
            let completed = self.state.project(core_event);
            let Some(message) = completed else {
                continue;
            };
            let sender = self.state.sender_for(&message.message_header.sender_uri);
            let channel = self.state.channel_for(&message.message_header.channel_uri);
            let is_self = event.origin == self.state.participant_id;
            self.state.agent.on_replay_message(
                &channel,
                &sender,
                is_self,
                message.message_header.date_time,
                &message.content,
            );
        }
        Ok(self)
    }

    pub async fn spawn(self) -> Result<ParticipantRuntime> {
        let mut participant = AgentParticipant { state: self.state };
        let (tx_in, rx_in) = mpsc::unbounded();
        let (tx_out, rx_out) = mpsc::unbounded();

        let task = tokio::spawn(async move { participant.run(rx_in, tx_out).await });
        let join: ParticipantJoin = Box::pin(async move {
            task.await
                .map_err(|err| anyhow::anyhow!("agent participant task failed: {err}"))?
        });

        Ok(ParticipantRuntime {
            rx: rx_out,
            tx: tx_in,
            join,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_events_format_with_chat_prefix() {
        assert_eq!(format_tool_event(&AgentEvent::TextDelta("x".into())), None);
        assert_eq!(
            format_tool_event(&AgentEvent::ToolCallStarted {
                id: "1".into(),
                name: "now".into(),
            })
            .unwrap(),
            "[tool] now"
        );
        assert_eq!(
            format_tool_event(&AgentEvent::ToolCallReady {
                id: "1".into(),
                name: "now".into(),
                arguments: "{}".into(),
            })
            .unwrap(),
            "[tool] running now with {}"
        );
    }
}
