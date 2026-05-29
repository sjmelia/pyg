use anyhow::Result;
use futures_util::stream;
use pyg::{Agent, Channel, Sender};
use std::collections::HashSet;
use std::sync::Arc;
use telesema_core::prelude::*;

use crate::AgentParticipantBuilder;

/// Generic orchestrator for "wire an [`Agent`] to one or more chat transports,
/// replay history, log everything, route with optional filters."
///
/// `AgentApp` takes care of:
///
/// - Calling [`Agent::on_replay_event`] for every historical [`Event`] so the
///   agent can rebuild its own state from past extension events.
/// - Feeding the same history through pyg-telesema's
///   [`AgentParticipantBuilder::catch_up`] so the message projection is
///   restored.
/// - Optionally sending bootstrap discovery events (`SenderDiscovered`,
///   `ChannelDiscovered`) into every participant — useful for transports
///   like the CLI/HTML chat where the world's identities aren't otherwise
///   discovered.
/// - Wiring all participants through a [`BridgeBuilder`] with optional
///   "silenced" and "ignored" channel filters on the agent ↔ transport
///   edges.
pub struct AgentApp<A: Agent> {
    agent_participant_id: ParticipantId,
    agent: A,
    transports: Vec<(String, ParticipantRuntime)>,
    history: Vec<Event>,
    bootstrap: Option<BootstrapIdentities>,
    observers: Vec<(String, ParticipantRuntime)>,
    silenced_channels: HashSet<ChannelUri>,
    ignored_channels: HashSet<ChannelUri>,
}

struct BootstrapIdentities {
    channel: Channel,
    transport_sender: Sender,
    agent_sender: Sender,
}

impl<A: Agent + 'static> AgentApp<A> {
    pub fn new(
        agent_participant_id: ParticipantId,
        agent: A,
        transport: ParticipantRuntime,
    ) -> Self {
        Self {
            agent_participant_id,
            agent,
            transports: vec![("transport".to_string(), transport)],
            history: Vec::new(),
            bootstrap: None,
            observers: Vec::new(),
            silenced_channels: HashSet::new(),
            ignored_channels: HashSet::new(),
        }
    }

    /// Attach an additional transport runtime alongside the primary one
    /// passed to `new`. Silenced/ignored filters apply to **every**
    /// agent→transport edge, so messages for silenced channels are blocked
    /// from reaching any transport.
    pub fn with_transport(
        mut self,
        name: impl Into<String>,
        transport: ParticipantRuntime,
    ) -> Self {
        self.transports.push((name.into(), transport));
        self
    }

    pub fn with_history(mut self, history: Vec<Event>) -> Self {
        self.history = history;
        self
    }

    /// Emit synthetic `ChannelDiscovered` / `SenderDiscovered` events to all
    /// participants at startup. Useful for transports like CLI and HTML chat
    /// where the channel and sender names are otherwise not discoverable.
    /// Transport-native discovery (e.g. WhatsApp) doesn't need this.
    pub fn with_bootstrap(
        mut self,
        channel: Channel,
        transport_sender: Sender,
        agent_sender: Sender,
    ) -> Self {
        self.bootstrap = Some(BootstrapIdentities {
            channel,
            transport_sender,
            agent_sender,
        });
        self
    }

    /// Attach a passive observer runtime — typically an append-only event
    /// log or a renderer for some extension-event class — that should
    /// receive every event flowing through the bridge. Can be called
    /// multiple times; names just disambiguate the bridge's routing
    /// rules and aren't otherwise meaningful.
    pub fn with_observer(mut self, name: impl Into<String>, observer: ParticipantRuntime) -> Self {
        self.observers.push((name.into(), observer));
        self
    }

    /// Channels in this set are still processed by the agent and persisted
    /// by the logger, but the agent's chat replies for them are dropped
    /// before reaching the transport. Useful for "dress rehearsal" modes.
    pub fn with_silenced_channels(
        mut self,
        channels: impl IntoIterator<Item = ChannelUri>,
    ) -> Self {
        self.silenced_channels = channels.into_iter().collect();
        self
    }

    /// Defense-in-depth filter for channels the agent should completely
    /// ignore. Any chat message event for one of these channels is dropped
    /// on **both** the transport→agent and the agent→transport bridge edges,
    /// so even if the agent's own ignore-handling slips, nothing leaks
    /// either way. The agent itself is also expected to short-circuit on
    /// these channels — this is the safety net.
    pub fn with_ignored_channels(
        mut self,
        channels: impl IntoIterator<Item = ChannelUri>,
    ) -> Self {
        self.ignored_channels = channels.into_iter().collect();
        self
    }

    pub async fn run(self) -> Result<()> {
        // Let the agent restore any custom state from its past extension
        // events before it starts taking live traffic.
        for event in &self.history {
            self.agent.on_replay_event(event);
        }

        let agent_runtime =
            AgentParticipantBuilder::new(self.agent_participant_id.clone(), self.agent)
                .catch_up(&mut stream::iter(self.history.into_iter()))
                .await?
                .spawn()
                .await?;

        if let Some(bootstrap) = &self.bootstrap {
            let events = [
                discover_channel(&bootstrap.channel),
                discover_sender(&bootstrap.transport_sender),
                discover_sender(&bootstrap.agent_sender),
            ];
            for event in &events {
                for (_, transport) in &self.transports {
                    transport.tx.unbounded_send(event.clone())?;
                }
                agent_runtime.tx.unbounded_send(event.clone())?;
                for (_, observer) in &self.observers {
                    observer.tx.unbounded_send(event.clone())?;
                }
            }
        }

        let silenced = Arc::new(self.silenced_channels);
        let ignored = Arc::new(self.ignored_channels);

        let mut bridge = BridgeBuilder::new().add("agent", agent_runtime);
        for (name, transport) in self.transports {
            let silenced = silenced.clone();
            let ignored_out = ignored.clone();
            let ignored_in = ignored.clone();
            let name_out = name.clone();
            let name_in = name.clone();
            bridge = bridge
                .add(name, transport)
                .filter("agent", name_out, move |event| {
                    !is_chat_event_in_set(event, &silenced)
                        && !is_chat_event_in_set(event, &ignored_out)
                })
                .filter(name_in, "agent", move |event| {
                    !is_chat_event_in_set(event, &ignored_in)
                });
        }
        for (name, observer) in self.observers {
            bridge = bridge.add(name, observer);
        }
        bridge.run().await
    }
}

fn discover_channel(channel: &Channel) -> Event {
    Event::new(
        ParticipantId::new("main"),
        CoreEvent::ChannelDiscovered(ChannelDiscovered {
            channel_uri: channel.uri.clone(),
            display_name: channel.display_name().to_string(),
            metadata: Metadata::new(),
        }),
    )
}

fn discover_sender(sender: &Sender) -> Event {
    Event::new(
        ParticipantId::new("main"),
        CoreEvent::SenderDiscovered(SenderDiscovered {
            sender_uri: sender.uri.clone(),
            display_name: sender.display_name().to_string(),
            metadata: Metadata::new(),
        }),
    )
}

fn is_chat_event_in_set(event: &Event, set: &HashSet<ChannelUri>) -> bool {
    let Some(core) = event.payload::<CoreEvent>() else {
        return false;
    };
    match core {
        CoreEvent::MessageStreamChunk(c) => set.contains(&c.message_header.channel_uri),
        CoreEvent::MessageStreamCompleted(c) => set.contains(&c.message_header.channel_uri),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_event_filter_matches_only_message_streams_in_listed_channels() {
        let set: HashSet<ChannelUri> = [ChannelUri::new("silent:room")].into_iter().collect();

        let chunk_match = Event::new(
            ParticipantId::new("agent"),
            CoreEvent::MessageStreamChunk(MessageStreamChunk {
                message_header: MessageHeader {
                    channel_uri: ChannelUri::new("silent:room"),
                    sender_uri: SenderUri::new("agent"),
                    date_time: chrono::Utc::now(),
                    message_uri: MessageUri::new("x:1"),
                },
                content: "hi".into(),
                metadata: None,
            }),
        );
        let chunk_other = Event::new(
            ParticipantId::new("agent"),
            CoreEvent::MessageStreamChunk(MessageStreamChunk {
                message_header: MessageHeader {
                    channel_uri: ChannelUri::new("loud:room"),
                    sender_uri: SenderUri::new("agent"),
                    date_time: chrono::Utc::now(),
                    message_uri: MessageUri::new("x:2"),
                },
                content: "hi".into(),
                metadata: None,
            }),
        );
        let discovery = Event::new(
            ParticipantId::new("agent"),
            CoreEvent::SenderDiscovered(SenderDiscovered {
                sender_uri: SenderUri::new("agent"),
                display_name: "Agent".into(),
                metadata: Metadata::new(),
            }),
        );

        assert!(is_chat_event_in_set(&chunk_match, &set));
        assert!(!is_chat_event_in_set(&chunk_other, &set));
        assert!(!is_chat_event_in_set(&discovery, &set));
    }
}
