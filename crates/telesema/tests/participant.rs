use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use futures_util::{StreamExt, stream};
use pyg::{Agent, AgentEvent, AgentStream, Channel, LlmError, Sender};
use pyg_telesema::AgentParticipantBuilder;
use telesema_core::prelude::*;
use telesema_core::MessageUri;

struct ScriptedAgent;

#[async_trait]
impl Agent for ScriptedAgent {
    async fn send(
        &self,
        _channel: &Channel,
        _sender: &Sender,
        _timestamp: DateTime<Utc>,
        _input: &str,
        _response_uri: &MessageUri,
    ) -> Result<AgentStream, LlmError> {
        Ok(Box::pin(stream::iter(vec![
            Ok(AgentEvent::ToolCallStarted {
                id: "call_1".to_string(),
                name: "now".to_string(),
            }),
            Ok(AgentEvent::ToolCallFinished {
                id: "call_1".to_string(),
                name: "now".to_string(),
                summary: "2026-05-28T00:00:00Z".to_string(),
            }),
            Ok(AgentEvent::TextDelta("done".to_string())),
        ])))
    }
}

async fn collect_message_chunks(
    rx: &mut futures_channel::mpsc::UnboundedReceiver<Event>,
    n: usize,
) -> Vec<(String, String)> {
    let mut chunks = Vec::new();
    while chunks.len() < n {
        let event = rx.next().await.expect("event emitted");
        if let Some(CoreEvent::MessageStreamChunk(chunk)) = event.payload::<CoreEvent>() {
            chunks.push((
                chunk.message_header.message_uri.to_string(),
                chunk.content.clone(),
            ));
        }
    }
    chunks
}

#[tokio::test]
async fn participant_emits_tool_events_with_participant_id_as_sender_uri() {
    let runtime =
        AgentParticipantBuilder::new(ParticipantId::new("scribe"), ScriptedAgent)
            .spawn()
            .await
            .expect("spawn succeeds");
    let mut rx = runtime.rx;
    let tx = runtime.tx;

    tx.unbounded_send(Event::new(
        ParticipantId::new("main"),
        CoreEvent::ChannelDiscovered(ChannelDiscovered {
            channel_uri: ChannelUri::new("room:1"),
            display_name: "Room".to_string(),
            metadata: Metadata::new(),
        }),
    ))
    .unwrap();
    tx.unbounded_send(Event::new(
        ParticipantId::new("main"),
        CoreEvent::SenderDiscovered(SenderDiscovered {
            sender_uri: SenderUri::new("cli:steve"),
            display_name: "Steve".to_string(),
            metadata: Metadata::new(),
        }),
    ))
    .unwrap();

    let header = MessageHeader {
        channel_uri: ChannelUri::new("room:1"),
        sender_uri: SenderUri::new("cli:steve"),
        date_time: Utc.with_ymd_and_hms(2026, 5, 28, 22, 44, 0).unwrap(),
        message_uri: MessageUri::new("cli:1"),
    };
    tx.unbounded_send(Event::new(
        ParticipantId::new("cli"),
        CoreEvent::MessageStreamChunk(MessageStreamChunk {
            message_header: header.clone(),
            content: "what time is it?".to_string(),
            metadata: None,
        }),
    ))
    .unwrap();
    tx.unbounded_send(Event::new(
        ParticipantId::new("cli"),
        CoreEvent::MessageStreamCompleted(MessageStreamCompleted {
            message_header: header,
            metadata: None,
        }),
    ))
    .unwrap();

    let chunks = collect_message_chunks(&mut rx, 3).await;
    assert_eq!(chunks[0].1, "[tool] now");
    assert!(chunks[0].0.ends_with(":tool:1"));
    assert!(chunks[0].0.starts_with("scribe:"));
    assert_eq!(chunks[1].1, "[tool] now returned 2026-05-28T00:00:00Z");
    assert_eq!(chunks[2].1, "done");
}
