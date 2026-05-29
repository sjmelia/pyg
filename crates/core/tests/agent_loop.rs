use async_trait::async_trait;
use futures_channel::mpsc;
use futures_util::{StreamExt, stream};
use pyg::{
    AgentEvent, AssistantOutputSink, Llm, LlmError, LlmEvent, LlmStream, Message, Tool,
    ToolDefinition, TurnSummary, run_agent_loop,
};
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct ScriptedLlm {
    sent_messages: Arc<Mutex<Vec<Vec<Message>>>>,
    responses: Arc<Mutex<Vec<Vec<LlmEvent>>>>,
}

impl ScriptedLlm {
    fn enqueue(&self, events: Vec<LlmEvent>) {
        self.responses.lock().unwrap().push(events);
    }
}

#[async_trait]
impl Llm for ScriptedLlm {
    async fn send(
        &self,
        messages: Vec<Message>,
        _tools: &[ToolDefinition],
    ) -> Result<LlmStream, LlmError> {
        self.sent_messages.lock().unwrap().push(messages);
        let events = self.responses.lock().unwrap().remove(0);
        Ok(Box::pin(stream::iter(events.into_iter().map(Ok))))
    }
}

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "echo".to_string(),
            description: "Echoes its `value` argument back.".to_string(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            }),
        }
    }

    async fn call(&self, arguments: serde_json::Value) -> Result<String, LlmError> {
        Ok(arguments["value"].as_str().unwrap_or("").to_string())
    }
}

struct CollectingSink {
    out: Arc<Mutex<String>>,
}

impl AssistantOutputSink for CollectingSink {
    fn handle_turn(
        &self,
        summary: TurnSummary,
        tx: &mut mpsc::UnboundedSender<Result<AgentEvent, LlmError>>,
    ) {
        *self.out.lock().unwrap() = summary.final_assistant_text.clone();
        let _ = tx.unbounded_send(Ok(AgentEvent::TextDelta(summary.final_assistant_text)));
    }
}

async fn drain(mut rx: mpsc::UnboundedReceiver<Result<AgentEvent, LlmError>>) -> Vec<AgentEvent> {
    let mut events = Vec::new();
    while let Some(event) = rx.next().await {
        events.push(event.expect("agent stream event succeeds"));
    }
    events
}

#[tokio::test]
async fn loop_dispatches_tool_calls_and_feeds_results_back_into_next_turn() {
    let llm = ScriptedLlm::default();
    // First turn: model asks to call echo.
    llm.enqueue(vec![
        LlmEvent::ToolCallDelta {
            index: 0,
            id: Some("call_1".to_string()),
            name: Some("echo".to_string()),
            arguments_delta: "{\"value\":\"hi\"}".to_string(),
        },
    ]);
    // Second turn: model produces final text after seeing the tool result.
    llm.enqueue(vec![LlmEvent::TextDelta("done: hi".to_string())]);

    let sink_out = Arc::new(Mutex::new(String::new()));
    let sink = Box::new(CollectingSink {
        out: sink_out.clone(),
    });

    let (tx, rx) = mpsc::unbounded();
    run_agent_loop(
        Arc::new(llm.clone()),
        vec![Arc::new(EchoTool)],
        sink,
        vec![Message::user("say hi")],
        tx,
    )
    .await;

    let events = drain(rx).await;
    assert!(matches!(events[0], AgentEvent::ToolCallStarted { .. }));
    assert!(matches!(events[1], AgentEvent::ToolCallReady { .. }));
    assert!(matches!(events[2], AgentEvent::ToolCallFinished { .. }));
    assert!(matches!(events[3], AgentEvent::TextDelta(ref t) if t == "done: hi"));
    assert_eq!(*sink_out.lock().unwrap(), "done: hi");

    let sent = llm.sent_messages.lock().unwrap();
    assert_eq!(sent.len(), 2);
    // Second turn must include the assistant tool-call message and the tool result.
    let second = &sent[1];
    assert!(matches!(second[1], Message::Assistant { .. }));
    assert!(matches!(second[2], Message::ToolResult { ref content, .. } if content == "hi"));
}

#[tokio::test]
async fn loop_fails_fast_when_tool_call_delta_missing_name_on_first_chunk() {
    let llm = ScriptedLlm::default();
    llm.enqueue(vec![LlmEvent::ToolCallDelta {
        index: 0,
        id: Some("call_1".to_string()),
        name: None,
        arguments_delta: String::new(),
    }]);

    let sink = Box::new(CollectingSink {
        out: Arc::new(Mutex::new(String::new())),
    });

    let (tx, mut rx) = mpsc::unbounded();
    run_agent_loop(
        Arc::new(llm),
        vec![Arc::new(EchoTool)],
        sink,
        vec![Message::user("hi")],
        tx,
    )
    .await;

    let first = rx.next().await.expect("error event");
    let error = first.expect_err("expected error");
    assert!(error.to_string().contains("missing id or name"));
}
