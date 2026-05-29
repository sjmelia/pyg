use chrono::{DateTime, Utc};
use futures_channel::mpsc;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, trace};

use crate::{AgentEvent, Llm, LlmError, LlmEvent, Message, Tool, ToolCall};

pub const MAX_TOOL_ITERATIONS: usize = 4;

const TOOL_SUMMARY_MAX_LEN: usize = 120;

pub trait AssistantOutputSink: Send + Sync {
    /// Called once per agent turn after the LLM produces a final assistant
    /// message (i.e. no more tool calls). Receives the full turn record so
    /// the sink can both decide what to emit on the chat stream AND surface
    /// the trace as an extension event.
    fn handle_turn(
        &self,
        summary: TurnSummary,
        tx: &mut mpsc::UnboundedSender<Result<AgentEvent, LlmError>>,
    );
}

/// Complete trace of a single agent turn, handed to [`AssistantOutputSink`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnSummary {
    pub final_assistant_text: String,
    pub messages_to_llm: Vec<Message>,
    pub tool_calls: Vec<RecordedToolCall>,
    pub started_at: DateTime<Utc>,
    pub ended_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
    pub result: ToolCallResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum ToolCallResult {
    Ok { content: String },
    Err { error: String },
}

pub async fn run_agent_loop(
    llm: Arc<dyn Llm>,
    tools: Vec<Arc<dyn Tool>>,
    sink: Box<dyn AssistantOutputSink>,
    mut messages: Vec<Message>,
    mut tx: mpsc::UnboundedSender<Result<AgentEvent, LlmError>>,
) {
    let tool_definitions = tools.iter().map(|t| t.definition()).collect::<Vec<_>>();
    let started_at = Utc::now();
    let mut recorded_tool_calls: Vec<RecordedToolCall> = Vec::new();

    for _ in 0..MAX_TOOL_ITERATIONS {
        let mut stream = match llm.send(messages.clone(), &tool_definitions).await {
            Ok(stream) => stream,
            Err(error) => {
                let _ = tx.unbounded_send(Err(error));
                return;
            }
        };

        let mut partials: HashMap<usize, PartialToolCall> = HashMap::new();
        let mut assistant_text = String::new();
        while let Some(event) = stream.next().await {
            match event {
                Ok(LlmEvent::TextDelta(content)) => assistant_text.push_str(&content),
                Ok(LlmEvent::ToolCallDelta {
                    index,
                    id,
                    name,
                    arguments_delta,
                }) => {
                    trace!(
                        index,
                        ?id,
                        ?name,
                        arguments_delta,
                        "received llm tool-call delta"
                    );
                    if !apply_tool_call_delta(
                        &mut partials,
                        index,
                        id,
                        name,
                        &arguments_delta,
                        &mut tx,
                    ) {
                        return;
                    }
                }
                Err(error) => {
                    let _ = tx.unbounded_send(Err(error));
                    return;
                }
            }
        }

        let completed = match completed_tool_calls(partials) {
            Ok(completed) => completed,
            Err(error) => {
                let _ = tx.unbounded_send(Err(error));
                return;
            }
        };

        if completed.is_empty() {
            debug!(text = %assistant_text, "agent loop received final assistant text");
            let summary = TurnSummary {
                final_assistant_text: assistant_text,
                messages_to_llm: messages,
                tool_calls: recorded_tool_calls,
                started_at,
                ended_at: Utc::now(),
            };
            sink.handle_turn(summary, &mut tx);
            return;
        }

        messages.push(Message::Assistant {
            text: assistant_text,
            tool_calls: completed.iter().map(CompletedToolCall::to_tool_call).collect(),
        });

        for call in completed {
            match dispatch_tool(&tools, call, &mut messages, &mut tx).await {
                DispatchOutcome::Recorded(record) => recorded_tool_calls.push(record),
                DispatchOutcome::Shutdown => return,
            }
        }
    }

    let _ = tx.unbounded_send(Err("tool loop exceeded maximum iterations".into()));
}

#[derive(Default)]
struct PartialToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
    started: bool,
}

struct CompletedToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl CompletedToolCall {
    fn to_tool_call(&self) -> ToolCall {
        ToolCall {
            id: self.id.clone(),
            name: self.name.clone(),
            arguments: self.arguments.clone(),
        }
    }
}

fn apply_tool_call_delta(
    partials: &mut HashMap<usize, PartialToolCall>,
    index: usize,
    id: Option<String>,
    name: Option<String>,
    arguments_delta: &str,
    tx: &mut mpsc::UnboundedSender<Result<AgentEvent, LlmError>>,
) -> bool {
    let partial = partials.entry(index).or_default();
    if let Some(id) = id {
        partial.id = Some(id);
    }
    if let Some(name) = name {
        partial.name = Some(name);
    }
    partial.arguments.push_str(arguments_delta);

    if partial.started {
        return true;
    }
    partial.started = true;

    let (Some(id), Some(name)) = (partial.id.clone(), partial.name.clone()) else {
        let _ = tx.unbounded_send(Err(format!(
            "tool call delta at index {index} missing id or name on first chunk"
        )
        .into()));
        return false;
    };
    send_event(tx, AgentEvent::ToolCallStarted { id, name })
}

fn completed_tool_calls(
    partials: HashMap<usize, PartialToolCall>,
) -> Result<Vec<CompletedToolCall>, LlmError> {
    let mut calls = partials
        .into_iter()
        .map(|(index, partial)| {
            let (Some(id), Some(name)) = (partial.id, partial.name) else {
                return Err(format!("tool call at index {index} missing id or name").into());
            };
            Ok(CompletedToolCall {
                id,
                name,
                arguments: partial.arguments,
            })
        })
        .collect::<Result<Vec<_>, LlmError>>()?;
    calls.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(calls)
}

enum DispatchOutcome {
    Recorded(RecordedToolCall),
    Shutdown,
}

async fn dispatch_tool(
    tools: &[Arc<dyn Tool>],
    call: CompletedToolCall,
    messages: &mut Vec<Message>,
    tx: &mut mpsc::UnboundedSender<Result<AgentEvent, LlmError>>,
) -> DispatchOutcome {
    if !send_event(
        tx,
        AgentEvent::ToolCallReady {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        },
    ) {
        return DispatchOutcome::Shutdown;
    }

    let result = match tools.iter().find(|t| t.definition().name == call.name) {
        Some(tool) => match serde_json::from_str(&call.arguments) {
            Ok(args) => tool.call(args).await,
            Err(error) => Err(format!("invalid tool arguments JSON: {error}").into()),
        },
        None => Err(format!("unknown tool: {}", call.name).into()),
    };

    let (content, recorded_result) = match result {
        Ok(result) => {
            if !send_event(
                tx,
                AgentEvent::ToolCallFinished {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    summary: tool_summary(&result),
                },
            ) {
                return DispatchOutcome::Shutdown;
            }
            let recorded = ToolCallResult::Ok {
                content: result.clone(),
            };
            (result, recorded)
        }
        Err(error) => {
            let text = error.to_string();
            if !send_event(
                tx,
                AgentEvent::ToolCallFailed {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    error: text.clone(),
                },
            ) {
                return DispatchOutcome::Shutdown;
            }
            let recorded = ToolCallResult::Err {
                error: text.clone(),
            };
            (text, recorded)
        }
    };

    messages.push(Message::ToolResult {
        tool_call_id: call.id.clone(),
        content,
    });

    DispatchOutcome::Recorded(RecordedToolCall {
        id: call.id,
        name: call.name,
        arguments: call.arguments,
        result: recorded_result,
    })
}

fn send_event(
    tx: &mut mpsc::UnboundedSender<Result<AgentEvent, LlmError>>,
    event: AgentEvent,
) -> bool {
    tx.unbounded_send(Ok(event)).is_ok()
}

fn tool_summary(result: &str) -> String {
    let mut summary = result.replace('\n', " ");
    if summary.len() > TOOL_SUMMARY_MAX_LEN {
        summary.truncate(TOOL_SUMMARY_MAX_LEN);
        summary.push_str("...");
    }
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_summary_truncates_long_text_and_collapses_newlines() {
        assert_eq!(tool_summary("a\nb\nc"), "a b c");
        let long = "x".repeat(TOOL_SUMMARY_MAX_LEN + 50);
        let summary = tool_summary(&long);
        assert_eq!(summary.len(), TOOL_SUMMARY_MAX_LEN + 3);
        assert!(summary.ends_with("..."));
    }
}
