use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestDeveloperMessageArgs,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs,
    ChatCompletionTool, ChatCompletionToolChoiceOption, ChatCompletionTools,
    CreateChatCompletionRequestArgs, CreateChatCompletionStreamResponse, FunctionCall,
    FunctionObject, ToolChoiceOptions,
};
use async_trait::async_trait;
use futures_util::{StreamExt, stream};

use crate::{Llm, LlmError, LlmEvent, LlmStream, Message, ToolDefinition};

pub struct OpenAiApiLlm {
    client: Client<OpenAIConfig>,
    model: String,
}

impl OpenAiApiLlm {
    pub fn new(api_base: &str, api_key: &str, model: &str) -> Self {
        let config = OpenAIConfig::new()
            .with_api_base(api_base)
            .with_api_key(api_key);
        Self {
            client: Client::with_config(config),
            model: model.to_string(),
        }
    }
}

#[async_trait]
impl Llm for OpenAiApiLlm {
    async fn send(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
    ) -> Result<LlmStream, LlmError> {
        let openai_messages = messages
            .into_iter()
            .map(to_openai_message)
            .collect::<Result<Vec<_>, _>>()?;

        let mut request = CreateChatCompletionRequestArgs::default();
        request.model(&self.model).messages(openai_messages);

        if !tools.is_empty() {
            request.tools(tools.iter().cloned().map(to_openai_tool).collect::<Vec<_>>());
            request.tool_choice(ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::Auto));
        }

        let stream = self.client.chat().create_stream(request.build()?).await?;
        let mapped = stream.flat_map(|result| {
            let events = match result {
                Ok(chunk) => extract_events(chunk),
                Err(err) => vec![Err(err.into())],
            };
            stream::iter(events)
        });

        Ok(Box::pin(mapped))
    }
}

fn to_openai_message(message: Message) -> Result<ChatCompletionRequestMessage, LlmError> {
    Ok(match message {
        Message::System { content } => ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessageArgs::default()
                .content(content)
                .build()?,
        ),
        Message::Developer { content } => ChatCompletionRequestMessage::Developer(
            ChatCompletionRequestDeveloperMessageArgs::default()
                .content(content)
                .build()?,
        ),
        Message::User { content } => ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content(content)
                .build()?,
        ),
        Message::Assistant { text, tool_calls } => {
            let mut args = ChatCompletionRequestAssistantMessageArgs::default();
            if !text.is_empty() {
                args.content(text);
            }
            if !tool_calls.is_empty() {
                args.tool_calls(
                    tool_calls
                        .into_iter()
                        .map(|call| {
                            ChatCompletionMessageToolCalls::Function(
                                ChatCompletionMessageToolCall {
                                    id: call.id,
                                    function: FunctionCall {
                                        name: call.name,
                                        arguments: call.arguments,
                                    },
                                },
                            )
                        })
                        .collect::<Vec<_>>(),
                );
            }
            ChatCompletionRequestMessage::Assistant(args.build()?)
        }
        Message::ToolResult {
            tool_call_id,
            content,
        } => ChatCompletionRequestMessage::Tool(
            ChatCompletionRequestToolMessageArgs::default()
                .tool_call_id(tool_call_id)
                .content(content)
                .build()?,
        ),
    })
}

fn to_openai_tool(tool: ToolDefinition) -> ChatCompletionTools {
    ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObject {
            name: tool.name,
            description: Some(tool.description),
            parameters: Some(tool.parameters_schema),
            strict: None,
        },
    })
}

fn extract_events(chunk: CreateChatCompletionStreamResponse) -> Vec<Result<LlmEvent, LlmError>> {
    let mut events = Vec::new();
    let Some(choice) = chunk.choices.first() else {
        return events;
    };
    if let Some(content) = choice.delta.content.clone() {
        events.push(Ok(LlmEvent::TextDelta(content)));
    }
    let Some(tool_calls) = choice.delta.tool_calls.clone() else {
        return events;
    };
    for tool_call in tool_calls {
        let (name, arguments_delta) = match tool_call.function {
            Some(function) => (function.name, function.arguments.unwrap_or_default()),
            None => (None, String::new()),
        };
        events.push(Ok(LlmEvent::ToolCallDelta {
            index: tool_call.index as usize,
            id: tool_call.id,
            name,
            arguments_delta,
        }));
    }
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_messages_translate_to_openai_request_messages() {
        let messages = vec![
            Message::system("sys"),
            Message::developer("dev"),
            Message::user("hi"),
            Message::Assistant {
                text: String::new(),
                tool_calls: vec![crate::ToolCall {
                    id: "call_1".to_string(),
                    name: "now".to_string(),
                    arguments: "{}".to_string(),
                }],
            },
            Message::ToolResult {
                tool_call_id: "call_1".to_string(),
                content: "result".to_string(),
            },
        ];

        let translated = messages
            .into_iter()
            .map(to_openai_message)
            .collect::<Result<Vec<_>, _>>()
            .expect("translation succeeds");

        assert!(matches!(translated[0], ChatCompletionRequestMessage::System(_)));
        assert!(matches!(translated[1], ChatCompletionRequestMessage::Developer(_)));
        assert!(matches!(translated[2], ChatCompletionRequestMessage::User(_)));
        assert!(matches!(translated[3], ChatCompletionRequestMessage::Assistant(_)));
        assert!(matches!(translated[4], ChatCompletionRequestMessage::Tool(_)));
    }
}
