use std::{
    fmt::Debug,
    path::{Path, PathBuf},
};

use tokio::io::AsyncWriteExt;

use async_openai::types::chat::CreateChatCompletionRequest;
use color_eyre::eyre::eyre;
use itertools::Itertools;
use llmy_types::error::LLMYError;
use serde::Serialize;

use async_openai::{
    Client,
    config::{AzureConfig, OpenAIConfig},
    error::OpenAIError,
    types::chat::{
        ChatChoice, ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
        ChatCompletionNamedToolChoiceCustom, ChatCompletionRequestAssistantMessageContent,
        ChatCompletionRequestAssistantMessageContentPart,
        ChatCompletionRequestDeveloperMessageContent,
        ChatCompletionRequestDeveloperMessageContentPart, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestSystemMessageContent,
        ChatCompletionRequestSystemMessageContentPart, ChatCompletionRequestToolMessageContent,
        ChatCompletionRequestToolMessageContentPart, ChatCompletionRequestUserMessageArgs,
        ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
        ChatCompletionResponseMessage, ChatCompletionResponseStream, ChatCompletionStreamOptions,
        ChatCompletionToolChoiceOption, ChatCompletionTools, CompletionUsage,
        CreateChatCompletionRequestArgs, CreateChatCompletionResponse,
        CreateChatCompletionStreamResponse, CustomName, FinishReason, FunctionCall,
        ReasoningEffort, Role, ToolChoiceOptions,
    },
};

pub fn completion_to_role(msg: &ChatCompletionRequestMessage) -> &'static str {
    match msg {
        ChatCompletionRequestMessage::Assistant(_) => "ASSISTANT",
        ChatCompletionRequestMessage::Developer(_) => "DEVELOPER",
        ChatCompletionRequestMessage::Function(_) => "FUNCTION",
        ChatCompletionRequestMessage::System(_) => "SYSTEM",
        ChatCompletionRequestMessage::Tool(_) => "TOOL",
        ChatCompletionRequestMessage::User(_) => "USER",
    }
}

pub fn toolcall_to_string(t: &ChatCompletionMessageToolCalls) -> String {
    match t {
        ChatCompletionMessageToolCalls::Function(t) => {
            format!(
                "<toolcall name=\"{}\">\n{}\n</toolcall>",
                &t.function.name, &t.function.arguments
            )
        }
        ChatCompletionMessageToolCalls::Custom(t) => {
            format!(
                "<customtoolcall name=\"{}\">\n{}\n</customtoolcall>",
                &t.custom_tool.name, &t.custom_tool.input
            )
        }
    }
}

pub fn response_to_string(resp: &ChatCompletionResponseMessage) -> String {
    let mut s = String::new();
    if let Some(content) = resp.content.as_ref() {
        s += content;
        s += "\n";
    }

    if let Some(tools) = resp.tool_calls.as_ref() {
        s += &tools.iter().map(|t| toolcall_to_string(t)).join("\n");
    }

    if let Some(refusal) = &resp.refusal {
        s += refusal;
        s += "\n";
    }

    let role = resp.role.to_string().to_uppercase();

    format!("<{}>\n{}\n</{}>\n", &role, s, &role)
}

pub fn completion_to_string(msg: &ChatCompletionRequestMessage) -> String {
    const CONT: &str = "<cont/>\n";
    const NONE: &str = "<none/>\n";
    let role = completion_to_role(msg);
    let content = match msg {
        ChatCompletionRequestMessage::Assistant(ass) => {
            let msg = ass
                .content
                .as_ref()
                .map(|ass| match ass {
                    ChatCompletionRequestAssistantMessageContent::Text(s) => s.clone(),
                    ChatCompletionRequestAssistantMessageContent::Array(arr) => arr
                        .iter()
                        .map(|v| match v {
                            ChatCompletionRequestAssistantMessageContentPart::Text(s) => {
                                s.text.clone()
                            }
                            ChatCompletionRequestAssistantMessageContentPart::Refusal(rf) => {
                                rf.refusal.clone()
                            }
                        })
                        .join(CONT),
                })
                .unwrap_or(NONE.to_string());
            let tool_calls = ass
                .tool_calls
                .iter()
                .flatten()
                .map(|t| toolcall_to_string(t))
                .join("\n");
            format!("{}\n{}", msg, tool_calls)
        }
        ChatCompletionRequestMessage::Developer(dev) => match &dev.content {
            ChatCompletionRequestDeveloperMessageContent::Text(t) => t.clone(),
            ChatCompletionRequestDeveloperMessageContent::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    ChatCompletionRequestDeveloperMessageContentPart::Text(v) => v.text.clone(),
                })
                .join(CONT),
        },
        ChatCompletionRequestMessage::Function(f) => f.content.clone().unwrap_or(NONE.to_string()),
        ChatCompletionRequestMessage::System(sys) => match &sys.content {
            ChatCompletionRequestSystemMessageContent::Text(t) => t.clone(),
            ChatCompletionRequestSystemMessageContent::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    ChatCompletionRequestSystemMessageContentPart::Text(t) => t.text.clone(),
                })
                .join(CONT),
        },
        ChatCompletionRequestMessage::Tool(tool) => match &tool.content {
            ChatCompletionRequestToolMessageContent::Text(t) => t.clone(),
            ChatCompletionRequestToolMessageContent::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    ChatCompletionRequestToolMessageContentPart::Text(t) => t.text.clone(),
                })
                .join(CONT),
        },
        ChatCompletionRequestMessage::User(usr) => match &usr.content {
            ChatCompletionRequestUserMessageContent::Text(t) => t.clone(),
            ChatCompletionRequestUserMessageContent::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    ChatCompletionRequestUserMessageContentPart::Text(t) => t.text.clone(),
                    ChatCompletionRequestUserMessageContentPart::ImageUrl(img) => {
                        format!("<img url=\"{}\"/>", &img.image_url.url)
                    }
                    ChatCompletionRequestUserMessageContentPart::InputAudio(audio) => {
                        format!("<audio>{}</audio>", audio.input_audio.data)
                    }
                    ChatCompletionRequestUserMessageContentPart::File(f) => {
                        format!("<file>{:?}</file>", f)
                    }
                })
                .join(CONT),
        },
    };

    format!("<{}>\n{}\n</{}>\n", role, content, role)
}

pub(crate) async fn rewrite_json<T: Serialize + Debug>(
    fpath: &Path,
    t: &T,
) -> Result<(), LLMYError> {
    let mut json_fp = fpath.to_path_buf();
    json_fp.set_file_name(format!(
        "{}.json",
        json_fp
            .file_stem()
            .ok_or_else(|| eyre!("no filename"))?
            .to_str()
            .ok_or_else(|| eyre!("non-utf fname"))?
    ));

    let mut fp = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .write(true)
        .open(&json_fp)
        .await?;
    let s = match serde_json::to_string(&t) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("can not serialize due to {}", e);
            format!("{:?}", &t)
        }
    };
    fp.write_all(s.as_bytes()).await?;
    fp.write_all(b"\n").await?;
    fp.flush().await?;

    Ok(())
}

pub(crate) async fn save_llm_user(
    fpath: &PathBuf,
    user_msg: &CreateChatCompletionRequest,
) -> Result<(), LLMYError> {
    let mut fp = tokio::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&fpath)
        .await?;
    fp.write_all(b"=====================\n<Request>\n").await?;
    for it in user_msg.messages.iter() {
        let msg = completion_to_string(it);
        fp.write_all(msg.as_bytes()).await?;
    }

    let mut tools = vec![];
    for tool in user_msg
        .tools
        .as_ref()
        .map(|t| t.iter())
        .into_iter()
        .flatten()
    {
        let s = match tool {
            ChatCompletionTools::Function(tool) => {
                format!(
                    "<tool name=\"{}\", description=\"{}\", strict={}>\n{}\n</tool>",
                    &tool.function.name,
                    &tool.function.description.clone().unwrap_or_default(),
                    tool.function.strict.unwrap_or_default(),
                    tool.function
                        .parameters
                        .as_ref()
                        .map(serde_json::to_string_pretty)
                        .transpose()?
                        .unwrap_or_default()
                )
            }
            ChatCompletionTools::Custom(tool) => {
                format!(
                    "<customtool name=\"{}\", description=\"{:?}\"></customtool>",
                    tool.custom.name, tool.custom.description
                )
            }
        };
        tools.push(s);
    }
    fp.write_all(tools.join("\n").as_bytes()).await?;
    fp.write_all(b"\n</Request>\n=====================\n")
        .await?;
    fp.flush().await?;

    rewrite_json(fpath, user_msg).await?;

    Ok(())
}

pub(crate) async fn save_llm_resp(
    fpath: &PathBuf,
    resp: &CreateChatCompletionResponse,
) -> Result<(), LLMYError> {
    let mut fp = tokio::fs::OpenOptions::new()
        .create(false)
        .append(true)
        .write(true)
        .open(&fpath)
        .await?;
    fp.write_all(b"=====================\n<Response>\n").await?;
    for it in &resp.choices {
        let msg = response_to_string(&it.message);
        fp.write_all(msg.as_bytes()).await?;
    }
    fp.write_all(b"\n</Response>\n=====================\n")
        .await?;
    fp.flush().await?;

    rewrite_json(fpath, resp).await?;

    Ok(())
}
