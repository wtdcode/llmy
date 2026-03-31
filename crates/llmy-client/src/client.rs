use std::{
    ops::Deref,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use crate::model::OpenAIModel;
use async_openai::{
    Client,
    config::{AzureConfig, OpenAIConfig},
    error::OpenAIError,
    types::chat::{
        ChatChoice, ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        ChatCompletionResponseMessage, ChatCompletionResponseStream, ChatCompletionStreamOptions,
        CompletionUsage, CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
        CreateChatCompletionResponse, CreateChatCompletionStreamResponse, FinishReason,
        FunctionCall, Role,
    },
};
use color_eyre::eyre::eyre;
use llmy_types::error::LLMYError;
use tokio::sync::RwLock;
use tokio_stream::StreamExt;

use crate::debug;
use crate::{billing::ModelBilling, settings::LLMSettings};

#[derive(Clone, Debug, Default)]
struct ToolCallAcc {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Clone)]
pub enum SupportedConfig {
    Azure(AzureConfig),
    OpenAI(OpenAIConfig),
}

impl SupportedConfig {
    pub fn new_azure(endpoint: &str, key: &str, deployment: &str, api_version: &str) -> Self {
        let cfg = AzureConfig::new()
            .with_api_base(endpoint)
            .with_api_key(key)
            .with_deployment_id(deployment)
            .with_api_version(api_version);
        Self::Azure(cfg)
    }

    pub fn new(endpoint: &str, key: &str) -> Self {
        let cfg = OpenAIConfig::new()
            .with_api_base(endpoint)
            .with_api_key(key);
        Self::OpenAI(cfg)
    }
}

#[derive(Debug, Clone)]
pub enum LLMClient {
    Azure(Client<AzureConfig>),
    OpenAI(Client<OpenAIConfig>),
}

impl LLMClient {
    pub fn new(config: SupportedConfig) -> Self {
        match config {
            SupportedConfig::Azure(cfg) => Self::Azure(Client::with_config(cfg)),
            SupportedConfig::OpenAI(cfg) => Self::OpenAI(Client::with_config(cfg)),
        }
    }

    pub async fn create_chat(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, OpenAIError> {
        match self {
            Self::Azure(cl) => cl.chat().create(req).await,
            Self::OpenAI(cl) => cl.chat().create(req).await,
        }
    }

    pub async fn create_chat_stream(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError> {
        match self {
            Self::Azure(cl) => cl.chat().create_stream(req).await,
            Self::OpenAI(cl) => cl.chat().create_stream(req).await,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LLM {
    llm: Arc<LLMInner>,
}

impl LLM {
    pub fn new(
        config: SupportedConfig,
        model: OpenAIModel,
        cap: f64,
        settings: LLMSettings,
        debug_prefix: Option<String>,
        debug_foler: Option<PathBuf>,
    ) -> Self {
        let billing = RwLock::new(ModelBilling::new(cap));

        let debug_path = if let Some(dbg) = debug_foler.as_ref() {
            let pid = std::process::id();

            let mut cnt = 0u64;
            let debug_path;
            loop {
                let prefix = if let Some(debug_prefix) = &debug_prefix {
                    if debug_prefix.is_empty() {
                        "main".to_string()
                    } else {
                        debug_prefix.to_lowercase()
                    }
                } else {
                    "main".to_string()
                };
                let test_path = dbg.join(format!("{}-{}-{}", pid, cnt, prefix));
                if !test_path.exists() {
                    std::fs::create_dir_all(&test_path).expect("Fail to create llm debug path?");
                    debug_path = Some(test_path);
                    tracing::debug!("The path to save LLM interactions is {:?}", &debug_path);
                    break;
                } else {
                    cnt += 1;
                }
            }
            debug_path
        } else {
            None
        };

        LLM {
            llm: Arc::new(LLMInner {
                client: LLMClient::new(config),
                model,
                billing,
                llm_debug: debug_path,
                llm_debug_index: AtomicU64::new(0),
                default_settings: settings,
            }),
        }
    }
}

impl Deref for LLM {
    type Target = LLMInner;

    fn deref(&self) -> &Self::Target {
        &self.llm
    }
}

#[derive(Debug)]
pub struct LLMInner {
    pub client: LLMClient,
    pub model: OpenAIModel,
    pub billing: RwLock<ModelBilling>,
    pub llm_debug: Option<PathBuf>,
    pub llm_debug_index: AtomicU64,
    pub default_settings: LLMSettings,
}

impl LLMInner {
    fn on_llm_debug(&self, prefix: &str) -> Option<PathBuf> {
        if let Some(output_folder) = self.llm_debug.as_ref() {
            let idx = self.llm_debug_index.fetch_add(1, Ordering::SeqCst);
            let fpath = output_folder.join(format!("{}-{:0>12}.xml", prefix, idx));
            Some(fpath)
        } else {
            None
        }
    }

    // we use t/s to estimate a timeout to avoid infinite repeating
    pub async fn prompt_once_with_retry(
        &self,
        sys_msg: &str,
        user_msg: &str,
        prefix: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<CreateChatCompletionResponse, LLMYError> {
        let settings = settings.unwrap_or_else(|| self.default_settings.clone());
        let timeout = settings.timeout();
        let sys = ChatCompletionRequestSystemMessageArgs::default()
            .content(sys_msg)
            .build()?;

        let user = ChatCompletionRequestUserMessageArgs::default()
            .content(user_msg)
            .build()?;
        let mut req = CreateChatCompletionRequestArgs::default();
        req.messages(vec![sys.into(), user.into()])
            .model(self.model.to_string())
            .temperature(settings.llm_temperature)
            .presence_penalty(settings.llm_presence_penalty)
            .max_completion_tokens(settings.llm_max_completion_tokens);

        if let Some(tc) = settings.llm_tool_choice {
            req.tool_choice(tc);
        }
        if let Some(effort) = settings.reasoning_effort {
            req.reasoning_effort(effort.0);
        }
        if let Some(prefix) = prefix {
            req.prompt_cache_key(prefix.to_string());
        }

        let req = req.build()?;

        self.complete_once_with_retry(&req, prefix, Some(timeout), Some(settings.llm_retry))
            .await
    }

    pub async fn complete_once_with_retry(
        &self,
        req: &CreateChatCompletionRequest,
        prefix: Option<&str>,
        timeout: Option<Duration>,
        retry: Option<u64>,
    ) -> Result<CreateChatCompletionResponse, LLMYError> {
        let retry = retry.unwrap_or(u64::MAX);

        let mut last = None;
        for idx in 0..retry {
            match self.complete(req.clone(), prefix, timeout).await {
                Ok(r) => return Ok(r),
                Err(e) => {
                    tracing::warn!("Having an error {} during {} retry", e, idx);
                    last = Some(Err(e));
                }
            }
        }

        last.ok_or_else(|| eyre!("no response after {} retries?!", retry))?
    }

    pub async fn complete(
        &self,
        req: CreateChatCompletionRequest,
        prefix: Option<&str>,
        timeout_overwrite: Option<Duration>,
    ) -> Result<CreateChatCompletionResponse, LLMYError> {
        let use_stream = self.default_settings.llm_stream;
        let prefix = if let Some(prefix) = prefix {
            prefix.to_string()
        } else {
            "llm".to_string()
        };
        let debug_fp = self.on_llm_debug(&prefix);

        if let Some(debug_fp) = debug_fp.as_ref()
            && let Err(e) = debug::save_llm_user(debug_fp, &req).await
        {
            tracing::warn!("Fail to save user due to {}", e);
        }

        let estimated_tokens = {
            let text = debug::extract_raw_text(&req);
            tracing::trace!("Text is {:?}", text);
            self.model.config.count_tokens(&text)
        };

        tracing::trace!(
            "Sending completion request: {:?}",
            &serde_json::to_string(&req)
        );

        let now = std::time::SystemTime::now();
        let llm_fut = async {
            if use_stream {
                self.complete_streaming(req).await
            } else {
                self.client.create_chat(req).await.map_err(|e| e.into())
            }
        };

        let timeout = timeout_overwrite.unwrap_or_else(|| self.default_settings.timeout());
        let resp = if timeout == Duration::MAX {
            llm_fut.await
        } else {
            tokio::time::timeout(timeout, llm_fut)
                .await
                .unwrap_or_else(|_| {
                    Err(LLMYError::Other(eyre!(
                        "LLM request timed out after {:?}",
                        timeout
                    )))
                })
        };

        let resp = match resp {
            Ok(resp) => resp,
            Err(e) => {
                if let Some(debug_fp) = debug_fp.as_ref() {
                    let err = format!("{:?}", e);
                    if let Err(je) =
                        debug::rewrite_json(debug_fp, &serde_json::json!({ "error": err })).await
                    {
                        tracing::warn!("can not save error: {} due to json error {}", err, je);
                    }
                }
                return Err(e);
            }
        };
        if let Some(debug_fp) = debug_fp.as_ref()
            && let Err(e) = debug::save_llm_resp(debug_fp, &resp).await
        {
            tracing::warn!("Fail to save resp due to {}", e);
        }

        let output_tokens = if let Some(usage) = &resp.usage {
            let mut billing = self.billing.write().await;

            let cached = usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|v| v.cached_tokens)
                .unwrap_or_default();
            let input = usage.prompt_tokens - cached;
            billing.input_tokens(&self.model, input as _, cached as _)?;
            let reasoning = usage
                .completion_tokens_details
                .as_ref()
                .and_then(|v| v.reasoning_tokens)
                .unwrap_or_default() as u64;

            billing.output_tokens(
                &self.model,
                usage.completion_tokens as u64 - reasoning,
                reasoning,
            )?;

            if let Some(debug_fp) = debug_fp.as_ref() {
                let billing_clone = billing.clone();
                drop(billing);
                if let Err(e) = debug::rewrite_json(debug_fp, &billing_clone).await {
                    tracing::warn!(
                        "can not write {} to debug file due to {}",
                        &billing_clone,
                        e
                    );
                }
            }
            if let Some(est) = estimated_tokens {
                let actual = usage.prompt_tokens as f64;
                let diff = (est as f64 - actual).abs();
                let pct = if actual > 0.0 {
                    diff / actual * 100.0
                } else {
                    0.0
                };
                tracing::info!(
                    "Token estimate: {} estimated vs {} actual (diff {:.1}%)",
                    est,
                    usage.prompt_tokens,
                    pct
                );
            }

            usage.completion_tokens
        } else {
            tracing::warn!("No usage from {:?}?!", &resp);
            0
        };

        let delta = std::time::SystemTime::now()
            .duration_since(now)
            .map(|v| v.as_secs_f64())
            .unwrap_or_default();
        tracing::info!(
            "Usage: {}, Speed: {:.2} tok/s",
            &self.billing.read().await,
            if delta.is_normal() && delta.is_sign_positive() {
                output_tokens as f64 / delta
            } else {
                0.0f64
            }
        );
        Ok(resp)
    }

    async fn complete_streaming(
        &self,
        mut req: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, LLMYError> {
        if req.stream_options.is_none() {
            req.stream_options = Some(ChatCompletionStreamOptions {
                include_usage: Some(true),
                include_obfuscation: None,
            });
        }

        let mut stream = self.client.create_chat_stream(req).await?;

        let mut id: Option<String> = None;
        let mut created: Option<u32> = None;
        let mut model: Option<String> = None;
        let mut service_tier = None;
        let mut system_fingerprint = None;
        let mut usage: Option<CompletionUsage> = None;

        let mut contents: Vec<String> = Vec::new();
        let mut finish_reasons: Vec<Option<FinishReason>> = Vec::new();
        let mut tool_calls: Vec<Vec<ToolCallAcc>> = Vec::new();

        while let Some(item) = stream.next().await {
            let chunk: CreateChatCompletionStreamResponse = item?;
            if id.is_none() {
                id = Some(chunk.id.clone());
            }
            created = Some(chunk.created);
            model = Some(chunk.model.clone());
            service_tier = chunk.service_tier.clone();
            system_fingerprint = chunk.system_fingerprint.clone();
            if let Some(u) = chunk.usage.clone() {
                usage = Some(u);
            }

            for ch in chunk.choices.into_iter() {
                let idx = ch.index as usize;
                if contents.len() <= idx {
                    contents.resize_with(idx + 1, String::new);
                    finish_reasons.resize_with(idx + 1, || None);
                    tool_calls.resize_with(idx + 1, Vec::new);
                }
                if let Some(delta) = ch.delta.content {
                    contents[idx].push_str(&delta);
                }
                if let Some(tcs) = ch.delta.tool_calls {
                    for tc in tcs.into_iter() {
                        let tc_idx = tc.index as usize;
                        if tool_calls[idx].len() <= tc_idx {
                            tool_calls[idx].resize_with(tc_idx + 1, ToolCallAcc::default);
                        }
                        let acc = &mut tool_calls[idx][tc_idx];
                        if let Some(id) = tc.id {
                            acc.id = id;
                        }
                        if let Some(func) = tc.function {
                            if let Some(name) = func.name {
                                acc.name = name;
                            }
                            if let Some(args) = func.arguments {
                                acc.arguments.push_str(&args);
                            }
                        }
                    }
                }
                if ch.finish_reason.is_some() {
                    finish_reasons[idx] = ch.finish_reason;
                }
            }
        }

        let mut choices = Vec::new();
        for (idx, content) in contents.into_iter().enumerate() {
            let finish_reason = finish_reasons.get(idx).cloned().unwrap_or(None);
            let built_tool_calls = tool_calls
                .get(idx)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter(|t| !t.name.trim().is_empty() || !t.arguments.trim().is_empty())
                .map(|t| {
                    ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
                        id: if t.id.trim().is_empty() {
                            format!("toolcall-{}", idx)
                        } else {
                            t.id
                        },
                        function: FunctionCall {
                            name: t.name,
                            arguments: t.arguments,
                        },
                    })
                })
                .collect::<Vec<_>>();
            let tool_calls_opt = if built_tool_calls.is_empty() {
                None
            } else {
                Some(built_tool_calls)
            };
            choices.push(ChatChoice {
                index: idx as u32,
                message: ChatCompletionResponseMessage {
                    content: if content.is_empty() {
                        None
                    } else {
                        Some(content)
                    },
                    refusal: None,
                    tool_calls: tool_calls_opt,
                    annotations: None,
                    role: Role::Assistant,
                    function_call: None,
                    audio: None,
                },
                finish_reason,
                logprobs: None,
            });
        }
        if choices.is_empty() {
            choices.push(ChatChoice {
                index: 0,
                message: ChatCompletionResponseMessage {
                    content: Some(String::new()),
                    refusal: None,
                    tool_calls: None,
                    annotations: None,
                    role: Role::Assistant,
                    function_call: None,
                    audio: None,
                },
                finish_reason: None,
                logprobs: None,
            });
        }

        Ok(CreateChatCompletionResponse {
            id: id.unwrap_or_else(|| "stream".to_string()),
            choices,
            created: created.unwrap_or(0),
            model: model.unwrap_or_else(|| self.model.to_string()),
            service_tier,
            system_fingerprint,
            object: "chat.completion".to_string(),
            usage,
        })
    }

    pub async fn prompt_once(
        &self,
        sys_msg: &str,
        user_msg: &str,
        prefix: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<CreateChatCompletionResponse, LLMYError> {
        let settings = settings.unwrap_or_else(|| self.default_settings.clone());
        let sys = ChatCompletionRequestSystemMessageArgs::default()
            .content(sys_msg)
            .build()?;

        let user = ChatCompletionRequestUserMessageArgs::default()
            .content(user_msg)
            .build()?;
        let mut req = CreateChatCompletionRequestArgs::default();

        if let Some(tc) = settings.llm_tool_choice {
            req.tool_choice(tc);
        }

        if let Some(effort) = settings.reasoning_effort {
            req.reasoning_effort(effort.0);
        }

        if let Some(prefix) = prefix.as_ref() {
            req.prompt_cache_key(prefix.to_string());
        }
        let req = req
            .messages(vec![sys.into(), user.into()])
            .model(self.model.to_string())
            .temperature(settings.llm_temperature)
            .presence_penalty(settings.llm_presence_penalty)
            .max_completion_tokens(settings.llm_max_completion_tokens)
            .build()?;
        self.complete(req, prefix, None).await
    }
}
