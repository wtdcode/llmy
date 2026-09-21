//! On-the-wire checks of the LLM fallback chain: a dead primary spills to
//! the backup profile (with the model id rewritten to the backup's), and a
//! profile over its own spend cap is skipped without failing the client.

use std::str::FromStr;
use std::time::Duration;

use llmy_client::client::{LLM, LLMProfile, SupportedConfig};
use llmy_client::model::OpenAIModel;
use llmy_client::req::{
    ChatCompletionRequestMessageRaw, ChatCompletionRequestUserMessageRaw,
    CreateChatCompletionRequestRaw,
};
use llmy_client::settings::LLMSettings;
use llmy_types::other::WithOtherFields;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Settings tuned for tests: no backoff pauses, no auto cache keys, two
/// attempts per profile.
fn test_settings() -> LLMSettings {
    LLMSettings {
        llm_temperature: None,
        llm_presence_penalty: None,
        llm_prompt_timeout: 10,
        llm_retry: 2,
        llm_retry_backoff_secs: 0.0,
        llm_retry_backoff_factor: 2.0,
        llm_retry_backoff_max_secs: 16.0,
        tool_reject_retries: 32,
        unknown_tool_hard_reject: true,
        llm_concurrent: 0,
        llm_max_completion_tokens: None,
        llm_tool_choice: None,
        llm_stream: false,
        top_p: None,
        reasoning_effort: None,
        auto_strip: false,
        auto_cache_key: false,
        cache_key_ttl: 0,
        cache_key_rpm: 0,
        billing_log_tokens: 0,
        token_estimate_pct: 100.0,
        allow_implicit_convert: false,
        llm_app: None,
    }
}

fn profile(name: &str, url: &str, model: &str) -> LLMProfile {
    LLMProfile {
        name: name.to_string(),
        config: SupportedConfig::new(url, "test-key"),
        model: OpenAIModel::from_str(model).expect("model"),
        settings: test_settings(),
        cap: None,
    }
}

/// Serve `count` chat completions on `listener`, returning the raw request
/// heads+bodies observed. Every response reports `prompt_tokens` prompt
/// tokens so billing has something to record.
async fn serve_chat_completions(
    listener: TcpListener,
    count: usize,
    prompt_tokens: u64,
) -> Vec<String> {
    let mut seen = Vec::new();
    for _ in 0..count {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let mut buffer = vec![0u8; 65536];
        let mut request = String::new();
        loop {
            let read = socket.read(&mut buffer).await.expect("read");
            request.push_str(&String::from_utf8_lossy(&buffer[..read]));
            if read == 0 {
                break;
            }
            // Stop once the whole declared body arrived.
            if let Some(head_end) = request.find("\r\n\r\n") {
                let content_length = request
                    .to_lowercase()
                    .lines()
                    .find_map(|line| {
                        line.strip_prefix("content-length:")
                            .map(str::trim)
                            .map(String::from)
                    })
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(0);
                if request.len() >= head_end + 4 + content_length {
                    break;
                }
            }
        }
        let body = serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "served-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 2,
                "total_tokens": prompt_tokens + 2
            }
        })
        .to_string();
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = socket.write_all(response.as_bytes()).await;
        seen.push(request);
    }
    seen
}

fn user_request(model: &str) -> CreateChatCompletionRequestRaw {
    let mut raw = CreateChatCompletionRequestRaw::default();
    raw.model = model.to_string();
    raw.messages = vec![WithOtherFields::new(ChatCompletionRequestMessageRaw::User(
        ChatCompletionRequestUserMessageRaw::new_text("hello"),
    ))];
    raw
}

#[tokio::test]
async fn a_dead_primary_falls_over_to_the_backup_with_its_model_id() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let backup_url = format!("http://{}/v1", listener.local_addr().expect("addr"));
    let server = tokio::spawn(serve_chat_completions(listener, 1, 10));

    // Port 1 refuses connections immediately: the primary's whole retry
    // budget burns fast and the chain moves on.
    let llm = LLM::new_with_fallback(
        vec![
            profile("dead", "http://127.0.0.1:1/v1", "primary-model,1,1"),
            profile("backup", &backup_url, "backup-model,1,1"),
        ],
        rust_decimal::dec!(100),
        Duration::from_secs(60),
        None,
    )
    .expect("llm");

    let resp = llm
        .complete_once_with_retry(&user_request("primary-model"), None, None, None)
        .await
        .expect("fallback response");
    assert_eq!(
        resp.choices[0].inner.message.inner.content.as_deref(),
        Some("ok")
    );

    // The backup rewrote the model id to its own before sending.
    let seen = server.await.expect("server");
    assert!(
        seen[0].contains("\"model\":\"backup-model\""),
        "{}",
        seen[0]
    );

    // The spend landed on the backup's ledger, none on the dead primary's.
    assert_eq!(llm.targets[0].name, "dead");
    assert_eq!(
        *llm.targets[0].spent.read().unwrap(),
        rust_decimal::Decimal::ZERO
    );
    assert!(*llm.targets[1].spent.read().unwrap() > rust_decimal::Decimal::ZERO);
}

#[tokio::test]
async fn a_profile_over_its_own_cap_spills_to_the_next() {
    let first = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let second = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let first_url = format!("http://{}/v1", first.local_addr().expect("addr"));
    let second_url = format!("http://{}/v1", second.local_addr().expect("addr"));
    // 1M prompt tokens at $1/1M => every request costs about $1.
    let first_server = tokio::spawn(serve_chat_completions(first, 1, 1_000_000));
    let second_server = tokio::spawn(serve_chat_completions(second, 1, 1_000_000));

    let mut capped = profile("capped", &first_url, "first-model,1,1");
    capped.cap = Some(rust_decimal::dec!(0.5));
    let llm = LLM::new_with_fallback(
        vec![capped, profile("spill", &second_url, "second-model,1,1")],
        rust_decimal::dec!(100),
        Duration::from_secs(60),
        None,
    )
    .expect("llm");

    // First request goes to the capped profile and blows its budget.
    llm.complete_once_with_retry(&user_request("first-model"), None, None, None)
        .await
        .expect("first response");
    assert!(*llm.targets[0].spent.read().unwrap() > rust_decimal::dec!(0.5));

    // The second request skips it and lands on the spill profile.
    llm.complete_once_with_retry(&user_request("first-model"), None, None, None)
        .await
        .expect("spilled response");
    let seen = second_server.await.expect("second server");
    assert!(
        seen[0].contains("\"model\":\"second-model\""),
        "{}",
        seen[0]
    );
    drop(first_server);
}
