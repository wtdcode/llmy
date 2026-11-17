//! On-the-wire checks of the HTTP clients llmy sends through: an
//! [`llmy_client::app::AppIdentity`]'s client carries its headers as defaults
//! on every request, and every client asks for and decodes compressed
//! responses.

use llmy_client::app::AppIdentity;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Answer one request with `response` (a full HTTP response), returning the
/// request head as received.
async fn capture_one_request(listener: TcpListener, response: Vec<u8>) -> String {
    let (mut socket, _) = listener.accept().await.expect("accept");
    let mut buffer = vec![0u8; 8192];
    let mut head = String::new();
    loop {
        let read = socket.read(&mut buffer).await.expect("read");
        head.push_str(&String::from_utf8_lossy(&buffer[..read]));
        if head.contains("\r\n\r\n") || read == 0 {
            break;
        }
    }
    let _ = socket.write_all(&response).await;
    head
}

fn empty_json_response() -> Vec<u8> {
    b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\n{}".to_vec()
}

/// `body` gzipped, served with `content-encoding: gzip`.
fn gzip_response(body: &str) -> Vec<u8> {
    use std::io::Write;
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(body.as_bytes()).expect("gzip");
    let compressed = encoder.finish().expect("gzip");
    let mut response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-encoding: gzip\r\ncontent-length: {}\r\n\r\n",
        compressed.len()
    )
    .into_bytes();
    response.extend(compressed);
    response
}

#[tokio::test]
async fn identity_headers_arrive_on_the_wire() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("local addr");
    let server = tokio::spawn(capture_one_request(listener, empty_json_response()));

    let identity = AppIdentity::claude_code();
    let client = identity.http_client().expect("http client");
    let _ = client
        .get(format!("http://{address}/v1/models"))
        .send()
        .await
        .expect("request");

    let head = server.await.expect("server task");
    let lowered = head.to_lowercase();
    assert!(
        lowered.contains(&format!(
            "user-agent: {}",
            identity.user_agent.to_lowercase()
        )),
        "{head}"
    );
    assert!(lowered.contains("x-app: cli"), "{head}");
    assert!(lowered.contains("x-stainless-lang: js"), "{head}");
}

#[tokio::test]
async fn codex_identity_sends_originator() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("local addr");
    let server = tokio::spawn(capture_one_request(listener, empty_json_response()));

    let client = AppIdentity::codex().http_client().expect("http client");
    let _ = client
        .get(format!("http://{address}/"))
        .send()
        .await
        .expect("request");

    let head = server.await.expect("server task").to_lowercase();
    assert!(head.contains("originator: codex_cli_rs"), "{head}");
    assert!(head.contains("user-agent: codex_cli_rs/"), "{head}");
}

#[tokio::test]
async fn every_client_asks_for_and_decodes_compressed_responses() {
    // async-openai builds its own client with `reqwest::Client::new()` when
    // no app identity is configured; the identity path builds one of its own.
    let body = r#"{"object": "list", "data": []}"#;
    for (path, client) in [
        ("default", reqwest::Client::new()),
        (
            "app identity",
            AppIdentity::codex().http_client().expect("http client"),
        ),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("local addr");
        let server = tokio::spawn(capture_one_request(listener, gzip_response(body)));

        let text = client
            .get(format!("http://{address}/v1/models"))
            .send()
            .await
            .expect("request")
            .text()
            .await
            .expect("decoded body");
        assert_eq!(text, body, "{path}");

        let head = server.await.expect("server task").to_lowercase();
        let accepted = head
            .lines()
            .find_map(|line| line.strip_prefix("accept-encoding:"))
            .unwrap_or_else(|| panic!("{path}: no accept-encoding in {head}"));
        for codec in ["gzip", "br", "zstd", "deflate"] {
            assert!(
                accepted.contains(codec),
                "{path}: {codec} missing from {accepted}"
            );
        }
    }
}
