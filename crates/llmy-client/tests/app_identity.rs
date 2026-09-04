//! On-the-wire check that an [`llmy_client::app::AppIdentity`] actually
//! reaches the server: the identity's HTTP client carries its headers as
//! defaults on every request.

use llmy_client::app::AppIdentity;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

async fn capture_one_request(listener: TcpListener) -> String {
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
    let _ = socket
        .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\n{}")
        .await;
    head
}

#[tokio::test]
async fn identity_headers_arrive_on_the_wire() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("local addr");
    let server = tokio::spawn(capture_one_request(listener));

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
    let server = tokio::spawn(capture_one_request(listener));

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
