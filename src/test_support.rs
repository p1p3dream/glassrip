//! Minimal local HTTP server for retry and timeout tests. Each connection
//! gets the next scripted reply (the last one repeats) and is then closed.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[derive(Debug, Clone)]
pub enum Reply {
    Http {
        status: u16,
        headers: Vec<(String, String)>,
        body: String,
        delay: Duration,
    },
    /// Read the request, then close the socket without responding.
    Drop,
}

impl Reply {
    pub fn json(status: u16, body: serde_json::Value) -> Self {
        Reply::Http {
            status,
            headers: Vec::new(),
            body: body.to_string(),
            delay: Duration::ZERO,
        }
    }

    pub fn header(mut self, name: &str, value: &str) -> Self {
        if let Reply::Http { headers, .. } = &mut self {
            headers.push((name.to_string(), value.to_string()));
        }
        self
    }

    pub fn delay(mut self, d: Duration) -> Self {
        if let Reply::Http { delay, .. } = &mut self {
            *delay = d;
        }
        self
    }
}

pub struct TestServer {
    pub url: String,
    hits: Arc<AtomicUsize>,
}

impl TestServer {
    pub fn hits(&self) -> usize {
        self.hits.load(Ordering::SeqCst)
    }
}

pub async fn serve(replies: Vec<Reply>) -> TestServer {
    assert!(!replies.is_empty());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let hits = Arc::new(AtomicUsize::new(0));
    let replies = Arc::new(replies);
    let counter = hits.clone();
    tokio::spawn(async move {
        while let Ok((sock, _)) = listener.accept().await {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            let reply = replies[n.min(replies.len() - 1)].clone();
            tokio::spawn(handle(sock, reply));
        }
    });
    TestServer { url, hits }
}

async fn handle(mut sock: TcpStream, reply: Reply) {
    read_request(&mut sock).await;
    match reply {
        Reply::Drop => drop(sock),
        Reply::Http {
            status,
            headers,
            body,
            delay,
        } => {
            tokio::time::sleep(delay).await;
            let mut head = format!(
                "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
                body.len()
            );
            for (k, v) in headers {
                head.push_str(&format!("{k}: {v}\r\n"));
            }
            head.push_str("\r\n");
            let _ = sock.write_all(head.as_bytes()).await;
            let _ = sock.write_all(body.as_bytes()).await;
            let _ = sock.shutdown().await;
        }
    }
}

async fn read_request(sock: &mut TcpStream) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let header_end = loop {
        match sock.read(&mut chunk).await {
            Ok(0) | Err(_) => return,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_ascii_lowercase();
    let content_length = head
        .lines()
        .find_map(|l| l.strip_prefix("content-length:"))
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0);
    while buf.len() < header_end + content_length {
        match sock.read(&mut chunk).await {
            Ok(0) | Err(_) => return,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    }
}
