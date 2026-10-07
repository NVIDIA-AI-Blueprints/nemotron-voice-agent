// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

//! Private stdin/stdout helper built on the pinned supported OpenShell SDK.
//! No CLI credential arguments, environment discovery, refresh, or replay.

use base64::{Engine, engine::general_purpose::STANDARD};
use openshell_sdk::{AuthConfig, ClientConfig, OpenShellClient, raw::proto};
use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::time::{Instant, timeout_at};

const INPUT_LIMIT: usize = 512 * 1024;
const OUTPUT_LIMIT: usize = 4 * 1024 * 1024;
const CONTROL_LIMIT: u64 = 1024 * 1024;

#[derive(Clone, Copy, Deserialize)]
enum AuthenticationMode {
    #[serde(rename = "oidcBearer")]
    OidcBearer,
    #[serde(rename = "none")]
    None,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    endpoint: String,
    workspace: String,
    sandbox: String,
    #[serde(rename = "sandboxId")]
    sandbox_id: String,
    argv: Vec<String>,
    stdin: String,
    #[serde(rename = "authenticationMode")]
    authentication_mode: AuthenticationMode,
    bearer: Option<String>,
    seconds: f64,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Output {
    Failure {
        failure: &'static str,
    },
    Completed {
        #[serde(rename = "exitCode")]
        exit_code: i32,
        stdout: String,
        stderr: String,
    },
}

fn failure(code: &'static str) -> Output {
    Output::Failure { failure: code }
}

fn status_code(status: &tonic::Status, dispatched: bool) -> &'static str {
    match status.code() {
        tonic::Code::Unauthenticated | tonic::Code::PermissionDenied => "access_denied",
        _ if dispatched => "outcome_unconfirmed",
        _ => "transport_unavailable",
    }
}

fn append_bound(
    stdout: &mut Vec<u8>,
    stderr: &mut Vec<u8>,
    data: &[u8],
    error: bool,
) -> Result<(), &'static str> {
    if data.len() > OUTPUT_LIMIT.saturating_sub(stdout.len() + stderr.len()) {
        return Err("output_limit");
    }
    if error {
        stderr.extend_from_slice(data);
    } else {
        stdout.extend_from_slice(data);
    }
    Ok(())
}

fn validate(input: &Input) -> Result<Vec<u8>, &'static str> {
    let url = url::Url::parse(&input.endpoint).map_err(|_| "protocol_error")?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.path() != "/"
        || input.workspace.is_empty()
        || input.sandbox_id.is_empty()
        || !input.seconds.is_finite()
        || input.seconds <= 0.0
        || input.seconds > 120.0
    {
        return Err("protocol_error");
    }
    match input.authentication_mode {
        AuthenticationMode::OidcBearer => {
            let bearer = input.bearer.as_ref().ok_or("protocol_error")?;
            if url.scheme() != "https"
                || bearer.is_empty()
                || bearer.len() > 16384
                || bearer.bytes().any(|b| !(33..=126).contains(&b))
            {
                return Err("protocol_error");
            }
        }
        AuthenticationMode::None => {
            let Some(url::Host::Ipv4(ip)) = url.host() else {
                return Err("protocol_error");
            };
            let port = url.port_or_known_default().ok_or("protocol_error")?;
            let canonical = format!("http://{ip}:{port}");
            if input.bearer.is_some()
                || !ip.is_private()
                || port == 0
                || (input.endpoint != canonical && input.endpoint != format!("{canonical}/"))
            {
                return Err("protocol_error");
            }
        }
    }
    // Python validates the descriptor. Independently constrain the helper to
    // the two allowed bridge argv forms before touching the network.
    let invoke = input.argv.len() == 6
        && input.argv[0] == "/usr/local/bin/fabric-agent"
        && input.argv[1] == "invoke"
        && input.argv[2] == "--agent"
        && !input.argv[3].is_empty()
        && input.argv[4] == "--input"
        && input.argv[5] == "-";
    let check = input.argv.len() == 5
        && input.argv[0] == "/usr/local/bin/fabric-agent"
        && input.argv[1] == "check"
        && input.argv[2] == "--agent"
        && !input.argv[3].is_empty()
        && input.argv[4] == "--live";
    if !invoke && !check {
        return Err("protocol_error");
    }
    let stdin = STANDARD
        .decode(&input.stdin)
        .map_err(|_| "protocol_error")?;
    if stdin.len() > INPUT_LIMIT || (check && !stdin.is_empty()) {
        return Err("protocol_error");
    }
    if invoke
        && !serde_json::from_slice::<serde_json::Value>(&stdin)
            .map_err(|_| "protocol_error")?
            .is_object()
    {
        return Err("protocol_error");
    }
    Ok(stdin)
}

fn client_config(input: &Input) -> Result<ClientConfig, &'static str> {
    let mut config = ClientConfig::new(input.endpoint.clone());
    config.auth = match (&input.authentication_mode, &input.bearer) {
        (AuthenticationMode::OidcBearer, Some(bearer)) => Some(AuthConfig::oidc(bearer.clone())),
        (AuthenticationMode::None, None) => None,
        _ => return Err("protocol_error"),
    };
    // SDK HTTP/no-auth or peer-verified HTTPS/OIDC, selected once. Never
    // discover credentials, disable HTTPS verification, refresh or downgrade.
    Ok(config)
}

async fn execute(
    input: Input,
    stdin: Vec<u8>,
    dispatched: &AtomicBool,
    deadline: Instant,
) -> Output {
    let config = match client_config(&input) {
        Ok(config) => config,
        Err(code) => return failure(code),
    };
    let client = match OpenShellClient::connect(config).await {
        Ok(client) => client,
        Err(_) => return failure("transport_unavailable"),
    };
    let mut grpc = client
        .raw_grpc()
        .max_decoding_message_size(OUTPUT_LIMIT + 1024);
    let mut lookup = tonic::Request::new(proto::GetSandboxRequest {
        name: input.sandbox.clone(),
        workspace_scope: Some(proto::workspace_selector(&input.workspace)),
    });
    lookup.set_timeout(deadline.saturating_duration_since(Instant::now()));
    let response = match grpc.get_sandbox(lookup).await {
        Ok(response) => response.into_inner(),
        Err(status) => return failure(status_code(&status, false)),
    };
    // No expected sandbox ID exists on ExecSandboxRequest at this pin. This
    // catches observed replacement, but is not atomic target-ID authorization.
    let observed = response
        .sandbox
        .as_ref()
        .and_then(|sandbox| sandbox.metadata.as_ref())
        .map(|meta| meta.id.as_str());
    if observed != Some(input.sandbox_id.as_str()) {
        return failure("target_replaced");
    }
    let remaining = deadline.saturating_duration_since(Instant::now());
    let mut request = tonic::Request::new(proto::ExecSandboxRequest {
        sandbox: input.sandbox,
        workspace_scope: Some(proto::workspace_selector(&input.workspace)),
        command: input.argv,
        stdin,
        tty: false,
        no_login_shell: true,
        execution_timeout: Some(prost_types::Duration {
            seconds: remaining.as_secs() as i64,
            nanos: remaining.subsec_nanos() as i32,
        }),
        ..Default::default()
    });
    request.set_timeout(remaining);
    dispatched.store(true, Ordering::Relaxed);
    let mut stream = match grpc.exec_sandbox(request).await {
        Ok(response) => response.into_inner(),
        Err(status) => return failure(status_code(&status, true)),
    };
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut exit = None;
    loop {
        let event = match stream.message().await {
            Ok(Some(event)) => event,
            Ok(None) => break,
            Err(status) => return failure(status_code(&status, true)),
        };
        if exit.is_some() {
            return failure("protocol_error");
        }
        let bounded = match event.payload {
            Some(proto::exec_sandbox_event::Payload::Stdout(chunk)) => {
                append_bound(&mut stdout, &mut stderr, &chunk.data, false)
            }
            Some(proto::exec_sandbox_event::Payload::Stderr(chunk)) => {
                append_bound(&mut stdout, &mut stderr, &chunk.data, true)
            }
            Some(proto::exec_sandbox_event::Payload::Exit(status)) => {
                exit = Some(status.exit_code);
                Ok(())
            }
            None => Err("protocol_error"),
        };
        if let Err(code) = bounded {
            return failure(code);
        }
    }
    match exit {
        Some(exit_code) => Output::Completed {
            exit_code,
            stdout: STANDARD.encode(stdout),
            stderr: STANDARD.encode(stderr),
        },
        None => failure("outcome_unconfirmed"),
    }
}

#[tokio::main]
async fn main() {
    if std::env::args().skip(1).eq(["--version"]) {
        println!(
            "voiceclaw-openshell-exec 0.1.0 openshell@6648bd0c290efbc41ba131ee9831ee45cd431f94"
        );
        return;
    }
    if std::env::args().len() != 1 {
        let _ = io::stdout().write_all(b"{\"failure\":\"protocol_error\"}");
        return;
    }
    let mut raw = Vec::new();
    let output = if io::stdin()
        .take(CONTROL_LIMIT + 1)
        .read_to_end(&mut raw)
        .is_err()
        || raw.len() as u64 > CONTROL_LIMIT
    {
        failure("protocol_error")
    } else if let Ok(input) = serde_json::from_slice::<Input>(&raw) {
        match validate(&input) {
            Err(code) => failure(code),
            Ok(stdin) => {
                let deadline = Instant::now() + Duration::from_secs_f64(input.seconds);
                let dispatched = AtomicBool::new(false);
                match timeout_at(deadline, execute(input, stdin, &dispatched, deadline)).await {
                    Ok(result) => result,
                    Err(_) => failure(if dispatched.load(Ordering::Relaxed) {
                        "outcome_unconfirmed"
                    } else {
                        "transport_unavailable"
                    }),
                }
            }
        }
    } else {
        failure("protocol_error")
    };
    if let Ok(encoded) = serde_json::to_vec(&output) {
        let _ = io::stdout().write_all(&encoded);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> Input {
        Input {
            endpoint: "https://gateway.example.test:8443".into(),
            workspace: "workspace-explicit".into(),
            sandbox: "assistant".into(),
            sandbox_id: "7e68ff54-a8a6-4b8c-b639-ae35939f6ba5".into(),
            argv: vec![
                "/usr/local/bin/fabric-agent",
                "invoke",
                "--agent",
                "assistant",
                "--input",
                "-",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            stdin: STANDARD.encode(b"{\"agent\":\"assistant\",\"message\":\"fixture\"}"),
            authentication_mode: AuthenticationMode::OidcBearer,
            bearer: Some("fixture.bearer".into()),
            seconds: 1.0,
        }
    }

    #[test]
    fn helper_rejects_plaintext_ambient_workspace_controls_and_shell_commands() {
        for change in 0..5 {
            let mut value = input();
            match change {
                0 => value.endpoint = "http://gateway.example.test:8443".into(),
                1 => value.workspace.clear(),
                2 => value.bearer = Some("fixture\r\nAuthorization: another".into()),
                3 => value.argv = vec!["sh".into(), "-c".into(), "echo fixture".into()],
                _ => value.seconds = 121.0,
            }
            assert_eq!(validate(&value), Err("protocol_error"));
        }
        assert!(validate(&input()).is_ok());
    }

    fn no_auth_input() -> Input {
        let mut value = input();
        value.endpoint = "http://172.18.0.2:8080".into();
        value.authentication_mode = AuthenticationMode::None;
        value.bearer = None;
        value
    }

    #[test]
    fn helper_accepts_only_upstream_absolute_invoke_and_live_check_commands() {
        let mut value = input();
        assert!(validate(&value).is_ok());
        value.argv = [
            "/usr/local/bin/fabric-agent",
            "check",
            "--agent",
            "assistant",
            "--live",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        value.stdin = STANDARD.encode(b"");
        assert!(validate(&value).is_ok());
        value.argv[4] = "--ready".into();
        assert_eq!(validate(&value), Err("protocol_error"));
        value.argv[4] = "--live".into();
        value.argv[0] = "fabric-agent".into();
        assert_eq!(validate(&value), Err("protocol_error"));
    }

    #[test]
    fn explicit_auth_modes_build_only_the_selected_sdk_configuration() {
        assert!(validate(&no_auth_input()).is_ok());
        let config = client_config(&no_auth_input()).unwrap();
        assert!(config.auth.is_none());
        assert!(config.ca_cert.is_none());
        assert!(!config.insecure_skip_verify);
        assert_eq!(config.gateway, "http://172.18.0.2:8080");
        let config = client_config(&input()).unwrap();
        assert!(!config.insecure_skip_verify);
        assert!(config.ca_cert.is_none());
        match config.auth.unwrap() {
            AuthConfig::Oidc {
                token,
                expires_at,
                refresh,
            } => {
                assert_eq!(token, "fixture.bearer");
                assert!(expires_at.is_none() && refresh.is_none());
            }
            _ => panic!("only static OIDC is accepted"),
        }
    }

    #[test]
    fn no_auth_rejects_tokens_tls_and_noncanonical_private_endpoints() {
        let mut with_token = no_auth_input();
        with_token.bearer = Some("fixture.bearer".into());
        assert_eq!(validate(&with_token), Err("protocol_error"));
        assert!(client_config(&with_token).is_err());
        for endpoint in [
            "https://172.18.0.2:8080",
            "http://gateway:8080",
            "http://8.8.8.8:8080",
            "http://127.0.0.1:8080",
            "http://0.0.0.0:8080",
            "http://169.254.0.1:8080",
            "http://100.64.0.1:8080",
            "http://198.18.0.1:8080",
            "http://[fd00::1]:8080",
            "http://[::ffff:172.18.0.2]:8080",
            "http://0xac120002:8080",
            "http://172.18.2:8080",
            "http://172.018.0.2:8080",
            "http://172.18.0.2",
            "http://172.18.0.2:0",
            "http://user:secret@172.18.0.2:8080",
            "http://172.18.0.2:8080?token=x",
        ] {
            let mut value = no_auth_input();
            value.endpoint = endpoint.into();
            assert_eq!(validate(&value), Err("protocol_error"), "{endpoint}");
        }
        let mut missing_token = input();
        missing_token.bearer = None;
        assert_eq!(validate(&missing_token), Err("protocol_error"));
        assert!(client_config(&missing_token).is_err());
    }

    #[tokio::test]
    async fn native_http_sdk_sends_no_auth_and_keeps_denial_and_target_checks() {
        for denied in [true, false] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            let peer = tokio::spawn(async move {
                let (socket, _) = listener.accept().await.unwrap();
                let mut connection = h2::server::handshake(socket).await.unwrap();
                let (request, mut respond) = connection.accept().await.unwrap().unwrap();
                assert!(request.uri().path().ends_with("/GetSandbox"));
                for header in ["authorization", "cookie", "cf-access-jwt-assertion"] {
                    assert!(request.headers().get(header).is_none());
                }
                let response = http::Response::builder()
                    .status(200)
                    .header("content-type", "application/grpc");
                if denied {
                    respond
                        .send_response(response.header("grpc-status", "7").body(()).unwrap(), true)
                        .unwrap();
                } else {
                    let mut stream = respond
                        .send_response(response.body(()).unwrap(), false)
                        .unwrap();
                    // Empty GetSandboxResponse: never accept an absent bound ID.
                    stream.send_data(vec![0u8; 5].into(), false).unwrap();
                    let mut trailers = http::HeaderMap::new();
                    trailers.insert("grpc-status", "0".parse().unwrap());
                    stream.send_trailers(trailers).unwrap();
                }
                // Drive the HTTP/2 connection; a rejected lookup cannot execute.
                while let Some(request) = connection.accept().await {
                    assert!(request.is_err(), "unexpected additional RPC");
                }
            });
            let mut value = no_auth_input();
            // Direct SDK fixture only; production parsing rejects loopback.
            value.endpoint = format!("http://127.0.0.1:{port}");
            let dispatched = AtomicBool::new(false);
            let result = timeout_at(
                Instant::now() + Duration::from_secs(2),
                execute(
                    value,
                    vec![],
                    &dispatched,
                    Instant::now() + Duration::from_secs(2),
                ),
            )
            .await
            .unwrap();
            assert_eq!(
                serde_json::to_string(&result).unwrap(),
                if denied {
                    "{\"failure\":\"access_denied\"}"
                } else {
                    "{\"failure\":\"target_replaced\"}"
                }
            );
            assert!(!dispatched.load(Ordering::Relaxed));
            peer.await.unwrap();
        }
    }

    #[test]
    fn denied_expired_and_revoked_grants_fail_without_raw_details() {
        for code in [tonic::Code::Unauthenticated, tonic::Code::PermissionDenied] {
            let status = tonic::Status::new(code, "fixture credentials must not escape");
            assert_eq!(status_code(&status, false), "access_denied");
            assert_eq!(status_code(&status, true), "access_denied");
        }
        assert_eq!(
            status_code(&tonic::Status::unavailable("disconnect"), true),
            "outcome_unconfirmed"
        );
    }

    #[tokio::test]
    async fn verified_https_rejects_plaintext_peer_without_fallback() {
        use tokio::io::AsyncWriteExt;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let peer = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let _ = socket.write_all(b"HTTP/1.1 200 OK\r\n\r\n").await;
        });
        let mut value = input();
        // Direct SDK fixture only. The production descriptor rejects loopback.
        value.endpoint = format!("https://localhost:{port}");
        let result = timeout_at(
            Instant::now() + Duration::from_secs(2),
            execute(
                value,
                vec![],
                &AtomicBool::new(false),
                Instant::now() + Duration::from_secs(2),
            ),
        )
        .await
        .unwrap();
        assert_eq!(
            serde_json::to_string(&result).unwrap(),
            "{\"failure\":\"transport_unavailable\"}"
        );
        peer.abort();
    }

    #[test]
    fn output_bound_is_enforced_before_extending_both_channels() {
        let mut stdout = vec![0; OUTPUT_LIMIT - 1];
        let mut stderr = vec![];
        assert!(append_bound(&mut stdout, &mut stderr, &[1], true).is_ok());
        assert_eq!(
            append_bound(&mut stdout, &mut stderr, &[2], false),
            Err("output_limit")
        );
        assert_eq!(stdout.len() + stderr.len(), OUTPUT_LIMIT);
    }

    #[test]
    fn failure_output_contains_only_allowlisted_category() {
        assert_eq!(
            serde_json::to_string(&failure("access_denied")).unwrap(),
            "{\"failure\":\"access_denied\"}"
        );
    }
}
