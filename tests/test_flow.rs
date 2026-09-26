use network_analyzer::flow::Flow;
use network_analyzer::packet::{Direction, parse_packet};
use network_analyzer::statistics::{ActiveIdleTracker, IatTracker, StreamingStats};

fn make_tcp_packet(
    src_ip: [u8; 4],
    dst_ip: [u8; 4],
    src_port: u16,
    dst_port: u16,
    seq: u32,
    ack: u32,
    flags: u16,
    payload: &[u8],
) -> Vec<u8> {
    let mut raw = Vec::new();
    // Ethernet
    raw.extend_from_slice(&[0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
    raw.extend_from_slice(&[0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb]);
    raw.extend_from_slice(&[0x08, 0x00]);

    // IPv4 (20 bytes)
    let total_len = 20 + 20 + payload.len() as u16;
    raw.push(0x45);
    raw.push(0x00);
    raw.extend_from_slice(&total_len.to_be_bytes());
    raw.extend_from_slice(&[0x12, 0x34]);
    raw.extend_from_slice(&[0x40, 0x00]);
    raw.push(64);
    raw.push(6); // TCP
    raw.extend_from_slice(&[0x00, 0x00]);
    raw.extend_from_slice(&src_ip);
    raw.extend_from_slice(&dst_ip);

    // TCP (20 bytes)
    raw.extend_from_slice(&src_port.to_be_bytes());
    raw.extend_from_slice(&dst_port.to_be_bytes());
    raw.extend_from_slice(&seq.to_be_bytes());
    raw.extend_from_slice(&ack.to_be_bytes());
    raw.extend_from_slice(&flags.to_be_bytes());
    raw.extend_from_slice(&65535u16.to_be_bytes()); // Window
    raw.extend_from_slice(&[0x00, 0x00]);
    raw.extend_from_slice(&[0x00, 0x00]);

    raw.extend_from_slice(payload);
    raw
}

#[test]
fn test_welford_stats() {
    let mut stats = StreamingStats::new();
    let values = [10.0, 20.0, 30.0, 40.0, 50.0];

    for &v in &values {
        stats.add(v);
    }

    assert_eq!(stats.count(), 5);
    assert_eq!(stats.min(), 10.0);
    assert_eq!(stats.max(), 50.0);
    assert_eq!(stats.sum(), 150.0);
    assert_eq!(stats.mean(), 30.0);
    assert_eq!(stats.variance(), 250.0); // Sample variance
    assert!((stats.std_dev() - 15.8113883).abs() < 1e-4);
}

#[test]
fn test_iat_tracker() {
    let mut iat = IatTracker::new();
    iat.update(1_000_000);
    iat.update(1_050_000); // 50,000 micros
    iat.update(1_150_000); // 100,000 micros
    iat.update(1_200_000); // 50,000 micros

    assert_eq!(iat.count(), 3);
    assert_eq!(iat.min(), 50_000.0);
    assert_eq!(iat.max(), 100_000.0);
    assert_eq!(iat.total(), 200_000.0);
    assert_eq!(iat.mean(), 200_000.0 / 3.0);
}

#[test]
fn test_active_idle_periods() {
    let mut tracker = ActiveIdleTracker::new(1_000_000); // 1 second threshold

    // First burst: t = 1.0s, 1.1s, 1.2s (duration = 0.2s = 200,000 µs)
    tracker.update(1_000_000);
    tracker.update(1_100_000);
    tracker.update(1_200_000);

    // Idle gap of 2.0s -> next packet at 3.2s
    tracker.update(3_200_000); // idle = 2,000,000 µs
    tracker.update(3_300_000); // second burst duration = 100,000 µs

    tracker.finalize();

    assert_eq!(tracker.idle_max(), 2_000_000.0);
    assert_eq!(tracker.idle_mean(), 2_000_000.0);
    assert_eq!(tracker.active_min(), 100_000.0);
    assert_eq!(tracker.active_max(), 200_000.0);
    assert_eq!(tracker.active_mean(), 150_000.0);
}

#[test]
fn test_flow_counters_and_direction() {
    let pkt1_bytes = make_tcp_packet([192, 168, 1, 100], [10, 0, 0, 1], 50000, 80, 1, 0, 0x5002, b"");
    let pkt1 = parse_packet(&pkt1_bytes, 1_000_000).unwrap();
    let mut flow = Flow::new(&pkt1);

    assert_eq!(flow.fwd_pkts, 1);
    assert_eq!(flow.bwd_pkts, 0);
    assert_eq!(flow.fwd_bytes, pkt1_bytes.len() as u64);
    assert_eq!(flow.bwd_bytes, 0);

    // Responder packet (backward)
    let pkt2_bytes = make_tcp_packet([10, 0, 0, 1], [192, 168, 1, 100], 80, 50000, 100, 2, 0x5012, b"");
    let pkt2 = parse_packet(&pkt2_bytes, 1_050_000).unwrap();
    assert_eq!(flow.direction_of(&pkt2), Direction::Backward);
    flow.update(&pkt2);

    assert_eq!(flow.fwd_pkts, 1);
    assert_eq!(flow.bwd_pkts, 1);
    assert_eq!(flow.bwd_bytes, pkt2_bytes.len() as u64);
}

#[test]
fn test_tcp_handshake_timing() {
    // 1. Client SYN at t = 1,000,000
    let syn_raw = make_tcp_packet([192, 168, 1, 10], [142, 250, 180, 1], 45000, 80, 100, 0, 0x5002, b"");
    let syn_pkt = parse_packet(&syn_raw, 1_000_000).unwrap();
    let mut flow = Flow::new(&syn_pkt);

    assert_eq!(flow.init_win_fwd, 65535);
    assert_eq!(flow.syn_time, Some(1_000_000));

    // 2. Server SYN-ACK at t = 1_040_000 (40ms synack delay)
    let synack_raw = make_tcp_packet([142, 250, 180, 1], [192, 168, 1, 10], 80, 45000, 200, 101, 0x5012, b"");
    let synack_pkt = parse_packet(&synack_raw, 1_040_000).unwrap();
    flow.update(&synack_pkt);

    assert_eq!(flow.init_win_bwd, 65535);
    assert_eq!(flow.synack_time, Some(1_040_000));
    assert_eq!(flow.synack_delay, Some(0.040));

    // 3. Client ACK at t = 1_055_000 (15ms ackdat delay, 55ms total tcprtt)
    let ack_raw = make_tcp_packet([192, 168, 1, 10], [142, 250, 180, 1], 45000, 80, 101, 201, 0x5010, b"");
    let ack_pkt = parse_packet(&ack_raw, 1_055_000).unwrap();
    flow.update(&ack_pkt);

    assert_eq!(flow.ack_time, Some(1_055_000));
    assert_eq!(flow.ackdat_delay, Some(0.015));
    assert!((flow.tcprtt.unwrap() - 0.055).abs() < 1e-6);
}

#[test]
fn test_tcp_flags_and_state_machine() {
    let p1 = make_tcp_packet([10, 0, 0, 1], [10, 0, 0, 2], 1234, 80, 1, 0, 0x5002, b""); // SYN
    let mut flow = Flow::new(&parse_packet(&p1, 100).unwrap());
    assert_eq!(flow.kdd_flag(), "S0"); // SYN only, no reply

    let p2 = make_tcp_packet([10, 0, 0, 2], [10, 0, 0, 1], 80, 1234, 1, 2, 0x5004, b""); // RST reply
    flow.update(&parse_packet(&p2, 120).unwrap());
    assert_eq!(flow.kdd_flag(), "REJ"); // SYN answered by RST
}

#[test]
fn test_application_service_detection() {
    // HTTP GET
    let http_get = make_tcp_packet([10, 0, 0, 1], [10, 0, 0, 2], 50000, 80, 1, 1, 0x5018, b"GET /index.html HTTP/1.1\r\n\r\n");
    let mut flow = Flow::new(&parse_packet(&http_get, 100).unwrap());

    assert_eq!(flow.service, "http");
    assert_eq!(flow.ct_flw_http_mthd, 1);
    assert_eq!(flow.trans_depth, 1);

    // HTTP Response with Content-Length
    let http_resp = make_tcp_packet(
        [10, 0, 0, 2], [10, 0, 0, 1], 80, 50000, 1, 1, 0x5018,
        b"HTTP/1.1 200 OK\r\nContent-Length: 1024\r\n\r\n[body]"
    );
    flow.update(&parse_packet(&http_resp, 150).unwrap());
    assert_eq!(flow.response_body_len, 1024);
}
