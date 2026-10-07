use network_analyzer::flow_table::FlowTable;
use network_analyzer::packet::parse_packet;
use network_analyzer::capture::CaptureSource;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn make_dummy_packet(src_port: u16, dst_port: u16, flags: u16) -> Vec<u8> {
    let mut raw = Vec::new();
    raw.extend_from_slice(&[0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
    raw.extend_from_slice(&[0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb]);
    raw.extend_from_slice(&[0x08, 0x00]);

    raw.push(0x45);
    raw.push(0x00);
    raw.extend_from_slice(&40u16.to_be_bytes());
    raw.extend_from_slice(&[0x12, 0x34, 0x40, 0x00]);
    raw.push(64);
    raw.push(6); // TCP
    raw.extend_from_slice(&[0x00, 0x00]);
    raw.extend_from_slice(&[192, 168, 1, 10]);
    raw.extend_from_slice(&[10, 0, 0, 1]);

    raw.extend_from_slice(&src_port.to_be_bytes());
    raw.extend_from_slice(&dst_port.to_be_bytes());
    raw.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
    raw.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
    raw.extend_from_slice(&flags.to_be_bytes());
    raw.extend_from_slice(&[0xff, 0xff, 0x00, 0x00, 0x00, 0x00]);

    raw
}

#[test]
fn test_flow_table_expiration_timeout() {
    let mut table = FlowTable::new(1000, 10, None, 60); // 10s timeout

    let raw = make_dummy_packet(12345, 80, 0x5002);
    let pkt = parse_packet(&raw, 10_000_000).unwrap(); // t = 10s

    let evicted = table.process_packet(&pkt);
    assert!(evicted.is_none());
    assert_eq!(table.active_flow_count(), 1);

    // At t = 15s (5s elapsed) -> not expired
    let expired = table.check_timeouts(15_000_000);
    assert_eq!(expired.len(), 0);
    assert_eq!(table.active_flow_count(), 1);

    // At t = 21s (11s elapsed) -> expired!
    let expired = table.check_timeouts(21_000_000);
    assert_eq!(expired.len(), 1);
    assert_eq!(table.active_flow_count(), 0);
    assert_eq!(table.total_completed_flows(), 1);
}

#[test]
fn test_max_flows_capacity_lru_eviction() {
    let max_capacity = 3;
    let mut table = FlowTable::new(max_capacity, 300, None, 60);

    // Insert 3 flows
    for port in 1001..=1003 {
        let raw = make_dummy_packet(port, 80, 0x5002);
        let pkt = parse_packet(&raw, 1_000_000).unwrap();
        let evicted = table.process_packet(&pkt);
        assert!(evicted.is_none());
    }
    assert_eq!(table.active_flow_count(), 3);

    // Insert 4th flow -> should trigger LRU eviction of oldest flow (port 1001)
    let raw4 = make_dummy_packet(1004, 80, 0x5002);
    let pkt4 = parse_packet(&raw4, 1_500_000).unwrap();
    let evicted = table.process_packet(&pkt4);

    assert!(evicted.is_some());
    assert_eq!(evicted.unwrap().src_port, 1001);
    assert_eq!(table.active_flow_count(), 3);
}

#[test]
fn test_host_service_rolling_window_rates() {
    let mut table = FlowTable::new(1000, 5, None, 60);

    // Generate 5 connections to host 10.0.0.1, 4 to port 80 (HTTP) and 1 to port 22 (SSH)
    for i in 0..4 {
        let raw = make_dummy_packet(20000 + i, 80, 0x5002);
        let pkt = parse_packet(&raw, (10 + i as u64) * 1_000_000).unwrap();
        table.process_packet(&pkt);
    }
    let raw_ssh = make_dummy_packet(30000, 22, 0x5002);
    let pkt_ssh = parse_packet(&raw_ssh, 15_000_000).unwrap();
    table.process_packet(&pkt_ssh);

    // Expire these 5 flows into host history
    table.check_timeouts(25_000_000);
    assert_eq!(table.total_completed_flows(), 5);

    // Now a new flow arrives to 10.0.0.1:80
    let raw_new = make_dummy_packet(40000, 80, 0x5002);
    let pkt_new = parse_packet(&raw_new, 26_000_000).unwrap();
    table.process_packet(&pkt_new);

    // Expire the new flow and verify host statistics
    let flushed = table.flush_all();
    assert_eq!(flushed.len(), 1);
    let record = &flushed[0];

    // Same host count should be 5
    assert_eq!(record.kdd_dst_host_count, 5);
    // Same service count should be 4 (out of 5)
    assert_eq!(record.kdd_dst_host_srv_count, 4);
    assert_eq!(record.kdd_dst_host_same_srv_rate, 4.0 / 5.0);
    assert_eq!(record.kdd_dst_host_diff_srv_rate, 1.0 / 5.0);
}

#[test]
fn test_pcap_write_and_replay() {
    let pcap_path = Path::new("tests/test_traffic.pcap");

    // Write a valid pcap file format manually to ensure test determinism without external files
    let mut f = File::create(pcap_path).expect("Create test pcap");

    // PCAP Global Header (24 bytes)
    // magic_number: 0xa1b2c3d4, v2.4, thiszone=0, sigfigs=0, snaplen=65535, network=1 (Ethernet)
    f.write_all(&0xa1b2c3d4u32.to_ne_bytes()).unwrap();
    f.write_all(&2u16.to_ne_bytes()).unwrap(); // major
    f.write_all(&4u16.to_ne_bytes()).unwrap(); // minor
    f.write_all(&0i32.to_ne_bytes()).unwrap(); // thiszone
    f.write_all(&0u32.to_ne_bytes()).unwrap(); // sigfigs
    f.write_all(&65535u32.to_ne_bytes()).unwrap(); // snaplen
    f.write_all(&1u32.to_ne_bytes()).unwrap(); // network: LINKTYPE_ETHERNET

    // Write 3 packets
    for i in 0..3 {
        let pkt_data = make_dummy_packet(10000 + i, 80, 0x5002);
        let len = pkt_data.len() as u32;

        // Packet Header (16 bytes)
        let ts_sec = 1700000000u32 + i as u32;
        let ts_usec = 500000u32;
        f.write_all(&ts_sec.to_ne_bytes()).unwrap();
        f.write_all(&ts_usec.to_ne_bytes()).unwrap();
        f.write_all(&len.to_ne_bytes()).unwrap(); // incl_len
        f.write_all(&len.to_ne_bytes()).unwrap(); // orig_len
        f.write_all(&pkt_data).unwrap();
    }
    f.flush().unwrap();
    drop(f);

    // Read back through CaptureSource::open_offline
    let mut capture = CaptureSource::open_offline(pcap_path, None).expect("Open offline pcap");
    let mut count = 0;
    while let Ok(Some((ts, data))) = capture.next_packet() {
        count += 1;
        assert!(ts > 0);
        assert!(!data.is_empty());
    }
    assert_eq!(count, 3);

    let _ = std::fs::remove_file(pcap_path);
}

#[test]
fn test_tcp_lifecycle_aware_timeouts() {
    // Flow table with 1s base timeout (e.g. --flow-timeout 1)
    let mut table = FlowTable::new(1000, 1, None, 60);

    // 1. Connection A: Unanswered SYN at t = 0
    let p_syn = make_dummy_packet(40001, 80, 0x5002); // SYN
    let pkt_syn = parse_packet(&p_syn, 0).unwrap();
    table.process_packet(&pkt_syn);

    // At t = 2s: should NOT be expired (handshake timeout is 5s)
    let exp_2s = table.check_timeouts(2_000_000);
    assert_eq!(exp_2s.len(), 0, "Unanswered SYN should not expire after only 2s");
    assert_eq!(table.active_flow_count(), 1);

    // At t = 6s: should be expired (exceeded 5s handshake timeout)
    let exp_6s = table.check_timeouts(6_000_000);
    assert_eq!(exp_6s.len(), 1, "Unanswered SYN should expire after 5s");
    assert_eq!(table.active_flow_count(), 0);

    // 2. Connection B: Completed connection with teardown
    // t = 10.0s: SYN (client -> server)
    let p1 = make_dummy_packet(50001, 80, 0x5002);
    table.process_packet(&parse_packet(&p1, 10_000_000).unwrap());

    // t = 10.1s: SYN-ACK (server -> client)
    let mut raw_sa = make_dummy_packet(80, 50001, 0x5012); // SYN|ACK
    // reverse IPs to make it true backward packet
    raw_sa[26..30].copy_from_slice(&[10, 0, 0, 1]);
    raw_sa[30..34].copy_from_slice(&[192, 168, 1, 10]);
    table.process_packet(&parse_packet(&raw_sa, 10_100_000).unwrap());

    // t = 10.2s: ACK (client -> server)
    let p3 = make_dummy_packet(50001, 80, 0x5010); // ACK
    table.process_packet(&parse_packet(&p3, 10_200_000).unwrap());

    // t = 10.3s: DATA (client -> server)
    let p4 = make_dummy_packet(50001, 80, 0x5018); // PSH|ACK
    table.process_packet(&parse_packet(&p4, 10_300_000).unwrap());

    // At t = 12s: 1.7s since last packet. Even with base_timeout=1s, established connection remains open!
    let exp_12s = table.check_timeouts(12_000_000);
    assert_eq!(exp_12s.len(), 0, "Established TCP connection should not expire after 1.7s");
    assert_eq!(table.active_flow_count(), 1);

    // t = 12.5s: FIN (client -> server)
    let p5 = make_dummy_packet(50001, 80, 0x5011); // FIN|ACK
    table.process_packet(&parse_packet(&p5, 12_500_000).unwrap());

    // t = 12.6s: FIN (server -> client)
    let mut p6 = make_dummy_packet(80, 50001, 0x5011); // FIN|ACK
    p6[26..30].copy_from_slice(&[10, 0, 0, 1]);
    p6[30..34].copy_from_slice(&[192, 168, 1, 10]);
    table.process_packet(&parse_packet(&p6, 12_600_000).unwrap());

    // Both FINs seen -> connection is terminated.
    // Linger is 500ms. At t = 12.8s (200ms after last packet): not yet expired.
    let exp_12_8s = table.check_timeouts(12_800_000);
    assert_eq!(exp_12_8s.len(), 0, "Terminated flow should linger for 500ms");

    // At t = 13.2s (600ms after last packet): should expire promptly!
    let exp_13_2s = table.check_timeouts(13_200_000);
    assert_eq!(exp_13_2s.len(), 1, "Terminated flow should be flushed after 500ms linger");
    let rec = &exp_13_2s[0];
    assert!(rec.total_fwd_packets >= 3);
    assert!(rec.total_bwd_packets >= 2);
    assert_eq!(rec.syn_flag_count, 2);
    assert_eq!(rec.fin_flag_count, 2);
    assert!(rec.flow_duration_micros > 0);
}
