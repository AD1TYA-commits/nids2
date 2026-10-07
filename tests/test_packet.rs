use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use network_analyzer::packet::{
    parse_packet, FlowKey, L4Header, PacketParseError,
};

#[test]
fn test_flow_key_normalization() {
    let ip_a = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 10));
    let ip_b = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));
    let port_a = 54321;
    let port_b = 80;
    let proto = 6;

    let key_fwd = FlowKey::canonical(ip_a, ip_b, port_a, port_b, proto);
    let key_bwd = FlowKey::canonical(ip_b, ip_a, port_b, port_a, proto);

    assert_eq!(key_fwd, key_bwd);
    assert_eq!(key_fwd.ip_a, ip_b); // 10.0.0.1 < 192.168.1.10
    assert_eq!(key_fwd.ip_b, ip_a);
    assert_eq!(key_fwd.port_a, port_b);
    assert_eq!(key_fwd.port_b, port_a);

    // Identical IPs, different ports
    let key_same_ip_fwd = FlowKey::canonical(ip_a, ip_a, 5000, 6000, 17);
    let key_same_ip_bwd = FlowKey::canonical(ip_a, ip_a, 6000, 5000, 17);
    assert_eq!(key_same_ip_fwd, key_same_ip_bwd);
    assert_eq!(key_same_ip_fwd.port_a, 5000);
    assert_eq!(key_same_ip_fwd.port_b, 6000);

    // IPv6 normalization
    let ip6_a = IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 1));
    let ip6_b = IpAddr::V6(Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 1));
    let key6_fwd = FlowKey::canonical(ip6_a, ip6_b, 443, 60123, 6);
    let key6_bwd = FlowKey::canonical(ip6_b, ip6_a, 60123, 443, 6);
    assert_eq!(key6_fwd, key6_bwd);
}

#[test]
fn test_ipv4_tcp_parsing() {
    let mut raw = Vec::new();
    // Ethernet header (14 bytes)
    raw.extend_from_slice(&[0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
    raw.extend_from_slice(&[0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb]);
    raw.extend_from_slice(&[0x08, 0x00]); // IPv4

    // IPv4 header (20 bytes)
    raw.push(0x45); // Version 4, IHL 5
    raw.push(0x00); // DSCP
    raw.extend_from_slice(&[0x00, 0x34]); // Total length: 52 bytes
    raw.extend_from_slice(&[0x1a, 0x2b]); // Identification
    raw.extend_from_slice(&[0x40, 0x00]); // Flags: DF
    raw.push(64); // TTL
    raw.push(6); // TCP
    raw.extend_from_slice(&[0x00, 0x00]); // Checksum
    raw.extend_from_slice(&[192, 168, 1, 50]);
    raw.extend_from_slice(&[10, 0, 0, 1]);

    // TCP header (20 bytes)
    raw.extend_from_slice(&8080u16.to_be_bytes()); // src port
    raw.extend_from_slice(&443u16.to_be_bytes()); // dst port
    raw.extend_from_slice(&100000u32.to_be_bytes()); // seq
    raw.extend_from_slice(&200000u32.to_be_bytes()); // ack
    raw.extend_from_slice(&[0x50, 0x18]); // Data offset 5, PSH + ACK
    raw.extend_from_slice(&16384u16.to_be_bytes()); // window size
    raw.extend_from_slice(&[0x00, 0x00]); // Checksum
    raw.extend_from_slice(&[0x00, 0x00]); // Urgent pointer

    // Payload (12 bytes)
    raw.extend_from_slice(b"HELLO PACKET");

    let parsed = parse_packet(&raw, 1000000).expect("Failed to parse valid IPv4 TCP");
    assert_eq!(parsed.src_ip, IpAddr::V4(Ipv4Addr::new(192, 168, 1, 50)));
    assert_eq!(parsed.dst_ip, IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)));
    assert_eq!(parsed.src_port, 8080);
    assert_eq!(parsed.dst_port, 443);
    assert_eq!(parsed.protocol, 6);
    assert_eq!(parsed.ttl, 64);
    assert_eq!(parsed.ip_header_len, 20);
    assert_eq!(parsed.l4_header_len, 20);
    assert_eq!(parsed.payload, b"HELLO PACKET");

    if let L4Header::Tcp { flags, seq, ack, window_size, .. } = parsed.l4 {
        assert!(flags.psh);
        assert!(flags.ack);
        assert!(!flags.syn);
        assert!(!flags.fin);
        assert_eq!(seq, 100000);
        assert_eq!(ack, 200000);
        assert_eq!(window_size, 16384);
    } else {
        panic!("Expected L4Header::Tcp");
    }
}

#[test]
fn test_ipv6_udp_parsing() {
    let mut raw = Vec::new();
    // Ethernet header
    raw.extend_from_slice(&[0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
    raw.extend_from_slice(&[0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb]);
    raw.extend_from_slice(&[0x86, 0xdd]); // IPv6

    // IPv6 header (40 bytes)
    raw.extend_from_slice(&[0x60, 0x00, 0x00, 0x00]); // Version 6, TC 0, FL 0
    raw.extend_from_slice(&16u16.to_be_bytes()); // Payload len (8 bytes UDP + 8 bytes data)
    raw.push(17); // Next header: UDP
    raw.push(128); // Hop limit
    raw.extend_from_slice(&[0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]); // src
    raw.extend_from_slice(&[0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]); // dst

    // UDP header (8 bytes)
    raw.extend_from_slice(&5353u16.to_be_bytes()); // src port
    raw.extend_from_slice(&53u16.to_be_bytes()); // dst port
    raw.extend_from_slice(&16u16.to_be_bytes()); // length
    raw.extend_from_slice(&[0x00, 0x00]); // checksum

    // Payload (8 bytes)
    raw.extend_from_slice(b"DNSQUERY");

    let parsed = parse_packet(&raw, 2000000).expect("Failed to parse valid IPv6 UDP");
    assert_eq!(parsed.protocol, 17);
    assert_eq!(parsed.src_port, 5353);
    assert_eq!(parsed.dst_port, 53);
    assert_eq!(parsed.ttl, 128);
    assert_eq!(parsed.ip_header_len, 40);
    assert_eq!(parsed.l4_header_len, 8);
    assert_eq!(parsed.payload, b"DNSQUERY");

    if let L4Header::Udp { src_port, dst_port, length } = parsed.l4 {
        assert_eq!(src_port, 5353);
        assert_eq!(dst_port, 53);
        assert_eq!(length, 16);
    } else {
        panic!("Expected L4Header::Udp");
    }
}

#[test]
fn test_icmp_parsing() {
    let mut raw = Vec::new();
    raw.extend_from_slice(&[0x00; 12]);
    raw.extend_from_slice(&[0x08, 0x00]); // IPv4

    // IPv4 header (20 bytes)
    raw.push(0x45);
    raw.push(0x00);
    raw.extend_from_slice(&28u16.to_be_bytes());
    raw.extend_from_slice(&[0x00; 4]);
    raw.push(64);
    raw.push(1); // ICMP
    raw.extend_from_slice(&[0x00, 0x00]);
    raw.extend_from_slice(&[10, 0, 0, 1]);
    raw.extend_from_slice(&[10, 0, 0, 2]);

    // ICMP header (4 bytes)
    raw.push(8); // Type 8: Echo request
    raw.push(0); // Code 0
    raw.extend_from_slice(&[0x00, 0x00]); // Checksum
    raw.extend_from_slice(b"ECHO");

    let parsed = parse_packet(&raw, 3000000).expect("Failed to parse ICMP");
    assert_eq!(parsed.protocol, 1);
    if let L4Header::Icmp { icmp_type, code } = parsed.l4 {
        assert_eq!(icmp_type, 8);
        assert_eq!(code, 0);
    } else {
        panic!("Expected L4Header::Icmp");
    }
}

#[test]
fn test_arp_parsing() {
    let mut raw = Vec::new();
    raw.extend_from_slice(&[0xff; 6]);
    raw.extend_from_slice(&[0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
    raw.extend_from_slice(&[0x08, 0x06]); // ARP

    // ARP payload (28 bytes)
    raw.extend_from_slice(&1u16.to_be_bytes()); // Ethernet HW
    raw.extend_from_slice(&0x0800u16.to_be_bytes()); // IPv4 proto
    raw.push(6); // HW size
    raw.push(4); // Proto size
    raw.extend_from_slice(&1u16.to_be_bytes()); // Opcode 1 (Request)
    raw.extend_from_slice(&[0x00, 0x11, 0x22, 0x33, 0x44, 0x55]); // sender MAC
    raw.extend_from_slice(&[192, 168, 1, 1]); // sender IP
    raw.extend_from_slice(&[0x00; 6]); // target MAC
    raw.extend_from_slice(&[192, 168, 1, 2]); // target IP

    let parsed = parse_packet(&raw, 4000000).expect("Failed to parse ARP");
    assert_eq!(parsed.protocol, 254);
    assert_eq!(parsed.src_ip, IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)));
    assert_eq!(parsed.dst_ip, IpAddr::V4(Ipv4Addr::new(192, 168, 1, 2)));
    if let L4Header::Arp { opcode } = parsed.l4 {
        assert_eq!(opcode, 1);
    } else {
        panic!("Expected L4Header::Arp");
    }
}

#[test]
fn test_vlan_tagged_packet() {
    let mut raw = Vec::new();
    raw.extend_from_slice(&[0x00; 12]);
    raw.extend_from_slice(&[0x81, 0x00]); // 802.1Q VLAN
    raw.extend_from_slice(&[0x00, 0x64]); // VLAN ID 100
    raw.extend_from_slice(&[0x08, 0x00]); // Inner: IPv4

    // IPv4 header (20 bytes)
    raw.push(0x45);
    raw.push(0x00);
    raw.extend_from_slice(&28u16.to_be_bytes());
    raw.extend_from_slice(&[0x00; 4]);
    raw.push(64);
    raw.push(17); // UDP
    raw.extend_from_slice(&[0x00, 0x00]);
    raw.extend_from_slice(&[172, 16, 0, 1]);
    raw.extend_from_slice(&[172, 16, 0, 2]);

    // UDP header (8 bytes)
    raw.extend_from_slice(&1234u16.to_be_bytes());
    raw.extend_from_slice(&5678u16.to_be_bytes());
    raw.extend_from_slice(&8u16.to_be_bytes());
    raw.extend_from_slice(&[0x00, 0x00]);

    let parsed = parse_packet(&raw, 5000000).expect("Failed to parse VLAN tagged packet");
    assert_eq!(parsed.src_port, 1234);
    assert_eq!(parsed.dst_port, 5678);
    assert_eq!(parsed.protocol, 17);
}

#[test]
fn test_malformed_truncated_packets() {
    // Empty buffer
    assert!(matches!(parse_packet(&[], 0), Err(PacketParseError::Truncated(_))));

    // Too short for ethernet
    let short_bytes = [0x00, 0x11, 0x22];
    assert!(parse_packet(&short_bytes, 0).is_err());

    // Truncated IPv4
    let truncated_ip = [
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0x08, 0x00,
        0x45, 0x00, 0x00, 0x20, // only 4 bytes of IP header
    ];
    assert!(matches!(parse_packet(&truncated_ip, 0), Err(PacketParseError::Truncated(_))));

    // Invalid IP version (version = 5)
    let invalid_version = [
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0x08, 0x00,
        0x55, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 64, 6, 0x00, 0x00,
        1, 1, 1, 1, 2, 2, 2, 2,
    ];
    assert!(matches!(parse_packet(&invalid_version, 0), Err(PacketParseError::InvalidIpVersion(5))));
}
