use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FlowKey {
    pub ip_a: IpAddr,
    pub ip_b: IpAddr,
    pub port_a: u16,
    pub port_b: u16,
    pub protocol: u8,
}

impl FlowKey {
    /// Create a canonical bidirectional FlowKey.
    /// Regardless of whether A->B or B->A is passed, the resulting FlowKey is identical.
    pub fn canonical(src_ip: IpAddr, dst_ip: IpAddr, src_port: u16, dst_port: u16, protocol: u8) -> Self {
        if src_ip < dst_ip || (src_ip == dst_ip && src_port <= dst_port) {
            Self {
                ip_a: src_ip,
                ip_b: dst_ip,
                port_a: src_port,
                port_b: dst_port,
                protocol,
            }
        } else {
            Self {
                ip_a: dst_ip,
                ip_b: src_ip,
                port_a: dst_port,
                port_b: src_port,
                protocol,
            }
        }
    }
}

impl fmt::Display for FlowKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{} <-> {}:{} (proto {})",
            self.ip_a, self.port_a, self.ip_b, self.port_b, self.protocol
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Backward,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TcpFlags {
    pub fin: bool,
    pub syn: bool,
    pub rst: bool,
    pub psh: bool,
    pub ack: bool,
    pub urg: bool,
    pub ece: bool,
    pub cwr: bool,
    pub ns: bool,
}

impl TcpFlags {
    pub fn from_bits(bits: u16) -> Self {
        Self {
            fin: (bits & 0x0001) != 0,
            syn: (bits & 0x0002) != 0,
            rst: (bits & 0x0004) != 0,
            psh: (bits & 0x0008) != 0,
            ack: (bits & 0x0010) != 0,
            urg: (bits & 0x0020) != 0,
            ece: (bits & 0x0040) != 0,
            cwr: (bits & 0x0080) != 0,
            ns: (bits & 0x0100) != 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum L4Header {
    Tcp {
        src_port: u16,
        dst_port: u16,
        seq: u32,
        ack: u32,
        header_len: usize,
        flags: TcpFlags,
        window_size: u16,
        urgent_ptr: u16,
    },
    Udp {
        src_port: u16,
        dst_port: u16,
        length: u16,
    },
    Icmp {
        icmp_type: u8,
        code: u8,
    },
    Arp {
        opcode: u16,
    },
    Other {
        protocol: u8,
    },
}

#[derive(Debug, Clone)]
pub struct ParsedPacket<'a> {
    pub timestamp_micros: u64,
    pub wire_len: usize,
    pub src_ip: IpAddr,
    pub dst_ip: IpAddr,
    pub src_port: u16,
    pub dst_port: u16,
    pub protocol: u8,
    pub ttl: u8,
    pub ip_header_len: usize,
    pub l4_header_len: usize,
    pub payload: &'a [u8],
    pub l4: L4Header,
    pub is_fragmented: bool,
    pub fragment_offset: u16,
    pub wrong_fragment: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PacketParseError {
    Truncated(&'static str),
    UnsupportedEtherType(u16),
    InvalidIpVersion(u8),
    Malformed(&'static str),
}

impl fmt::Display for PacketParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PacketParseError::Truncated(msg) => write!(f, "Truncated packet: {}", msg),
            PacketParseError::UnsupportedEtherType(e) => write!(f, "Unsupported EtherType: 0x{:04x}", e),
            PacketParseError::InvalidIpVersion(v) => write!(f, "Invalid IP version: {}", v),
            PacketParseError::Malformed(msg) => write!(f, "Malformed packet: {}", msg),
        }
    }
}

impl std::error::Error for PacketParseError {}

/// Parse raw captured packet bytes into a structured ParsedPacket.
/// Supports Ethernet II, IEEE 802.1Q (VLAN), Linux SLL (cooked capture), and raw IP packets.
pub fn parse_packet<'a>(
    raw_data: &'a [u8],
    timestamp_micros: u64,
) -> Result<ParsedPacket<'a>, PacketParseError> {
    if raw_data.is_empty() {
        return Err(PacketParseError::Truncated("Empty packet buffer"));
    }

    let wire_len = raw_data.len();
    let mut offset = 0;

    // Detect framing (Ethernet, Linux SLL, or Raw IP)
    let ether_type = if raw_data.len() >= 14 && is_likely_ethernet(&raw_data[0..14]) {
        let mut ethertype = u16::from_be_bytes([raw_data[12], raw_data[13]]);
        offset = 14;

        // Handle 802.1Q VLAN tagging
        while ethertype == 0x8100 || ethertype == 0x88A8 {
            if raw_data.len() < offset + 4 {
                return Err(PacketParseError::Truncated("VLAN header truncated"));
            }
            ethertype = u16::from_be_bytes([raw_data[offset + 2], raw_data[offset + 3]]);
            offset += 4;
        }
        ethertype
    } else if raw_data.len() >= 16 && is_linux_sll(&raw_data[0..16]) {
        // Linux SLL: 16 bytes header, protocol at bytes 14..16
        let ethertype = u16::from_be_bytes([raw_data[14], raw_data[15]]);
        offset = 16;
        ethertype
    } else {
        // Assume raw IP (0x45..0x4F for IPv4, 0x60..0x6F for IPv6)
        let first_nibble = raw_data[0] >> 4;
        if first_nibble == 4 {
            0x0800
        } else if first_nibble == 6 {
            0x86DD
        } else {
            return Err(PacketParseError::Malformed("Unknown link layer framing"));
        }
    };

    match ether_type {
        0x0800 => parse_ipv4(&raw_data[offset..], timestamp_micros, wire_len),
        0x86DD => parse_ipv6(&raw_data[offset..], timestamp_micros, wire_len),
        0x0806 => parse_arp(&raw_data[offset..], timestamp_micros, wire_len),
        other => Err(PacketParseError::UnsupportedEtherType(other)),
    }
}

fn is_likely_ethernet(hdr: &[u8]) -> bool {
    let et = u16::from_be_bytes([hdr[12], hdr[13]]);
    et >= 0x0600 || et == 0x8100
}

fn is_linux_sll(hdr: &[u8]) -> bool {
    let pkt_type = u16::from_be_bytes([hdr[0], hdr[1]]);
    let arphrd = u16::from_be_bytes([hdr[2], hdr[3]]);
    pkt_type <= 4 && (arphrd == 1 || arphrd == 772 || arphrd == 824)
}

fn parse_ipv4<'a>(
    ip_data: &'a [u8],
    timestamp_micros: u64,
    wire_len: usize,
) -> Result<ParsedPacket<'a>, PacketParseError> {
    if ip_data.len() < 20 {
        return Err(PacketParseError::Truncated("IPv4 header smaller than 20 bytes"));
    }

    let version = ip_data[0] >> 4;
    if version != 4 {
        return Err(PacketParseError::InvalidIpVersion(version));
    }

    let ihl = (ip_data[0] & 0x0F) as usize * 4;
    if ihl < 20 || ip_data.len() < ihl {
        return Err(PacketParseError::Malformed("IPv4 IHL invalid or truncated"));
    }

    let total_len = u16::from_be_bytes([ip_data[2], ip_data[3]]) as usize;
    let effective_len = if total_len > 0 && total_len <= ip_data.len() {
        total_len
    } else {
        ip_data.len()
    };

    let flags_frag = u16::from_be_bytes([ip_data[6], ip_data[7]]);
    let mf = (flags_frag & 0x2000) != 0;
    let frag_offset = (flags_frag & 0x1FFF) * 8;
    let is_fragmented = mf || frag_offset > 0;
    let wrong_fragment = frag_offset > 0 && (frag_offset % 8 != 0);

    let ttl = ip_data[8];
    let protocol = ip_data[9];
    let src_ip = IpAddr::V4(Ipv4Addr::new(ip_data[12], ip_data[13], ip_data[14], ip_data[15]));
    let dst_ip = IpAddr::V4(Ipv4Addr::new(ip_data[16], ip_data[17], ip_data[18], ip_data[19]));

    let l4_data = &ip_data[ihl..effective_len];

    let (l4, l4_header_len, payload) = parse_l4(protocol, l4_data)?;

    Ok(ParsedPacket {
        timestamp_micros,
        wire_len,
        src_ip,
        dst_ip,
        src_port: get_src_port(&l4),
        dst_port: get_dst_port(&l4),
        protocol,
        ttl,
        ip_header_len: ihl,
        l4_header_len,
        payload,
        l4,
        is_fragmented,
        fragment_offset: frag_offset,
        wrong_fragment,
    })
}

fn parse_ipv6<'a>(
    ip_data: &'a [u8],
    timestamp_micros: u64,
    wire_len: usize,
) -> Result<ParsedPacket<'a>, PacketParseError> {
    if ip_data.len() < 40 {
        return Err(PacketParseError::Truncated("IPv6 header smaller than 40 bytes"));
    }

    let version = ip_data[0] >> 4;
    if version != 6 {
        return Err(PacketParseError::InvalidIpVersion(version));
    }

    let payload_len = u16::from_be_bytes([ip_data[4], ip_data[5]]) as usize;
    let next_header = ip_data[6];
    let ttl = ip_data[7]; // Hop limit acts as TTL

    let mut src_bytes = [0u8; 16];
    src_bytes.copy_from_slice(&ip_data[8..24]);
    let src_ip = IpAddr::V6(Ipv6Addr::from(src_bytes));

    let mut dst_bytes = [0u8; 16];
    dst_bytes.copy_from_slice(&ip_data[24..40]);
    let dst_ip = IpAddr::V6(Ipv6Addr::from(dst_bytes));

    let header_len = 40;
    let effective_len = if payload_len > 0 && (header_len + payload_len) <= ip_data.len() {
        header_len + payload_len
    } else {
        ip_data.len()
    };

    let l4_data = &ip_data[header_len..effective_len];
    let (l4, l4_header_len, payload) = parse_l4(next_header, l4_data)?;

    Ok(ParsedPacket {
        timestamp_micros,
        wire_len,
        src_ip,
        dst_ip,
        src_port: get_src_port(&l4),
        dst_port: get_dst_port(&l4),
        protocol: next_header,
        ttl,
        ip_header_len: header_len,
        l4_header_len,
        payload,
        l4,
        is_fragmented: false,
        fragment_offset: 0,
        wrong_fragment: false,
    })
}

fn parse_arp<'a>(
    arp_data: &'a [u8],
    timestamp_micros: u64,
    wire_len: usize,
) -> Result<ParsedPacket<'a>, PacketParseError> {
    if arp_data.len() < 28 {
        return Err(PacketParseError::Truncated("ARP header smaller than 28 bytes"));
    }

    let opcode = u16::from_be_bytes([arp_data[6], arp_data[7]]);
    let sender_ip = IpAddr::V4(Ipv4Addr::new(arp_data[14], arp_data[15], arp_data[16], arp_data[17]));
    let target_ip = IpAddr::V4(Ipv4Addr::new(arp_data[24], arp_data[25], arp_data[26], arp_data[27]));

    Ok(ParsedPacket {
        timestamp_micros,
        wire_len,
        src_ip: sender_ip,
        dst_ip: target_ip,
        src_port: 0,
        dst_port: 0,
        protocol: 254, // Custom protocol identifier for ARP
        ttl: 64,
        ip_header_len: 28,
        l4_header_len: 0,
        payload: &arp_data[28..],
        l4: L4Header::Arp { opcode },
        is_fragmented: false,
        fragment_offset: 0,
        wrong_fragment: false,
    })
}

fn parse_l4<'a>(
    protocol: u8,
    data: &'a [u8],
) -> Result<(L4Header, usize, &'a [u8]), PacketParseError> {
    match protocol {
        6 => {
            // TCP
            if data.len() < 20 {
                return Err(PacketParseError::Truncated("TCP header truncated"));
            }
            let src_port = u16::from_be_bytes([data[0], data[1]]);
            let dst_port = u16::from_be_bytes([data[2], data[3]]);
            let seq = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
            let ack = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);

            let data_offset = ((data[12] >> 4) as usize) * 4;
            if data_offset < 20 || data.len() < data_offset {
                return Err(PacketParseError::Malformed("Invalid TCP data offset"));
            }

            let flags_raw = ((data[12] as u16 & 0x0F) << 8) | (data[13] as u16);
            let flags = TcpFlags::from_bits(flags_raw);
            let window_size = u16::from_be_bytes([data[14], data[15]]);
            let urgent_ptr = u16::from_be_bytes([data[18], data[19]]);

            let payload = &data[data_offset..];
            Ok((
                L4Header::Tcp {
                    src_port,
                    dst_port,
                    seq,
                    ack,
                    header_len: data_offset,
                    flags,
                    window_size,
                    urgent_ptr,
                },
                data_offset,
                payload,
            ))
        }
        17 => {
            // UDP
            if data.len() < 8 {
                return Err(PacketParseError::Truncated("UDP header truncated"));
            }
            let src_port = u16::from_be_bytes([data[0], data[1]]);
            let dst_port = u16::from_be_bytes([data[2], data[3]]);
            let length = u16::from_be_bytes([data[4], data[5]]);

            let payload_end = if (length as usize) >= 8 && (length as usize) <= data.len() {
                length as usize
            } else {
                data.len()
            };

            let payload = &data[8..payload_end];
            Ok((L4Header::Udp { src_port, dst_port, length }, 8, payload))
        }
        1 | 58 => {
            // ICMP / ICMPv6
            if data.len() < 4 {
                return Err(PacketParseError::Truncated("ICMP header truncated"));
            }
            let icmp_type = data[0];
            let code = data[1];
            let payload = if data.len() > 4 { &data[4..] } else { &[] };
            Ok((L4Header::Icmp { icmp_type, code }, 4, payload))
        }
        other => Ok((L4Header::Other { protocol: other }, 0, data)),
    }
}

fn get_src_port(l4: &L4Header) -> u16 {
    match l4 {
        L4Header::Tcp { src_port, .. } => *src_port,
        L4Header::Udp { src_port, .. } => *src_port,
        L4Header::Icmp { icmp_type, .. } => *icmp_type as u16,
        _ => 0,
    }
}

fn get_dst_port(l4: &L4Header) -> u16 {
    match l4 {
        L4Header::Tcp { dst_port, .. } => *dst_port,
        L4Header::Udp { dst_port, .. } => *dst_port,
        L4Header::Icmp { code, .. } => *code as u16,
        _ => 0,
    }
}
