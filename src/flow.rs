use std::net::IpAddr;
use crate::packet::{Direction, FlowKey, L4Header, ParsedPacket, TcpFlags};
use crate::statistics::{ActiveIdleTracker, IatTracker, StreamingStats};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcpConnState {
    Start,
    SynSent,
    SynRcvd,
    Established,
    FinWait1,
    FinWait2,
    CloseWait,
    Closing,
    LastAck,
    Closed,
    Reset,
}

#[derive(Debug, Clone)]
pub struct Flow {
    pub key: FlowKey,
    pub orig_src_ip: IpAddr,
    pub orig_dst_ip: IpAddr,
    pub orig_src_port: u16,
    pub orig_dst_port: u16,
    pub protocol: u8,

    pub start_time: u64,
    pub last_time: u64,

    // Packet and Byte Counters
    pub fwd_pkts: u64,
    pub bwd_pkts: u64,
    pub fwd_bytes: u64,
    pub bwd_bytes: u64,
    pub fwd_payload_bytes: u64,
    pub bwd_payload_bytes: u64,
    pub fwd_header_len: u64,
    pub bwd_header_len: u64,

    // Streaming Statistics
    pub fwd_len_stats: StreamingStats,
    pub bwd_len_stats: StreamingStats,
    pub flow_len_stats: StreamingStats,

    pub fwd_iat: IatTracker,
    pub bwd_iat: IatTracker,
    pub flow_iat: IatTracker,

    pub active_idle: ActiveIdleTracker,

    // TCP Flags
    pub fin_count: u64,
    pub syn_count: u64,
    pub rst_count: u64,
    pub psh_count: u64,
    pub ack_count: u64,
    pub urg_count: u64,
    pub cwe_count: u64,
    pub ece_count: u64,

    pub fwd_psh_flags: u64,
    pub bwd_psh_flags: u64,
    pub fwd_urg_flags: u64,
    pub bwd_urg_flags: u64,
    pub fwd_syn_flags: u64,
    pub bwd_syn_flags: u64,
    pub fwd_rst_flags: u64,
    pub bwd_rst_flags: u64,

    // TCP Handshake & Windows
    pub init_win_fwd: i64,
    pub init_win_bwd: i64,
    pub last_win_fwd: u16,
    pub last_win_bwd: u16,

    pub syn_time: Option<u64>,
    pub synack_time: Option<u64>,
    pub ack_time: Option<u64>,

    pub synack_delay: Option<f64>,
    pub ackdat_delay: Option<f64>,
    pub tcprtt: Option<f64>,

    // Sequence & Loss Tracking
    pub stcpb: u32,
    pub dtcpb: u32,
    pub has_stcpb: bool,
    pub has_dtcpb: bool,
    pub fwd_highest_seq: u32,
    pub bwd_highest_seq: u32,
    pub sloss: u64,
    pub dloss: u64,

    // Jitter (RFC 3550 style)
    pub fwd_last_iat: Option<f64>,
    pub bwd_last_iat: Option<f64>,
    pub sjit: f64,
    pub djit: f64,

    // TTL
    pub sttl: u8,
    pub dttl: u8,

    // Application & Deep Packet Context
    pub service: String,
    pub trans_depth: u64,
    pub response_body_len: u64,
    pub ct_flw_http_mthd: u64,
    pub ct_ftp_cmd: u64,
    pub ftp_user_seen: bool,
    pub ftp_pass_seen: bool,

    // Specific Dataset Features
    pub act_data_pkt_fwd: u64,
    pub min_seg_size_forward: u64,
    pub wrong_fragment_count: u64,
    pub urgent_count: u64,
    pub is_land: bool,

    // State Tracking
    pub tcp_state: TcpConnState,
}

impl Flow {
    /// Initialize a new bidirectional flow from the first packet seen.
    pub fn new(packet: &ParsedPacket) -> Self {
        let key = FlowKey::canonical(
            packet.src_ip,
            packet.dst_ip,
            packet.src_port,
            packet.dst_port,
            packet.protocol,
        );

        let is_land = packet.src_ip == packet.dst_ip && packet.src_port == packet.dst_port;

        let detected_service = identify_service(packet.src_port, packet.dst_port, packet.protocol, packet.payload);

        let mut flow = Self {
            key,
            orig_src_ip: packet.src_ip,
            orig_dst_ip: packet.dst_ip,
            orig_src_port: packet.src_port,
            orig_dst_port: packet.dst_port,
            protocol: packet.protocol,

            start_time: packet.timestamp_micros,
            last_time: packet.timestamp_micros,

            fwd_pkts: 0,
            bwd_pkts: 0,
            fwd_bytes: 0,
            bwd_bytes: 0,
            fwd_payload_bytes: 0,
            bwd_payload_bytes: 0,
            fwd_header_len: 0,
            bwd_header_len: 0,

            fwd_len_stats: StreamingStats::new(),
            bwd_len_stats: StreamingStats::new(),
            flow_len_stats: StreamingStats::new(),

            fwd_iat: IatTracker::new(),
            bwd_iat: IatTracker::new(),
            flow_iat: IatTracker::new(),

            active_idle: ActiveIdleTracker::default(),

            fin_count: 0,
            syn_count: 0,
            rst_count: 0,
            psh_count: 0,
            ack_count: 0,
            urg_count: 0,
            cwe_count: 0,
            ece_count: 0,

            fwd_psh_flags: 0,
            bwd_psh_flags: 0,
            fwd_urg_flags: 0,
            bwd_urg_flags: 0,
            fwd_syn_flags: 0,
            bwd_syn_flags: 0,
            fwd_rst_flags: 0,
            bwd_rst_flags: 0,

            init_win_fwd: -1,
            init_win_bwd: -1,
            last_win_fwd: 0,
            last_win_bwd: 0,

            syn_time: None,
            synack_time: None,
            ack_time: None,

            synack_delay: None,
            ackdat_delay: None,
            tcprtt: None,

            stcpb: 0,
            dtcpb: 0,
            has_stcpb: false,
            has_dtcpb: false,
            fwd_highest_seq: 0,
            bwd_highest_seq: 0,
            sloss: 0,
            dloss: 0,

            fwd_last_iat: None,
            bwd_last_iat: None,
            sjit: 0.0,
            djit: 0.0,

            sttl: packet.ttl,
            dttl: 0,

            service: detected_service,
            trans_depth: 0,
            response_body_len: 0,
            ct_flw_http_mthd: 0,
            ct_ftp_cmd: 0,
            ftp_user_seen: false,
            ftp_pass_seen: false,

            act_data_pkt_fwd: 0,
            min_seg_size_forward: u64::MAX,
            wrong_fragment_count: 0,
            urgent_count: 0,
            is_land,

            tcp_state: TcpConnState::Start,
        };

        flow.update(packet);
        flow
    }

    /// Determine if an incoming packet matches the forward or backward direction.
    #[inline]
    pub fn direction_of(&self, packet: &ParsedPacket) -> Direction {
        if packet.src_ip == self.orig_src_ip
            && packet.dst_ip == self.orig_dst_ip
            && packet.src_port == self.orig_src_port
            && packet.dst_port == self.orig_dst_port
        {
            Direction::Forward
        } else {
            Direction::Backward
        }
    }

    /// Update the flow with a new packet.
    pub fn update(&mut self, packet: &ParsedPacket) {
        let ts = packet.timestamp_micros;
        self.last_time = ts;

        let direction = self.direction_of(packet);
        let wire_len = packet.wire_len as f64;
        let payload_len = packet.payload.len() as u64;
        let total_hdr_len = (packet.ip_header_len + packet.l4_header_len) as u64;

        // Global flow stats
        self.flow_len_stats.add(wire_len);
        self.flow_iat.update(ts);
        self.active_idle.update(ts);

        if packet.wrong_fragment {
            self.wrong_fragment_count += 1;
        }

        // Refine service detection if unknown
        if self.service == "-" {
            let s = identify_service(packet.src_port, packet.dst_port, packet.protocol, packet.payload);
            if s != "-" {
                self.service = s;
            }
        }

        match direction {
            Direction::Forward => {
                self.fwd_pkts += 1;
                self.fwd_bytes += packet.wire_len as u64;
                self.fwd_payload_bytes += payload_len;
                self.fwd_header_len += total_hdr_len;
                self.fwd_len_stats.add(wire_len);

                if let Some(prev) = self.fwd_iat.last_seen() {
                    let cur_iat = if ts >= prev { (ts - prev) as f64 } else { 0.0 };
                    if let Some(last_iat) = self.fwd_last_iat {
                        let diff = (cur_iat - last_iat).abs();
                        self.sjit += (diff - self.sjit) / 16.0;
                    }
                    self.fwd_last_iat = Some(cur_iat);
                }
                self.fwd_iat.update(ts);

                if payload_len > 0 {
                    self.act_data_pkt_fwd += 1;
                }

                if (packet.l4_header_len as u64) < self.min_seg_size_forward && packet.l4_header_len > 0 {
                    self.min_seg_size_forward = packet.l4_header_len as u64;
                }

                // Inspect forward payload for application metrics
                self.inspect_fwd_payload(packet.payload);
            }
            Direction::Backward => {
                self.bwd_pkts += 1;
                self.bwd_bytes += packet.wire_len as u64;
                self.bwd_payload_bytes += payload_len;
                self.bwd_header_len += total_hdr_len;
                self.bwd_len_stats.add(wire_len);

                if self.dttl == 0 {
                    self.dttl = packet.ttl;
                }

                if let Some(prev) = self.bwd_iat.last_seen() {
                    let cur_iat = if ts >= prev { (ts - prev) as f64 } else { 0.0 };
                    if let Some(last_iat) = self.bwd_last_iat {
                        let diff = (cur_iat - last_iat).abs();
                        self.djit += (diff - self.djit) / 16.0;
                    }
                    self.bwd_last_iat = Some(cur_iat);
                }
                self.bwd_iat.update(ts);

                // Inspect backward payload for response body length
                self.inspect_bwd_payload(packet.payload);
            }
        }

        // L4 protocol specific tracking
        if let L4Header::Tcp {
            seq,
            ack,
            header_len: _,
            flags,
            window_size,
            urgent_ptr,
            ..
        } = packet.l4
        {
            self.update_tcp(direction, ts, seq, ack, flags, window_size, urgent_ptr, payload_len);
        }
    }

    fn update_tcp(
        &mut self,
        direction: Direction,
        ts: u64,
        seq: u32,
        _ack: u32,
        flags: TcpFlags,
        window_size: u16,
        urgent_ptr: u16,
        payload_len: u64,
    ) {
        // TCP flag counts
        if flags.fin { self.fin_count += 1; }
        if flags.syn { self.syn_count += 1; }
        if flags.rst { self.rst_count += 1; }
        if flags.psh { self.psh_count += 1; }
        if flags.ack { self.ack_count += 1; }
        if flags.urg { self.urg_count += 1; }
        if flags.cwr { self.cwe_count += 1; }
        if flags.ece { self.ece_count += 1; }

        if flags.urg || urgent_ptr > 0 {
            self.urgent_count += 1;
        }

        match direction {
            Direction::Forward => {
                if flags.psh { self.fwd_psh_flags += 1; }
                if flags.urg { self.fwd_urg_flags += 1; }
                if flags.syn { self.fwd_syn_flags += 1; }
                if flags.rst { self.fwd_rst_flags += 1; }

                if !self.has_stcpb {
                    self.stcpb = seq;
                    self.has_stcpb = true;
                    self.fwd_highest_seq = seq;
                } else {
                    let end_seq = seq.wrapping_add(payload_len as u32);
                    if payload_len > 0 && seq < self.fwd_highest_seq {
                        self.sloss += 1; // Retransmission or loss indicator
                    }
                    if end_seq > self.fwd_highest_seq {
                        self.fwd_highest_seq = end_seq;
                    }
                }

                self.last_win_fwd = window_size;
                if self.init_win_fwd < 0 && flags.syn {
                    self.init_win_fwd = window_size as i64;
                }

                // Handshake timing
                if flags.syn && !flags.ack && self.syn_time.is_none() {
                    self.syn_time = Some(ts);
                    self.tcp_state = TcpConnState::SynSent;
                } else if flags.ack && !flags.syn && self.ack_time.is_none() && self.synack_time.is_some() {
                    self.ack_time = Some(ts);
                    if let Some(sa_time) = self.synack_time {
                        let delay = if ts >= sa_time { (ts - sa_time) as f64 / 1_000_000.0 } else { 0.0 };
                        self.ackdat_delay = Some(delay);
                        if let Some(syn_delay) = self.synack_delay {
                            self.tcprtt = Some(syn_delay + delay);
                        }
                    }
                    self.tcp_state = TcpConnState::Established;
                }

                if flags.fin {
                    self.tcp_state = match self.tcp_state {
                        TcpConnState::Established => TcpConnState::FinWait1,
                        TcpConnState::CloseWait => TcpConnState::LastAck,
                        other => other,
                    };
                }
                if flags.rst {
                    self.tcp_state = TcpConnState::Reset;
                }
            }
            Direction::Backward => {
                if flags.psh { self.bwd_psh_flags += 1; }
                if flags.urg { self.bwd_urg_flags += 1; }
                if flags.syn { self.bwd_syn_flags += 1; }
                if flags.rst { self.bwd_rst_flags += 1; }

                if !self.has_dtcpb {
                    self.dtcpb = seq;
                    self.has_dtcpb = true;
                    self.bwd_highest_seq = seq;
                } else {
                    let end_seq = seq.wrapping_add(payload_len as u32);
                    if payload_len > 0 && seq < self.bwd_highest_seq {
                        self.dloss += 1;
                    }
                    if end_seq > self.bwd_highest_seq {
                        self.bwd_highest_seq = end_seq;
                    }
                }

                self.last_win_bwd = window_size;
                if self.init_win_bwd < 0 && flags.syn {
                    self.init_win_bwd = window_size as i64;
                }

                // SYN-ACK Handshake timing
                if flags.syn && flags.ack && self.synack_time.is_none() {
                    self.synack_time = Some(ts);
                    if let Some(s_time) = self.syn_time {
                        let delay = if ts >= s_time { (ts - s_time) as f64 / 1_000_000.0 } else { 0.0 };
                        self.synack_delay = Some(delay);
                    }
                    self.tcp_state = TcpConnState::SynRcvd;
                }

                if flags.fin {
                    self.tcp_state = match self.tcp_state {
                        TcpConnState::Established => TcpConnState::CloseWait,
                        TcpConnState::FinWait1 => TcpConnState::Closing,
                        TcpConnState::FinWait2 => TcpConnState::Closed,
                        other => other,
                    };
                }
                if flags.rst {
                    self.tcp_state = TcpConnState::Reset;
                }
            }
        }
    }

    fn inspect_fwd_payload(&mut self, payload: &[u8]) {
        if payload.is_empty() {
            return;
        }

        // HTTP inspection
        if payload.starts_with(b"GET ")
            || payload.starts_with(b"POST ")
            || payload.starts_with(b"PUT ")
            || payload.starts_with(b"DELETE ")
            || payload.starts_with(b"HEAD ")
            || payload.starts_with(b"OPTIONS ")
        {
            self.ct_flw_http_mthd += 1;
            self.trans_depth += 1;
            if self.service == "-" {
                self.service = "http".to_string();
            }
        }

        // FTP command inspection
        if payload.starts_with(b"USER ") {
            self.ct_ftp_cmd += 1;
            self.ftp_user_seen = true;
            if self.service == "-" { self.service = "ftp".to_string(); }
        } else if payload.starts_with(b"PASS ") {
            self.ct_ftp_cmd += 1;
            self.ftp_pass_seen = true;
        } else if payload.starts_with(b"RETR ")
            || payload.starts_with(b"STOR ")
            || payload.starts_with(b"LIST ")
            || payload.starts_with(b"CWD ")
            || payload.starts_with(b"QUIT ")
            || payload.starts_with(b"TYPE ")
            || payload.starts_with(b"PORT ")
            || payload.starts_with(b"PASV ")
        {
            self.ct_ftp_cmd += 1;
        }
    }

    fn inspect_bwd_payload(&mut self, payload: &[u8]) {
        if payload.is_empty() {
            return;
        }

        // HTTP Content-Length inspection
        if payload.starts_with(b"HTTP/1.") || payload.starts_with(b"HTTP/2.") {
            if let Ok(text) = std::str::from_utf8(payload) {
                for line in text.lines() {
                    let lower = line.to_ascii_lowercase();
                    if lower.starts_with("content-length:") {
                        if let Some(val_str) = line.split(':').nth(1) {
                            if let Ok(len) = val_str.trim().parse::<u64>() {
                                self.response_body_len = len;
                                break;
                            }
                        }
                    }
                }
            }
        } else {
            self.response_body_len += payload.len() as u64;
        }
    }

    /// Derive the canonical NSL-KDD flag string.
    pub fn kdd_flag(&self) -> String {
        if self.protocol == 6 {
            // TCP
            if self.fwd_syn_flags > 0 && self.bwd_rst_flags > 0 && self.synack_time.is_none() {
                "REJ".to_string()
            } else if self.fwd_rst_flags > 0 && self.bwd_pkts == 0 {
                "RSTOS0".to_string()
            } else if self.fwd_rst_flags > 0 {
                "RSTO".to_string()
            } else if self.bwd_rst_flags > 0 {
                "RSTR".to_string()
            } else if self.fwd_syn_flags > 0 && self.bwd_pkts == 0 {
                "S0".to_string()
            } else if self.fwd_syn_flags > 0 && self.bwd_syn_flags > 0 && self.fin_count > 0 {
                "SF".to_string()
            } else if self.synack_time.is_some() && self.ack_time.is_some() && self.fin_count == 0 {
                "S1".to_string()
            } else if self.synack_time.is_some() && self.fin_count > 0 && self.bwd_pkts > 0 {
                "SF".to_string()
            } else if self.fwd_syn_flags > 0 && self.fin_count > 0 && self.bwd_pkts == 0 {
                "SH".to_string()
            } else {
                "OTH".to_string()
            }
        } else {
            // UDP / ICMP
            "SF".to_string()
        }
    }

    /// Derive the canonical UNSW-NB15 state string.
    pub fn unsw_state(&self) -> String {
        if self.protocol == 6 {
            if self.rst_count > 0 {
                "RST".to_string()
            } else if self.fin_count > 0 {
                "FIN".to_string()
            } else if self.synack_time.is_some() && self.ack_time.is_some() {
                "CON".to_string()
            } else if self.fwd_syn_flags > 0 && self.bwd_pkts == 0 {
                "REQ".to_string()
            } else if self.fwd_pkts + self.bwd_pkts <= 2 {
                "INT".to_string()
            } else {
                "CLO".to_string()
            }
        } else if self.protocol == 17 {
            if self.bwd_pkts > 0 {
                "CON".to_string()
            } else {
                "INT".to_string()
            }
        } else if self.protocol == 1 || self.protocol == 58 {
            "ECO".to_string()
        } else {
            "CON".to_string()
        }
    }

    #[inline]
    pub fn duration_micros(&self) -> u64 {
        if self.last_time >= self.start_time {
            self.last_time - self.start_time
        } else {
            0
        }
    }

    #[inline]
    pub fn duration_secs(&self) -> f64 {
        self.duration_micros() as f64 / 1_000_000.0
    }
}

fn identify_service(src_port: u16, dst_port: u16, protocol: u8, payload: &[u8]) -> String {
    // Probing payload first
    if payload.starts_with(b"GET ")
        || payload.starts_with(b"POST ")
        || payload.starts_with(b"HTTP/")
        || payload.starts_with(b"HEAD ")
        || payload.starts_with(b"OPTIONS ")
    {
        return "http".to_string();
    }
    if payload.starts_with(b"SSH-") {
        return "ssh".to_string();
    }
    if payload.len() >= 3 && payload[0] == 0x16 && payload[1] == 0x03 {
        return "ssl".to_string();
    }
    if payload.starts_with(b"220 ") || payload.starts_with(b"EHLO ") || payload.starts_with(b"HELO ") {
        return "smtp".to_string();
    }
    if payload.starts_with(b"USER ") || payload.starts_with(b"PASS ") {
        return "ftp".to_string();
    }

    // Well-known port mappings
    let p = if dst_port <= 1024 || (src_port > 1024 && dst_port < src_port) {
        dst_port
    } else {
        src_port
    };

    match p {
        80 | 8080 | 8000 => "http",
        443 | 8443 => "ssl",
        53 => "dns",
        21 | 20 => "ftp",
        22 => "ssh",
        25 | 465 | 587 => "smtp",
        110 | 995 => "pop3",
        143 | 993 => "imap",
        67 | 68 => "dhcp",
        123 => "ntp",
        161 | 162 => "snmp",
        389 | 636 => "ldap",
        1883 | 8883 => "mqtt",
        _ => {
            if protocol == 1 || protocol == 58 {
                "icmp"
            } else if protocol == 254 {
                "arp"
            } else {
                "-"
            }
        }
    }.to_string()
}
