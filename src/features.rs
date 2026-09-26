use serde::{Deserialize, Serialize};
use crate::flow::Flow;

/// Strongly-typed canonical FlowRecord containing features extracted for
/// CICIDS2017, UNSW-NB15, and NSL-KDD dataset schemas.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FlowRecord {
    // Identity & Metadata
    pub flow_id: String,
    pub timestamp: String,
    pub src_ip: String,
    pub dst_ip: String,
    pub src_port: u16,
    pub dst_port: u16,
    pub protocol: String,

    // Timing
    pub flow_duration_micros: u64,
    pub flow_duration_secs: f64,

    // Directional Packets & Bytes
    pub total_fwd_packets: u64,
    pub total_bwd_packets: u64,
    pub total_fwd_bytes: u64,
    pub total_bwd_bytes: u64,

    // ==========================================
    // CICIDS2017 Features
    // ==========================================
    pub fwd_pkt_len_max: f64,
    pub fwd_pkt_len_min: f64,
    pub fwd_pkt_len_mean: f64,
    pub fwd_pkt_len_std: f64,
    pub bwd_pkt_len_max: f64,
    pub bwd_pkt_len_min: f64,
    pub bwd_pkt_len_mean: f64,
    pub bwd_pkt_len_std: f64,

    pub flow_bytes_per_sec: f64,
    pub flow_packets_per_sec: f64,

    pub flow_iat_mean: f64,
    pub flow_iat_std: f64,
    pub flow_iat_max: f64,
    pub flow_iat_min: f64,

    pub fwd_iat_total: f64,
    pub fwd_iat_mean: f64,
    pub fwd_iat_std: f64,
    pub fwd_iat_max: f64,
    pub fwd_iat_min: f64,

    pub bwd_iat_total: f64,
    pub bwd_iat_mean: f64,
    pub bwd_iat_std: f64,
    pub bwd_iat_max: f64,
    pub bwd_iat_min: f64,

    pub fwd_psh_flags: u64,
    pub bwd_psh_flags: u64,
    pub fwd_urg_flags: u64,
    pub bwd_urg_flags: u64,

    pub fwd_header_length: u64,
    pub bwd_header_length: u64,

    pub fwd_packets_per_sec: f64,
    pub bwd_packets_per_sec: f64,

    pub min_packet_length: f64,
    pub max_packet_length: f64,
    pub packet_length_mean: f64,
    pub packet_length_std: f64,
    pub packet_length_variance: f64,

    pub fin_flag_count: u64,
    pub syn_flag_count: u64,
    pub rst_flag_count: u64,
    pub psh_flag_count: u64,
    pub ack_flag_count: u64,
    pub urg_flag_count: u64,
    pub cwe_flag_count: u64,
    pub ece_flag_count: u64,

    pub down_up_ratio: f64,
    pub average_packet_size: f64,
    pub avg_fwd_segment_size: f64,
    pub avg_bwd_segment_size: f64,

    pub subflow_fwd_packets: u64,
    pub subflow_fwd_bytes: u64,
    pub subflow_bwd_packets: u64,
    pub subflow_bwd_bytes: u64,

    pub init_win_bytes_forward: i64,
    pub init_win_bytes_backward: i64,
    pub act_data_pkt_fwd: u64,
    pub min_seg_size_forward: u64,

    pub active_mean: f64,
    pub active_std: f64,
    pub active_max: f64,
    pub active_min: f64,
    pub idle_mean: f64,
    pub idle_std: f64,
    pub idle_max: f64,
    pub idle_min: f64,

    // ==========================================
    // UNSW-NB15 Features
    // ==========================================
    pub unsw_dur: f64,
    pub unsw_proto: String,
    pub unsw_service: String,
    pub unsw_state: String,
    pub unsw_spkts: u64,
    pub unsw_dpkts: u64,
    pub unsw_sbytes: u64,
    pub unsw_dbytes: u64,
    pub unsw_rate: f64,
    pub unsw_sttl: u8,
    pub unsw_dttl: u8,
    pub unsw_sload: f64,
    pub unsw_dload: f64,
    pub unsw_sloss: u64,
    pub unsw_dloss: u64,
    pub unsw_sinpkt: f64,
    pub unsw_dinpkt: f64,
    pub unsw_sjit: f64,
    pub unsw_djit: f64,
    pub unsw_swin: u16,
    pub unsw_stcpb: u32,
    pub unsw_dtcpb: u32,
    pub unsw_dwin: u16,
    pub unsw_tcprtt: f64,
    pub unsw_synack: f64,
    pub unsw_ackdat: f64,
    pub unsw_smean: f64,
    pub unsw_dmean: f64,
    pub unsw_trans_depth: u64,
    pub unsw_response_body_len: u64,
    pub unsw_ct_srv_src: u64,
    pub unsw_ct_state_ttl: u64,
    pub unsw_ct_dst_ltm: u64,
    pub unsw_ct_src_dport_ltm: u64,
    pub unsw_ct_dst_sport_ltm: u64,
    pub unsw_ct_dst_src_ltm: u64,
    pub unsw_is_ftp_login: u8,
    pub unsw_ct_ftp_cmd: u64,
    pub unsw_ct_flw_http_mthd: u64,
    pub unsw_ct_src_ltm: u64,
    pub unsw_ct_srv_dst: u64,
    pub unsw_is_sm_ips_ports: u8,

    // ==========================================
    // NSL-KDD Features
    // ==========================================
    pub kdd_duration: u64,
    pub kdd_protocol_type: String,
    pub kdd_service: String,
    pub kdd_flag: String,
    pub kdd_src_bytes: u64,
    pub kdd_dst_bytes: u64,
    pub kdd_land: u8,
    pub kdd_wrong_fragment: u64,
    pub kdd_urgent: u64,
    pub kdd_count: u64,
    pub kdd_srv_count: u64,
    pub kdd_serror_rate: f64,
    pub kdd_srv_serror_rate: f64,
    pub kdd_rerror_rate: f64,
    pub kdd_srv_rerror_rate: f64,
    pub kdd_same_srv_rate: f64,
    pub kdd_diff_srv_rate: f64,
    pub kdd_srv_diff_host_rate: f64,
    pub kdd_dst_host_count: u64,
    pub kdd_dst_host_srv_count: u64,
    pub kdd_dst_host_same_srv_rate: f64,
    pub kdd_dst_host_diff_srv_rate: f64,
    pub kdd_dst_host_same_src_port_rate: f64,
    pub kdd_dst_host_srv_diff_host_rate: f64,
    pub kdd_dst_host_serror_rate: f64,
    pub kdd_dst_host_srv_serror_rate: f64,
    pub kdd_dst_host_rerror_rate: f64,
    pub kdd_dst_host_srv_rerror_rate: f64,
}

#[derive(Debug, Clone, Default)]
pub struct HostContextStats {
    pub count: u64,
    pub srv_count: u64,
    pub serror_rate: f64,
    pub srv_serror_rate: f64,
    pub rerror_rate: f64,
    pub srv_rerror_rate: f64,
    pub same_srv_rate: f64,
    pub diff_srv_rate: f64,
    pub srv_diff_host_rate: f64,
    pub dst_host_count: u64,
    pub dst_host_srv_count: u64,
    pub dst_host_same_srv_rate: f64,
    pub dst_host_diff_srv_rate: f64,
    pub dst_host_same_src_port_rate: f64,
    pub dst_host_srv_diff_host_rate: f64,
    pub dst_host_serror_rate: f64,
    pub dst_host_srv_serror_rate: f64,
    pub dst_host_rerror_rate: f64,
    pub dst_host_srv_rerror_rate: f64,
    pub ct_srv_src: u64,
    pub ct_srv_dst: u64,
    pub ct_dst_ltm: u64,
    pub ct_src_ltm: u64,
    pub ct_src_dport_ltm: u64,
    pub ct_dst_sport_ltm: u64,
    pub ct_dst_src_ltm: u64,
    pub ct_state_ttl: u64,
}

/// Convert a completed Flow and its rolling HostContextStats into a canonical FlowRecord.
pub fn build_flow_record(mut flow: Flow, host_stats: &HostContextStats) -> FlowRecord {
    // Finalize active/idle bursts
    flow.active_idle.finalize();

    let dur_micros = flow.duration_micros();
    let dur_secs = flow.duration_secs();

    let total_packets = flow.fwd_pkts + flow.bwd_pkts;
    let total_bytes = flow.fwd_bytes + flow.bwd_bytes;

    let flow_bytes_per_sec = if dur_secs > 0.0 { (total_bytes as f64) / dur_secs } else { 0.0 };
    let flow_packets_per_sec = if dur_secs > 0.0 { (total_packets as f64) / dur_secs } else { 0.0 };
    let fwd_packets_per_sec = if dur_secs > 0.0 { (flow.fwd_pkts as f64) / dur_secs } else { 0.0 };
    let bwd_packets_per_sec = if dur_secs > 0.0 { (flow.bwd_pkts as f64) / dur_secs } else { 0.0 };

    let down_up_ratio = if flow.fwd_pkts > 0 {
        (flow.bwd_pkts as f64) / (flow.fwd_pkts as f64)
    } else {
        0.0
    };

    let average_packet_size = if total_packets > 0 {
        (total_bytes as f64) / (total_packets as f64)
    } else {
        0.0
    };

    let avg_fwd_segment_size = if flow.fwd_pkts > 0 {
        (flow.fwd_bytes as f64) / (flow.fwd_pkts as f64)
    } else {
        0.0
    };

    let avg_bwd_segment_size = if flow.bwd_pkts > 0 {
        (flow.bwd_bytes as f64) / (flow.bwd_pkts as f64)
    } else {
        0.0
    };

    let min_seg_size_forward = if flow.min_seg_size_forward != u64::MAX {
        flow.min_seg_size_forward
    } else {
        0
    };

    let proto_str = match flow.protocol {
        6 => "TCP",
        17 => "UDP",
        1 => "ICMP",
        58 => "ICMPv6",
        254 => "ARP",
        _ => "OTHER",
    }.to_string();

    let kdd_proto_str = match flow.protocol {
        6 => "tcp",
        17 => "udp",
        1 | 58 => "icmp",
        _ => "other",
    }.to_string();

    let flow_id = format!(
        "{}:{}-{}:{}-{}",
        flow.orig_src_ip, flow.orig_src_port, flow.orig_dst_ip, flow.orig_dst_port, flow.protocol
    );

    let timestamp_str = format_micros_to_iso(flow.start_time);

    let kdd_flag = flow.kdd_flag();
    let unsw_state = flow.unsw_state();

    let sload = if dur_secs > 0.0 { ((flow.fwd_bytes * 8) as f64) / dur_secs } else { 0.0 };
    let dload = if dur_secs > 0.0 { ((flow.bwd_bytes * 8) as f64) / dur_secs } else { 0.0 };

    let sinpkt = if flow.fwd_pkts > 1 {
        (flow.fwd_iat.total() / ((flow.fwd_pkts - 1) as f64)) / 1000.0 // in milliseconds
    } else {
        0.0
    };

    let dinpkt = if flow.bwd_pkts > 1 {
        (flow.bwd_iat.total() / ((flow.bwd_pkts - 1) as f64)) / 1000.0
    } else {
        0.0
    };

    let is_ftp_login = if flow.ftp_user_seen && flow.ftp_pass_seen { 1 } else { 0 };
    let is_sm_ips_ports = if flow.is_land { 1 } else { 0 };

    FlowRecord {
        flow_id,
        timestamp: timestamp_str,
        src_ip: flow.orig_src_ip.to_string(),
        dst_ip: flow.orig_dst_ip.to_string(),
        src_port: flow.orig_src_port,
        dst_port: flow.orig_dst_port,
        protocol: proto_str,

        flow_duration_micros: dur_micros,
        flow_duration_secs: dur_secs,

        total_fwd_packets: flow.fwd_pkts,
        total_bwd_packets: flow.bwd_pkts,
        total_fwd_bytes: flow.fwd_bytes,
        total_bwd_bytes: flow.bwd_bytes,

        // CICIDS2017 Lengths
        fwd_pkt_len_max: flow.fwd_len_stats.max(),
        fwd_pkt_len_min: flow.fwd_len_stats.min(),
        fwd_pkt_len_mean: flow.fwd_len_stats.mean(),
        fwd_pkt_len_std: flow.fwd_len_stats.std_dev(),
        bwd_pkt_len_max: flow.bwd_len_stats.max(),
        bwd_pkt_len_min: flow.bwd_len_stats.min(),
        bwd_pkt_len_mean: flow.bwd_len_stats.mean(),
        bwd_pkt_len_std: flow.bwd_len_stats.std_dev(),

        flow_bytes_per_sec,
        flow_packets_per_sec,

        // CICIDS2017 IATs
        flow_iat_mean: flow.flow_iat.mean(),
        flow_iat_std: flow.flow_iat.std_dev(),
        flow_iat_max: flow.flow_iat.max(),
        flow_iat_min: flow.flow_iat.min(),

        fwd_iat_total: flow.fwd_iat.total(),
        fwd_iat_mean: flow.fwd_iat.mean(),
        fwd_iat_std: flow.fwd_iat.std_dev(),
        fwd_iat_max: flow.fwd_iat.max(),
        fwd_iat_min: flow.fwd_iat.min(),

        bwd_iat_total: flow.bwd_iat.total(),
        bwd_iat_mean: flow.bwd_iat.mean(),
        bwd_iat_std: flow.bwd_iat.std_dev(),
        bwd_iat_max: flow.bwd_iat.max(),
        bwd_iat_min: flow.bwd_iat.min(),

        fwd_psh_flags: flow.fwd_psh_flags,
        bwd_psh_flags: flow.bwd_psh_flags,
        fwd_urg_flags: flow.fwd_urg_flags,
        bwd_urg_flags: flow.bwd_urg_flags,

        fwd_header_length: flow.fwd_header_len,
        bwd_header_length: flow.bwd_header_len,

        fwd_packets_per_sec,
        bwd_packets_per_sec,

        min_packet_length: flow.flow_len_stats.min(),
        max_packet_length: flow.flow_len_stats.max(),
        packet_length_mean: flow.flow_len_stats.mean(),
        packet_length_std: flow.flow_len_stats.std_dev(),
        packet_length_variance: flow.flow_len_stats.variance(),

        fin_flag_count: flow.fin_count,
        syn_flag_count: flow.syn_count,
        rst_flag_count: flow.rst_count,
        psh_flag_count: flow.psh_count,
        ack_flag_count: flow.ack_count,
        urg_flag_count: flow.urg_count,
        cwe_flag_count: flow.cwe_count,
        ece_flag_count: flow.ece_count,

        down_up_ratio,
        average_packet_size,
        avg_fwd_segment_size,
        avg_bwd_segment_size,

        subflow_fwd_packets: flow.fwd_pkts,
        subflow_fwd_bytes: flow.fwd_bytes,
        subflow_bwd_packets: flow.bwd_pkts,
        subflow_bwd_bytes: flow.bwd_bytes,

        init_win_bytes_forward: flow.init_win_fwd,
        init_win_bytes_backward: flow.init_win_bwd,
        act_data_pkt_fwd: flow.act_data_pkt_fwd,
        min_seg_size_forward,

        active_mean: flow.active_idle.active_mean(),
        active_std: flow.active_idle.active_std(),
        active_max: flow.active_idle.active_max(),
        active_min: flow.active_idle.active_min(),
        idle_mean: flow.active_idle.idle_mean(),
        idle_std: flow.active_idle.idle_std(),
        idle_max: flow.active_idle.idle_max(),
        idle_min: flow.active_idle.idle_min(),

        // UNSW-NB15 Features
        unsw_dur: dur_secs,
        unsw_proto: flow.protocol.to_string(),
        unsw_service: flow.service.clone(),
        unsw_state,
        unsw_spkts: flow.fwd_pkts,
        unsw_dpkts: flow.bwd_pkts,
        unsw_sbytes: flow.fwd_bytes,
        unsw_dbytes: flow.bwd_bytes,
        unsw_rate: flow_packets_per_sec,
        unsw_sttl: flow.sttl,
        unsw_dttl: flow.dttl,
        unsw_sload: sload,
        unsw_dload: dload,
        unsw_sloss: flow.sloss,
        unsw_dloss: flow.dloss,
        unsw_sinpkt: sinpkt,
        unsw_dinpkt: dinpkt,
        unsw_sjit: flow.sjit,
        unsw_djit: flow.djit,
        unsw_swin: flow.last_win_fwd,
        unsw_stcpb: flow.stcpb,
        unsw_dtcpb: flow.dtcpb,
        unsw_dwin: flow.last_win_bwd,
        unsw_tcprtt: flow.tcprtt.unwrap_or(0.0),
        unsw_synack: flow.synack_delay.unwrap_or(0.0),
        unsw_ackdat: flow.ackdat_delay.unwrap_or(0.0),
        unsw_smean: avg_fwd_segment_size,
        unsw_dmean: avg_bwd_segment_size,
        unsw_trans_depth: flow.trans_depth,
        unsw_response_body_len: flow.response_body_len,
        unsw_ct_srv_src: host_stats.ct_srv_src,
        unsw_ct_state_ttl: host_stats.ct_state_ttl,
        unsw_ct_dst_ltm: host_stats.ct_dst_ltm,
        unsw_ct_src_dport_ltm: host_stats.ct_src_dport_ltm,
        unsw_ct_dst_sport_ltm: host_stats.ct_dst_sport_ltm,
        unsw_ct_dst_src_ltm: host_stats.ct_dst_src_ltm,
        unsw_is_ftp_login: is_ftp_login,
        unsw_ct_ftp_cmd: flow.ct_ftp_cmd,
        unsw_ct_flw_http_mthd: flow.ct_flw_http_mthd,
        unsw_ct_src_ltm: host_stats.ct_src_ltm,
        unsw_ct_srv_dst: host_stats.ct_srv_dst,
        unsw_is_sm_ips_ports: is_sm_ips_ports,

        // NSL-KDD Features
        kdd_duration: dur_secs.floor() as u64,
        kdd_protocol_type: kdd_proto_str,
        kdd_service: flow.service.clone(),
        kdd_flag,
        kdd_src_bytes: flow.fwd_payload_bytes,
        kdd_dst_bytes: flow.bwd_payload_bytes,
        kdd_land: if flow.is_land { 1 } else { 0 },
        kdd_wrong_fragment: flow.wrong_fragment_count,
        kdd_urgent: flow.urgent_count,
        kdd_count: host_stats.count,
        kdd_srv_count: host_stats.srv_count,
        kdd_serror_rate: host_stats.serror_rate,
        kdd_srv_serror_rate: host_stats.srv_serror_rate,
        kdd_rerror_rate: host_stats.rerror_rate,
        kdd_srv_rerror_rate: host_stats.srv_rerror_rate,
        kdd_same_srv_rate: host_stats.same_srv_rate,
        kdd_diff_srv_rate: host_stats.diff_srv_rate,
        kdd_srv_diff_host_rate: host_stats.srv_diff_host_rate,
        kdd_dst_host_count: host_stats.dst_host_count,
        kdd_dst_host_srv_count: host_stats.dst_host_srv_count,
        kdd_dst_host_same_srv_rate: host_stats.dst_host_same_srv_rate,
        kdd_dst_host_diff_srv_rate: host_stats.dst_host_diff_srv_rate,
        kdd_dst_host_same_src_port_rate: host_stats.dst_host_same_src_port_rate,
        kdd_dst_host_srv_diff_host_rate: host_stats.dst_host_srv_diff_host_rate,
        kdd_dst_host_serror_rate: host_stats.dst_host_serror_rate,
        kdd_dst_host_srv_serror_rate: host_stats.dst_host_srv_serror_rate,
        kdd_dst_host_rerror_rate: host_stats.dst_host_rerror_rate,
        kdd_dst_host_srv_rerror_rate: host_stats.dst_host_srv_rerror_rate,
    }
}

fn format_micros_to_iso(micros: u64) -> String {
    let secs = (micros / 1_000_000) as i64;
    let nsecs = ((micros % 1_000_000) * 1000) as u32;
    if let Some(dt) = chrono::DateTime::from_timestamp(secs, nsecs) {
        dt.to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
    } else {
        format!("{} us", micros)
    }
}
