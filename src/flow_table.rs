use std::collections::{HashMap, VecDeque};
use std::net::IpAddr;

use crate::features::{build_flow_record, FlowRecord, HostContextStats};
use crate::flow::Flow;
use crate::packet::{FlowKey, ParsedPacket};

#[derive(Debug, Clone)]
pub struct HostEvent {
    pub timestamp: u64,
    pub src_ip: IpAddr,
    pub dst_ip: IpAddr,
    pub src_port: u16,
    pub dst_port: u16,
    pub service: String,
    pub is_serror: bool,
    pub is_rerror: bool,
    pub state: String,
    pub sttl: u8,
}

/// Bounded sliding window tracker for historical host/service statistics.
#[derive(Debug, Clone)]
pub struct HostServiceTracker {
    events: VecDeque<HostEvent>,
    window_micros: u64,
    max_events: usize,
}

impl HostServiceTracker {
    pub fn new(window_secs: u64) -> Self {
        Self {
            events: VecDeque::with_capacity(10000),
            window_micros: window_secs * 1_000_000,
            max_events: 50_000, // Hard bound against memory exhaustion
        }
    }

    pub fn record_connection(&mut self, flow: &Flow) {
        let is_serror = flow.fwd_syn_flags > 0 && flow.bwd_pkts == 0;
        let is_rerror = flow.rst_count > 0;

        let event = HostEvent {
            timestamp: flow.last_time,
            src_ip: flow.orig_src_ip,
            dst_ip: flow.orig_dst_ip,
            src_port: flow.orig_src_port,
            dst_port: flow.orig_dst_port,
            service: flow.service.clone(),
            is_serror,
            is_rerror,
            state: flow.kdd_flag(),
            sttl: flow.sttl,
        };

        self.events.push_back(event);

        if self.events.len() > self.max_events {
            self.events.pop_front();
        }
    }

    pub fn expire_before(&mut self, now_micros: u64) {
        let cutoff = now_micros.saturating_sub(self.window_micros);
        while let Some(front) = self.events.front() {
            if front.timestamp < cutoff {
                self.events.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn compute_stats(&self, flow: &Flow) -> HostContextStats {
        let mut same_dst_count = 0u64;
        let mut same_srv_count = 0u64;
        let mut same_dst_serror = 0u64;
        let mut same_srv_serror = 0u64;
        let mut same_dst_rerror = 0u64;
        let mut same_srv_rerror = 0u64;
        let mut same_dst_same_srv = 0u64;
        let mut same_srv_diff_host = 0u64;

        let mut same_src_count = 0u64;
        let mut same_src_same_srv = 0u64;
        let mut same_dst_same_srv_unsw = 0u64;
        let mut same_src_dport_count = 0u64;
        let mut same_dst_sport_count = 0u64;
        let mut same_src_dst_pair = 0u64;
        let mut same_state_ttl = 0u64;

        let cur_dst = flow.orig_dst_ip;
        let cur_src = flow.orig_src_ip;
        let cur_sport = flow.orig_src_port;
        let cur_dport = flow.orig_dst_port;
        let cur_srv = &flow.service;
        let cur_state = flow.kdd_flag();
        let cur_ttl = flow.sttl;

        for ev in &self.events {
            let is_same_dst = ev.dst_ip == cur_dst;
            let is_same_src = ev.src_ip == cur_src;
            let is_same_srv = &ev.service == cur_srv;

            if is_same_dst {
                same_dst_count += 1;
                if ev.is_serror { same_dst_serror += 1; }
                if ev.is_rerror { same_dst_rerror += 1; }
                if is_same_srv { same_dst_same_srv += 1; }
                if ev.src_port == cur_sport { same_dst_sport_count += 1; }
            }

            if is_same_srv {
                same_srv_count += 1;
                if ev.is_serror { same_srv_serror += 1; }
                if ev.is_rerror { same_srv_rerror += 1; }
                if ev.dst_ip != cur_dst { same_srv_diff_host += 1; }
            }

            if is_same_src {
                same_src_count += 1;
                if is_same_srv { same_src_same_srv += 1; }
                if ev.dst_port == cur_dport { same_src_dport_count += 1; }
            }

            if is_same_src && is_same_dst {
                same_src_dst_pair += 1;
            }

            if is_same_dst && is_same_srv {
                same_dst_same_srv_unsw += 1;
            }

            if ev.state == cur_state && ev.sttl == cur_ttl {
                same_state_ttl += 1;
            }
        }

        let serror_rate = if same_dst_count > 0 { (same_dst_serror as f64) / (same_dst_count as f64) } else { 0.0 };
        let srv_serror_rate = if same_srv_count > 0 { (same_srv_serror as f64) / (same_srv_count as f64) } else { 0.0 };
        let rerror_rate = if same_dst_count > 0 { (same_dst_rerror as f64) / (same_dst_count as f64) } else { 0.0 };
        let srv_rerror_rate = if same_srv_count > 0 { (same_srv_rerror as f64) / (same_srv_count as f64) } else { 0.0 };

        let same_srv_rate = if same_dst_count > 0 { (same_dst_same_srv as f64) / (same_dst_count as f64) } else { 0.0 };
        let diff_srv_rate = if same_dst_count > 0 { ((same_dst_count - same_dst_same_srv) as f64) / (same_dst_count as f64) } else { 0.0 };
        let srv_diff_host_rate = if same_srv_count > 0 { (same_srv_diff_host as f64) / (same_srv_count as f64) } else { 0.0 };

        let dst_host_same_src_port_rate = if same_dst_count > 0 { (same_dst_sport_count as f64) / (same_dst_count as f64) } else { 0.0 };

        HostContextStats {
            count: same_dst_count,
            srv_count: same_srv_count,
            serror_rate,
            srv_serror_rate,
            rerror_rate,
            srv_rerror_rate,
            same_srv_rate,
            diff_srv_rate,
            srv_diff_host_rate,
            dst_host_count: same_dst_count,
            dst_host_srv_count: same_dst_same_srv,
            dst_host_same_srv_rate: same_srv_rate,
            dst_host_diff_srv_rate: diff_srv_rate,
            dst_host_same_src_port_rate,
            dst_host_srv_diff_host_rate: srv_diff_host_rate,
            dst_host_serror_rate: serror_rate,
            dst_host_srv_serror_rate: srv_serror_rate,
            dst_host_rerror_rate: rerror_rate,
            dst_host_srv_rerror_rate: srv_rerror_rate,
            ct_srv_src: same_src_same_srv,
            ct_srv_dst: same_dst_same_srv_unsw,
            ct_dst_ltm: same_dst_count,
            ct_src_ltm: same_src_count,
            ct_src_dport_ltm: same_src_dport_count,
            ct_dst_sport_ltm: same_dst_sport_count,
            ct_dst_src_ltm: same_src_dst_pair,
            ct_state_ttl: same_state_ttl,
        }
    }
}

/// Stateful bidirectional flow table.
pub struct FlowTable {
    flows: HashMap<FlowKey, Flow>,
    lru_keys: VecDeque<FlowKey>,
    max_flows: usize,
    flow_timeout_micros: u64,
    active_timeout_micros: Option<u64>,
    host_tracker: HostServiceTracker,
    total_completed_flows: u64,
}

impl FlowTable {
    pub fn new(
        max_flows: usize,
        flow_timeout_secs: u64,
        active_timeout_secs: Option<u64>,
        stats_window_secs: u64,
    ) -> Self {
        Self {
            flows: HashMap::with_capacity(max_flows.min(10000)),
            lru_keys: VecDeque::with_capacity(max_flows.min(10000)),
            max_flows,
            flow_timeout_micros: flow_timeout_secs * 1_000_000,
            active_timeout_micros: active_timeout_secs.map(|s| s * 1_000_000),
            host_tracker: HostServiceTracker::new(stats_window_secs),
            total_completed_flows: 0,
        }
    }

    /// Process a parsed packet. If a flow was evicted due to capacity limits or active timeout,
    /// returns the finalized FlowRecord.
    pub fn process_packet(&mut self, packet: &ParsedPacket) -> Option<FlowRecord> {
        let key = FlowKey::canonical(
            packet.src_ip,
            packet.dst_ip,
            packet.src_port,
            packet.dst_port,
            packet.protocol,
        );

        let mut evicted_record = None;

        if let Some(flow) = self.flows.get_mut(&key) {
            flow.update(packet);

            // Check if flow exceeded active timeout limit
            if let Some(max_active) = self.active_timeout_micros {
                if flow.duration_micros() >= max_active {
                    let flow_clone = self.flows.remove(&key).unwrap();
                    let record = self.finalize_flow(flow_clone);
                    evicted_record = Some(record);
                }
            }
        } else {
            // Need to insert new flow. Check capacity first.
            if self.flows.len() >= self.max_flows {
                if let Some(evicted_key) = self.lru_keys.pop_front() {
                    if let Some(old_flow) = self.flows.remove(&evicted_key) {
                        let record = self.finalize_flow(old_flow);
                        evicted_record = Some(record);
                    }
                }
            }

            let new_flow = Flow::new(packet);
            self.flows.insert(key, new_flow);
            self.lru_keys.push_back(key);
        }

        evicted_record
    }

    /// Check and evict any inactive flows whose last seen time is older than flow_timeout.
    pub fn check_timeouts(&mut self, now_micros: u64) -> Vec<FlowRecord> {
        self.host_tracker.expire_before(now_micros);

        let mut expired_keys = Vec::new();

        for (k, flow) in &self.flows {
            let timeout_micros = flow.inactivity_timeout_micros(self.flow_timeout_micros);
            let timeout_thresh = now_micros.saturating_sub(timeout_micros);
            if flow.last_time < timeout_thresh {
                expired_keys.push(*k);
            }
        }

        let mut records = Vec::with_capacity(expired_keys.len());
        for k in expired_keys {
            if let Some(flow) = self.flows.remove(&k) {
                records.push(self.finalize_flow(flow));
            }
        }

        if !records.is_empty() {
            // Retain only live keys in LRU
            self.lru_keys.retain(|k| self.flows.contains_key(k));
        }

        records
    }

    /// Flush all remaining active flows (e.g. at end of PCAP or shutdown).
    pub fn flush_all(&mut self) -> Vec<FlowRecord> {
        let mut records = Vec::with_capacity(self.flows.len());
        let flows: Vec<Flow> = self.flows.drain().map(|(_, v)| v).collect();
        self.lru_keys.clear();

        for flow in flows {
            records.push(self.finalize_flow(flow));
        }

        records
    }

    fn finalize_flow(&mut self, flow: Flow) -> FlowRecord {
        self.total_completed_flows += 1;
        let host_stats = self.host_tracker.compute_stats(&flow);
        self.host_tracker.record_connection(&flow);
        build_flow_record(flow, &host_stats)
    }

    #[inline]
    pub fn active_flow_count(&self) -> usize {
        self.flows.len()
    }

    #[inline]
    pub fn total_completed_flows(&self) -> u64 {
        self.total_completed_flows
    }
}
