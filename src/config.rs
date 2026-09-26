use clap::Parser;
use std::path::PathBuf;

/// Command-line configuration for the Network Traffic Analysis Engine.
#[derive(Parser, Debug, Clone)]
#[command(
    name = "network-analyzer",
    author = "Network Traffic Analysis Engine",
    version = "0.1.0",
    about = "High-performance Rust network traffic analysis, stateful flow tracking, and NIDS feature extraction engine"
)]
pub struct Config {
    /// Network interface to capture live packets from (e.g. eth0, wlan0, lo)
    #[arg(short, long)]
    pub interface: Option<String>,

    /// List available network interfaces and exit
    #[arg(long)]
    pub list_interfaces: bool,

    /// Offline PCAP file to read and analyze
    #[arg(short, long)]
    pub pcap: Option<PathBuf>,

    /// Output destination file (e.g. flows.jsonl or flows.csv). Defaults to stdout.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Output format: "jsonl" or "csv". If omitted, inferred from output extension or defaults to jsonl.
    #[arg(short, long)]
    pub format: Option<String>,

    /// Inactive flow expiration timeout in seconds.
    #[arg(long, default_value = "30")]
    pub flow_timeout: u64,

    /// Maximum active flow duration in seconds before rotation/flush. (Optional)
    #[arg(long)]
    pub active_timeout: Option<u64>,

    /// Historical host/service statistics sliding window in seconds (NSL-KDD / UNSW-NB15 context).
    #[arg(long, default_value = "60")]
    pub stats_window: u64,

    /// Maximum concurrent flows to track before LRU eviction.
    #[arg(long, default_value = "100000")]
    pub max_flows: usize,

    /// Promiscuous mode for live capture.
    #[arg(long, default_value = "true")]
    pub promisc: bool,

    /// Packet snapshot length in bytes.
    #[arg(long, default_value = "65535")]
    pub snaplen: i32,

    /// Optional BPF filter expression (e.g. 'tcp or udp').
    #[arg(long)]
    pub filter: Option<String>,

    /// Suppress periodic console status logs to stderr.
    #[arg(short, long)]
    pub quiet: bool,

    /// Run synthetic performance benchmark and exit.
    #[arg(long)]
    pub benchmark: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Jsonl,
    Csv,
}

impl Config {
    pub fn resolved_format(&self) -> OutputFormat {
        if let Some(ref fmt) = self.format {
            match fmt.to_lowercase().as_str() {
                "csv" => OutputFormat::Csv,
                _ => OutputFormat::Jsonl,
            }
        } else if let Some(ref path) = self.output {
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if ext.eq_ignore_ascii_case("csv") {
                    return OutputFormat::Csv;
                }
            }
            OutputFormat::Jsonl
        } else {
            OutputFormat::Jsonl
        }
    }
}
