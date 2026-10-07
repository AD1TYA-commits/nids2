use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use network_analyzer::capture::{list_interfaces, CaptureSource};
use network_analyzer::config::Config;
use network_analyzer::flow_table::FlowTable;
use network_analyzer::output::FlowWriter;
use network_analyzer::packet::parse_packet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::parse();

    // 1. Handle --list-interfaces
    if config.list_interfaces {
        println!("{:<15} {:<10} {:<30} Description", "INTERFACE", "STATUS", "ADDRESSES");
        println!("{:-<75}", "");
        match list_interfaces() {
            Ok(interfaces) => {
                for iface in interfaces {
                    let status = if iface.is_up { "UP" } else { "DOWN" };
                    let addrs = if iface.addresses.is_empty() {
                        "-".to_string()
                    } else {
                        iface.addresses.join(", ")
                    };
                    let desc = iface.description.unwrap_or_else(|| "-".to_string());
                    println!("{:<15} {:<10} {:<30} {}", iface.name, status, addrs, desc);
                }
            }
            Err(e) => {
                eprintln!("Error enumerating network interfaces: {}", e);
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    // 2. Handle --benchmark
    if config.benchmark {
        run_benchmark(&config)?;
        return Ok(());
    }

    // 3. Setup Capture Source
    let is_offline = config.pcap.is_some();
    let mut capture = if let Some(ref pcap_path) = config.pcap {
        if !pcap_path.exists() {
            eprintln!("Error: PCAP file '{}' does not exist.", pcap_path.display());
            std::process::exit(1);
        }
        eprintln!("[INFO] Opening PCAP file for replay: {}", pcap_path.display());
        CaptureSource::open_offline(pcap_path, config.filter.as_deref())?
    } else if let Some(ref iface_name) = config.interface {
        eprintln!(
            "[INFO] Starting live capture on interface '{}' (promisc: {}, snaplen: {})...",
            iface_name, config.promisc, config.snaplen
        );
        CaptureSource::open_live(
            iface_name,
            config.promisc,
            config.snaplen,
            config.filter.as_deref(),
        )?
    } else {
        eprintln!("Error: Either --interface <NAME> or --pcap <PATH> must be specified.");
        eprintln!("Run with --help for usage details, or --list-interfaces to view available devices.");
        std::process::exit(1);
    };

    // 4. Setup Signal Handling
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
        eprintln!("\n[INFO] Interrupt signal received, flushing active flows and shutting down...");
    })?;

    // 5. Initialize Engine Components
    let mut flow_table = FlowTable::new(
        config.max_flows,
        config.flow_timeout,
        config.active_timeout,
        config.stats_window,
    );

    let format = config.resolved_format();
    let backend_target = config.backend.as_deref().filter(|s| !s.is_empty() && *s != "none");
    let mut writer = FlowWriter::new(config.output.as_deref(), format, backend_target)?;

    eprintln!(
        "[INFO] Flow Engine configured: flow_timeout={}s, stats_window={}s, max_flows={}, format={:?}, backend={:?}",
        config.flow_timeout, config.stats_window, config.max_flows, format, backend_target
    );

    let start_instant = Instant::now();
    let mut total_packets: u64 = 0;
    let mut malformed_packets: u64 = 0;
    let mut last_status_update = Instant::now();
    let mut last_timeout_check = Instant::now();
    let mut last_packet_ts_micros: u64 = 0;

    // 6. Packet Processing Loop
    while running.load(Ordering::SeqCst) {
        match capture.next_packet() {
            Ok(Some((ts_micros, raw_bytes))) => {
                total_packets += 1;
                last_packet_ts_micros = ts_micros;

                match parse_packet(raw_bytes, ts_micros) {
                    Ok(parsed) => {
                        if let Some(evicted_record) = flow_table.process_packet(&parsed) {
                            writer.write_record(&evicted_record)?;
                            writer.flush()?;
                        }
                    }
                    Err(_) => {
                        malformed_packets += 1;
                    }
                }

                // Check periodic timeouts frequently (every 50 packets or 500ms) for real-time live streaming
                if total_packets % 50 == 0 || last_timeout_check.elapsed().as_millis() >= 500 {
                    let expired = flow_table.check_timeouts(ts_micros);
                    if !expired.is_empty() {
                        writer.write_records(&expired)?;
                        writer.flush()?;
                    }
                    last_timeout_check = Instant::now();

                    if !config.quiet && last_status_update.elapsed().as_millis() >= 1000 {
                        let elapsed_sec = start_instant.elapsed().as_secs_f64();
                        let pps = if elapsed_sec > 0.0 { (total_packets as f64) / elapsed_sec } else { 0.0 };
                        eprintln!(
                            "[RUNNING] Pkts: {} | Active Flows: {} | Completed: {} | Rate: {:.1} pkts/s",
                            total_packets,
                            flow_table.active_flow_count(),
                            flow_table.total_completed_flows(),
                            pps
                        );
                        last_status_update = Instant::now();
                    }
                }
            }
            Ok(None) => {
                if is_offline {
                    // Reached end of PCAP file
                    eprintln!("[INFO] Reached end of PCAP file.");
                    break;
                }
                // Live capture idle timeout, perform housekeeping
                let now_micros = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_micros() as u64)
                    .unwrap_or(last_packet_ts_micros);

                let expired = flow_table.check_timeouts(now_micros);
                if !expired.is_empty() {
                    writer.write_records(&expired)?;
                    writer.flush()?;
                }
            }
            Err(e) => {
                eprintln!("[ERROR] Capture error: {}", e);
                break;
            }
        }
    }

    // 7. Finalize and Flush
    let remaining_records = flow_table.flush_all();
    writer.write_records(&remaining_records)?;
    writer.flush()?;

    let total_elapsed = start_instant.elapsed().as_secs_f64();
    let total_completed = flow_table.total_completed_flows();
    let pps = if total_elapsed > 0.0 { (total_packets as f64) / total_elapsed } else { 0.0 };
    let fps = if total_elapsed > 0.0 { (total_completed as f64) / total_elapsed } else { 0.0 };

    eprintln!("\n================ CAPTURE SUMMARY ================");
    eprintln!("Duration:            {:.3} seconds", total_elapsed);
    eprintln!("Total Packets:       {}", total_packets);
    eprintln!("Malformed Packets:   {}", malformed_packets);
    eprintln!("Completed Flows:     {}", total_completed);
    eprintln!("Packet Throughput:   {:.2} packets/sec", pps);
    eprintln!("Flow Throughput:     {:.2} flows/sec", fps);
    if let Some(ref out) = config.output {
        eprintln!("Output Written To:   {}", out.display());
    }
    eprintln!("=================================================");

    Ok(())
}

/// Run an in-engine synthetic benchmark generating 100,000 realistic packets across 1,000 flows
/// to benchmark parsing, flow tracking, active/idle bursts, and feature extraction.
fn run_benchmark(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("\n========================================================");
    eprintln!("  Network Traffic Analysis Engine - Performance Benchmark");
    eprintln!("========================================================");

    let num_flows = 2_000;
    let packets_per_flow = 50;
    let total_benchmark_packets = num_flows * packets_per_flow;

    eprintln!(
        "Generating & processing {} packets across {} bidirectional flows...",
        total_benchmark_packets, num_flows
    );

    let mut flow_table = FlowTable::new(
        config.max_flows,
        config.flow_timeout,
        config.active_timeout,
        config.stats_window,
    );

    let format = config.resolved_format();
    let mut writer = FlowWriter::new(config.output.as_deref(), format, None)?;

    let bench_start = Instant::now();
    let base_time: u64 = 1_700_000_000_000_000; // microsecond epoch

    for pkt_idx in 0..packets_per_flow {
        for flow_idx in 0..num_flows {
            let flow_id = flow_idx as u32;
            let is_fwd = pkt_idx % 2 == 0;

            let (src_ip_b, dst_ip_b, src_port, dst_port) = if is_fwd {
                (10, 20, 10000 + (flow_id as u16 % 50000), 80)
            } else {
                (20, 10, 80, 10000 + (flow_id as u16 % 50000))
            };

            let ts_micros = base_time + (pkt_idx as u64 * 50_000) + (flow_id as u64 * 10);
            let raw_pkt = build_synthetic_packet(src_ip_b, dst_ip_b, src_port, dst_port, pkt_idx);

            let parsed = parse_packet(&raw_pkt, ts_micros)?;
            if let Some(record) = flow_table.process_packet(&parsed) {
                writer.write_record(&record)?;
            }
        }
    }

    let records = flow_table.flush_all();
    writer.write_records(&records)?;
    writer.flush()?;

    let elapsed = bench_start.elapsed().as_secs_f64();
    let total_flows = flow_table.total_completed_flows();
    let pps = (total_benchmark_packets as f64) / elapsed;
    let fps = (total_flows as f64) / elapsed;

    eprintln!("\nBenchmark Results:");
    eprintln!("  Elapsed Time:       {:.4} seconds", elapsed);
    eprintln!("  Packets Processed:  {}", total_benchmark_packets);
    eprintln!("  Flows Generated:    {}", total_flows);
    eprintln!("  Packets / Second:   {:.2} pkts/sec", pps);
    eprintln!("  Flows / Second:     {:.2} flows/sec", fps);
    eprintln!("  Feature Time/Flow:  {:.3} microseconds", (elapsed / total_flows as f64) * 1_000_000.0);
    eprintln!("========================================================\n");

    Ok(())
}

fn build_synthetic_packet(
    src_last_byte: u8,
    dst_last_byte: u8,
    src_port: u16,
    dst_port: u16,
    seq_step: usize,
) -> Vec<u8> {
    let mut pkt = Vec::with_capacity(74);

    // Ethernet header (14 bytes)
    pkt.extend_from_slice(&[0x00, 0x11, 0x22, 0x33, 0x44, 0x55]); // Dst MAC
    pkt.extend_from_slice(&[0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb]); // Src MAC
    pkt.extend_from_slice(&[0x08, 0x00]); // EtherType: IPv4

    // IPv4 header (20 bytes)
    pkt.push(0x45); // Version 4, IHL 5 (20 bytes)
    pkt.push(0x00); // DSCP/ECN
    pkt.extend_from_slice(&[0x00, 0x3c]); // Total length: 60 bytes
    pkt.extend_from_slice(&[0x12, 0x34]); // Identification
    pkt.extend_from_slice(&[0x40, 0x00]); // Don't fragment
    pkt.push(64); // TTL
    pkt.push(6); // Protocol: TCP
    pkt.extend_from_slice(&[0x00, 0x00]); // Checksum
    pkt.extend_from_slice(&[192, 168, 1, src_last_byte]); // Src IP
    pkt.extend_from_slice(&[192, 168, 1, dst_last_byte]); // Dst IP

    // TCP header (20 bytes)
    pkt.extend_from_slice(&src_port.to_be_bytes());
    pkt.extend_from_slice(&dst_port.to_be_bytes());
    let seq = (seq_step as u32) * 1000 + 1;
    pkt.extend_from_slice(&seq.to_be_bytes());
    pkt.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]); // Ack num
    let flags: u16 = if seq_step == 0 {
        0x5002 // Data offset 5 (20 bytes), SYN
    } else if seq_step == 49 {
        0x5011 // FIN + ACK
    } else {
        0x5018 // PSH + ACK
    };
    pkt.extend_from_slice(&flags.to_be_bytes());
    pkt.extend_from_slice(&[0xff, 0xff]); // Window size 65535
    pkt.extend_from_slice(&[0x00, 0x00]); // Checksum
    pkt.extend_from_slice(&[0x00, 0x00]); // Urgent pointer

    // Payload (20 bytes)
    pkt.extend_from_slice(b"GET /index.html HTTP");

    pkt
}
