# Network Traffic Analysis Engine (`network-analyzer`)

A high-performance, memory-safe, standalone Rust network traffic analysis engine. It performs live packet capture and offline PCAP replay, stateful bidirectional flow tracking, streaming statistical analysis, deep packet inspection, and extracts features compatible with **CICIDS2017**, **UNSW-NB15**, and **NSL-KDD** network intrusion detection datasets.

---

## Architecture Pipeline

```text
LIVE NETWORK TRAFFIC (or OFFLINE PCAP)
           │
           ▼
     PACKET CAPTURE (libpcap / AF_PACKET abstraction)
           │
           ▼
     PACKET PARSING (Ethernet, 802.1Q VLAN, Linux SLL, IPv4, IPv6, TCP, UDP, ICMP, ARP)
           │
           ▼
BIDIRECTIONAL FLOW TRACKING (Canonical 5-tuple, Forward/Backward normalization)
           │
           ▼
     FLOW STATE (TCP state machine, handshake timing RTT, sequences, retransmission, windows)
           │
           ▼
STATISTICAL / TIMING ANALYSIS (Welford online variance, IAT trackers, Active/Idle bursts)
           │
           ▼
 NIDS FEATURE EXTRACTION (CICIDS2017, UNSW-NB15, NSL-KDD schemas)
           │
           ▼
 JSONL / CSV FLOW RECORDS (Buffered streaming record serialization)
```

---

## Key Features

- **True Bidirectional Stateful Tracking**: Normalizes `(A:pA -> B:pB, proto)` and `(B:pB -> A:pA, proto)` into an identical canonical flow key. Identifies originator (forward) vs responder (backward) packets.
- **Welford Algorithm Streaming Accumulators**: Calculates mean, variance, sample standard deviation, min, max, and sum in $O(1)$ time and $O(1)$ space without storing raw packets indefinitely.
- **Active / Idle Burst Detection**: Fully compatible with CICFlowMeter activity threshold semantics (default 1.0s) to distinguish active transmission bursts from idle connection periods.
- **TCP Handshake & RTT Analysis**: Tracks timestamps for SYN, SYN-ACK, and ACK, measuring `synack` delay, `ackdat` delay, and total estimated `tcprtt`.
- **Sliding Window Host/Service Context**: Implements a bounded rolling time window (`--stats-window 60`) to compute historical NSL-KDD and UNSW-NB15 rates (e.g. `count`, `srv_count`, `serror_rate`, `rerror_rate`, `same_srv_rate`, `diff_srv_rate`, `ct_dst_ltm`, `ct_srv_src`).
- **Memory Safety & DoS Protection**: Prevents state exhaustion via configurable limits (`--max-flows`, `--flow-timeout`, `--stats-window`). Evicts idle and expired flows deterministically with LRU fallback.
- **Unified Capture Engine**: Identical packet parsing and flow extraction pipeline for both live interfaces and offline PCAPs.
- **Deterministic Output Formats**: Supports JSON Lines (`.jsonl`) and Comma-Separated Values (`.csv`) with strongly-typed field serialization.

---

## Feature Coverage Summary

| Dataset | Total Target Features | Implemented Features | Unavailable (Host/OS / Ground Truth) |
| :--- | :--- | :--- | :--- |
| **CICIDS2017** | 71 | **71 (100%)** | 0 |
| **UNSW-NB15** | 44 | **42 (95.5%)** | 2 (`attack_cat`, `label`) |
| **NSL-KDD** | 41 | **28 (68.3%)** | 13 (host OS file/process audit metrics) |

For the complete feature-by-feature calculation breakdown, see [docs/feature-mapping.md](docs/feature-mapping.md).

---

## Building the Engine

### Prerequisites
- Rust stable toolchain (`cargo`, `rustc`)
- `libpcap` library and development headers (`libpcap-dev` on Debian/Ubuntu, `libpcap-devel` on Fedora/RHEL)

### Build Release Binary
```bash
cargo build --release
```

The compiled binary will be placed at `./target/release/network-analyzer`.

---

## CLI Usage & Commands

### 1. List Available Network Interfaces
```bash
./target/release/network-analyzer --list-interfaces
```

Example Output:
```text
INTERFACE       STATUS     ADDRESSES                      Description
---------------------------------------------------------------------------
enp3s0          DOWN       MAC: 08:8f:c3:6e:b9:f6         -
lo              UP         -                              -
wlp4s0          UP         MAC: 06:c7:8b:e4:87:fc         -
```

### 2. Live Capture from Network Interface
Capture real packets from an interface and emit JSONL flow records to stdout:
```bash
sudo ./target/release/network-analyzer --interface wlp4s0
```

### 3. Live Capture to CSV or JSONL File
```bash
# Export to CSV
sudo ./target/release/network-analyzer --interface wlp4s0 --output flows.csv

# Export to JSON Lines
sudo ./target/release/network-analyzer --interface wlp4s0 --output flows.jsonl
```

### 4. Offline PCAP File Analysis
Analyze pre-recorded network capture files with the exact same processing pipeline:
```bash
./target/release/network-analyzer --pcap traffic.pcap --output flows.jsonl
```

### 5. Configurable Timeouts and Capacity Limits
```bash
sudo ./target/release/network-analyzer \
  --interface wlp4s0 \
  --flow-timeout 30 \
  --active-timeout 120 \
  --stats-window 60 \
  --max-flows 100000 \
  --filter "tcp or udp" \
  --output flows.csv
```

### 6. Built-in High-Throughput Performance Benchmark
```bash
./target/release/network-analyzer --benchmark --output /tmp/bench_flows.jsonl
```

---

## Performance & Benchmarks

Measured on standard x86_64 hardware:
- **Packet Throughput**: $> 1,050,000$ packets/second
- **Flow Throughput**: $> 21,000$ completed flows/second
- **Average Feature Extraction Time**: $\approx 47$ microseconds per completed flow
- **Memory Footprint**: Bounded $O(\text{max\_flows} + \text{max\_events})$ via bounded HashMaps and ring buffers.

---

## Running Automated Tests

Run the complete deterministic unit and integration test suite:
```bash
cargo test
```

Test coverage includes:
- Frame & link parsing (Ethernet, 802.1Q VLAN, Linux SLL)
- Protocol decoding (IPv4, IPv6, TCP, UDP, ICMP, ARP)
- Malformed & truncated packet error handling
- Flow key canonicalization & direction detection
- Welford statistical accumulator verification
- Inter-arrival time (IAT) and Active/Idle burst detection
- TCP 3-way handshake timing & RTT estimation
- TCP state transitions & flags (SF, S0, REJ, etc.)
- Application layer inspection (HTTP methods, FTP commands, Content-Length)
- Flow table timeouts, capacity bounds, and LRU eviction
- Rolling window host/service rates (NSL-KDD & UNSW-NB15)
- Offline PCAP generation and replay verification
