# Feature Availability Mapping & Specification

This document provides a comprehensive mapping of every target feature from **CICIDS2017**, **UNSW-NB15**, and **NSL-KDD** to the Rust network traffic analysis engine.

## Status Definitions

- **DIRECT**: Directly extracted from raw packet headers (e.g., ports, flags, IP fields).
- **DERIVED**: Calculated in real-time from single or bidirectional packet streams and timestamps (e.g., durations, IAT, length statistics, Welford variance, rates).
- **STATEFUL**: Derived from ongoing state machines or historical rolling windows across multiple packets or flows (e.g., TCP handshake timing, connection counts, error rates, active/idle bursts).
- **UNAVAILABLE**: Feature relies on host-level OS state (e.g., process tables, su attempts, root shell creation, file audits) or dataset ground-truth labels not present in network wire traffic.

---

## 1. CICIDS2017 Feature Mapping

All 71 extractable features from the CICIDS2017 dataset specification are genuinely calculated from real network packets and flow state.

| Dataset | Feature | Status | Source | Calculation |
| :--- | :--- | :--- | :--- | :--- |
| CICIDS2017 | Destination Port | DIRECT | TCP/UDP Header | Extracted from transport layer destination port field. |
| CICIDS2017 | Flow Duration | DERIVED | Packet Timestamps | `last_time - start_time` in microseconds. |
| CICIDS2017 | Total Fwd Packets | DERIVED | Flow Tracker | Count of packets observed in the forward (originator) direction. |
| CICIDS2017 | Total Backward Packets | DERIVED | Flow Tracker | Count of packets observed in the backward (responder) direction. |
| CICIDS2017 | Total Length of Fwd Packets | DERIVED | IP/Wire Length | Cumulative sum of wire lengths for forward packets. |
| CICIDS2017 | Total Length of Bwd Packets | DERIVED | IP/Wire Length | Cumulative sum of wire lengths for backward packets. |
| CICIDS2017 | Fwd Packet Length Max | DERIVED | Streaming Stats | Maximum packet wire length observed forward. |
| CICIDS2017 | Fwd Packet Length Min | DERIVED | Streaming Stats | Minimum packet wire length observed forward. |
| CICIDS2017 | Fwd Packet Length Mean | DERIVED | Welford Accumulator | Numerically stable online mean of forward packet lengths. |
| CICIDS2017 | Fwd Packet Length Std | DERIVED | Welford Accumulator | Square root of sample variance for forward packet lengths. |
| CICIDS2017 | Bwd Packet Length Max | DERIVED | Streaming Stats | Maximum packet wire length observed backward. |
| CICIDS2017 | Bwd Packet Length Min | DERIVED | Streaming Stats | Minimum packet wire length observed backward. |
| CICIDS2017 | Bwd Packet Length Mean | DERIVED | Welford Accumulator | Online mean of backward packet lengths. |
| CICIDS2017 | Bwd Packet Length Std | DERIVED | Welford Accumulator | Sample standard deviation of backward packet lengths. |
| CICIDS2017 | Flow Bytes/s | DERIVED | Flow Tracker | `(total_fwd_bytes + total_bwd_bytes) / duration_seconds`. |
| CICIDS2017 | Flow Packets/s | DERIVED | Flow Tracker | `(total_fwd_packets + total_bwd_packets) / duration_seconds`. |
| CICIDS2017 | Flow IAT Mean | DERIVED | IAT Tracker | Mean inter-arrival time across all packets in the flow. |
| CICIDS2017 | Flow IAT Std | DERIVED | IAT Tracker | Standard deviation of inter-arrival times across all packets. |
| CICIDS2017 | Flow IAT Max | DERIVED | IAT Tracker | Maximum inter-arrival time between consecutive packets in the flow. |
| CICIDS2017 | Flow IAT Min | DERIVED | IAT Tracker | Minimum inter-arrival time between consecutive packets in the flow. |
| CICIDS2017 | Fwd IAT Total | DERIVED | IAT Tracker | Sum of all inter-arrival times between forward packets. |
| CICIDS2017 | Fwd IAT Mean | DERIVED | IAT Tracker | Mean inter-arrival time between forward packets. |
| CICIDS2017 | Fwd IAT Std | DERIVED | IAT Tracker | Standard deviation of forward inter-arrival times. |
| CICIDS2017 | Fwd IAT Max | DERIVED | IAT Tracker | Maximum inter-arrival time between forward packets. |
| CICIDS2017 | Fwd IAT Min | DERIVED | IAT Tracker | Minimum inter-arrival time between forward packets. |
| CICIDS2017 | Bwd IAT Total | DERIVED | IAT Tracker | Sum of all inter-arrival times between backward packets. |
| CICIDS2017 | Bwd IAT Mean | DERIVED | IAT Tracker | Mean inter-arrival time between backward packets. |
| CICIDS2017 | Bwd IAT Std | DERIVED | IAT Tracker | Standard deviation of backward inter-arrival times. |
| CICIDS2017 | Bwd IAT Max | DERIVED | IAT Tracker | Maximum inter-arrival time between backward packets. |
| CICIDS2017 | Bwd IAT Min | DERIVED | IAT Tracker | Minimum inter-arrival time between backward packets. |
| CICIDS2017 | Fwd PSH Flags | DIRECT | TCP Header | Count of forward packets with the PSH flag set. |
| CICIDS2017 | Bwd PSH Flags | DIRECT | TCP Header | Count of backward packets with the PSH flag set. |
| CICIDS2017 | Fwd URG Flags | DIRECT | TCP Header | Count of forward packets with the URG flag set. |
| CICIDS2017 | Bwd URG Flags | DIRECT | TCP Header | Count of backward packets with the URG flag set. |
| CICIDS2017 | Fwd Header Length | DERIVED | IP + L4 Headers | Cumulative sum of IP header and transport header bytes forward. |
| CICIDS2017 | Bwd Header Length | DERIVED | IP + L4 Headers | Cumulative sum of IP header and transport header bytes backward. |
| CICIDS2017 | Fwd Packets/s | DERIVED | Flow Tracker | `total_fwd_packets / duration_seconds`. |
| CICIDS2017 | Bwd Packets/s | DERIVED | Flow Tracker | `total_bwd_packets / duration_seconds`. |
| CICIDS2017 | Min Packet Length | DERIVED | Streaming Stats | Minimum wire length across all packets in the flow. |
| CICIDS2017 | Max Packet Length | DERIVED | Streaming Stats | Maximum wire length across all packets in the flow. |
| CICIDS2017 | Packet Length Mean | DERIVED | Welford Accumulator | Mean wire length across all packets in the flow. |
| CICIDS2017 | Packet Length Std | DERIVED | Welford Accumulator | Standard deviation of packet lengths across all packets. |
| CICIDS2017 | Packet Length Variance | DERIVED | Welford Accumulator | Sample variance (`M2 / (N - 1)`) of packet lengths. |
| CICIDS2017 | FIN Flag Count | DIRECT | TCP Header | Total packets with FIN flag set. |
| CICIDS2017 | SYN Flag Count | DIRECT | TCP Header | Total packets with SYN flag set. |
| CICIDS2017 | RST Flag Count | DIRECT | TCP Header | Total packets with RST flag set. |
| CICIDS2017 | PSH Flag Count | DIRECT | TCP Header | Total packets with PSH flag set. |
| CICIDS2017 | ACK Flag Count | DIRECT | TCP Header | Total packets with ACK flag set. |
| CICIDS2017 | URG Flag Count | DIRECT | TCP Header | Total packets with URG flag set. |
| CICIDS2017 | CWE Flag Count | DIRECT | TCP Header | Total packets with CWR (Congestion Window Reduced) flag set. |
| CICIDS2017 | ECE Flag Count | DIRECT | TCP Header | Total packets with ECE (ECN Echo) flag set. |
| CICIDS2017 | Down/Up Ratio | DERIVED | Flow Tracker | `total_bwd_packets / total_fwd_packets`. |
| CICIDS2017 | Average Packet Size | DERIVED | Flow Tracker | `total_flow_bytes / total_flow_packets`. |
| CICIDS2017 | Avg Fwd Segment Size | DERIVED | Flow Tracker | `total_fwd_bytes / total_fwd_packets`. |
| CICIDS2017 | Avg Bwd Segment Size | DERIVED | Flow Tracker | `total_bwd_bytes / total_bwd_packets`. |
| CICIDS2017 | Subflow Fwd Packets | STATEFUL | Flow Tracker | Number of packets in the forward subflow. |
| CICIDS2017 | Subflow Fwd Bytes | STATEFUL | Flow Tracker | Total bytes in the forward subflow. |
| CICIDS2017 | Subflow Bwd Packets | STATEFUL | Flow Tracker | Number of packets in the backward subflow. |
| CICIDS2017 | Subflow Bwd Bytes | STATEFUL | Flow Tracker | Total bytes in the backward subflow. |
| CICIDS2017 | Init_Win_bytes_forward | STATEFUL | TCP Header | Advertised TCP window size from initial forward SYN packet. |
| CICIDS2017 | Init_Win_bytes_backward | STATEFUL | TCP Header | Advertised TCP window size from initial backward SYN-ACK packet. |
| CICIDS2017 | act_data_pkt_fwd | DERIVED | Payload Inspection | Count of forward packets containing non-zero transport payload. |
| CICIDS2017 | min_seg_size_forward | DIRECT | TCP Header | Minimum transport header size observed in forward packets. |
| CICIDS2017 | Active Mean | STATEFUL | Burst Tracker | Mean duration of active transmission bursts (IAT <= 1.0s). |
| CICIDS2017 | Active Std | STATEFUL | Burst Tracker | Standard deviation of active burst durations. |
| CICIDS2017 | Active Max | STATEFUL | Burst Tracker | Maximum duration of an active burst. |
| CICIDS2017 | Active Min | STATEFUL | Burst Tracker | Minimum duration of an active burst. |
| CICIDS2017 | Idle Mean | STATEFUL | Burst Tracker | Mean duration of idle gaps between bursts (IAT > 1.0s). |
| CICIDS2017 | Idle Std | STATEFUL | Burst Tracker | Standard deviation of idle periods. |
| CICIDS2017 | Idle Max | STATEFUL | Burst Tracker | Maximum idle gap observed. |
| CICIDS2017 | Idle Min | STATEFUL | Burst Tracker | Minimum idle gap observed. |

---

## 2. UNSW-NB15 Feature Mapping

| Dataset | Feature | Status | Source | Calculation |
| :--- | :--- | :--- | :--- | :--- |
| UNSW-NB15 | dur | DERIVED | Packet Timestamps | Total flow duration in seconds. |
| UNSW-NB15 | proto | DIRECT | IP Header | Protocol identifier string/number (e.g. TCP, UDP, ICMP). |
| UNSW-NB15 | service | DERIVED | Port & DPI | Detected application service (http, ftp, dns, ssl, ssh, smtp, etc.). |
| UNSW-NB15 | state | STATEFUL | TCP/Protocol State | Protocol state: CON, FIN, INT, REQ, RST, ECO, CLO. |
| UNSW-NB15 | spkts | DERIVED | Flow Tracker | Source (forward) packet count. |
| UNSW-NB15 | dpkts | DERIVED | Flow Tracker | Destination (backward) packet count. |
| UNSW-NB15 | sbytes | DERIVED | Flow Tracker | Source to destination total bytes. |
| UNSW-NB15 | dbytes | DERIVED | Flow Tracker | Destination to source total bytes. |
| UNSW-NB15 | rate | DERIVED | Flow Tracker | Packets per second across the flow. |
| UNSW-NB15 | sttl | DIRECT | IP Header | Source TTL from first forward packet. |
| UNSW-NB15 | dttl | DIRECT | IP Header | Destination TTL from first backward packet. |
| UNSW-NB15 | sload | DERIVED | Flow Tracker | Source bits per second: `(sbytes * 8) / duration`. |
| UNSW-NB15 | dload | DERIVED | Flow Tracker | Destination bits per second: `(dbytes * 8) / duration`. |
| UNSW-NB15 | sloss | STATEFUL | TCP Sequence | Source packet loss / retransmissions derived from sequence drops. |
| UNSW-NB15 | dloss | STATEFUL | TCP Sequence | Destination packet loss / retransmissions. |
| UNSW-NB15 | sinpkt | DERIVED | IAT Tracker | Source inter-arrival time in milliseconds. |
| UNSW-NB15 | dinpkt | DERIVED | IAT Tracker | Destination inter-arrival time in milliseconds. |
| UNSW-NB15 | sjit | DERIVED | RFC 3550 Jitter | Source packet jitter computed via sliding IAT variance. |
| UNSW-NB15 | djit | DERIVED | RFC 3550 Jitter | Destination packet jitter. |
| UNSW-NB15 | swin | DIRECT | TCP Header | Last observed source TCP window advertisement. |
| UNSW-NB15 | stcpb | DIRECT | TCP Header | Source TCP base sequence number. |
| UNSW-NB15 | dtcpb | DIRECT | TCP Header | Destination TCP base sequence number. |
| UNSW-NB15 | dwin | DIRECT | TCP Header | Last observed destination TCP window advertisement. |
| UNSW-NB15 | tcprtt | STATEFUL | TCP Handshake | Estimated round trip time: `synack + ackdat`. |
| UNSW-NB15 | synack | STATEFUL | TCP Handshake | Time delay between SYN and SYN-ACK. |
| UNSW-NB15 | ackdat | STATEFUL | TCP Handshake | Time delay between SYN-ACK and completion ACK. |
| UNSW-NB15 | smean | DERIVED | Flow Tracker | Mean size of source packets. |
| UNSW-NB15 | dmean | DERIVED | Flow Tracker | Mean size of destination packets. |
| UNSW-NB15 | trans_depth | DERIVED | DPI | Pipelined application transaction count (HTTP requests). |
| UNSW-NB15 | response_body_len | DERIVED | DPI | HTTP Content-Length header or actual body payload size. |
| UNSW-NB15 | ct_srv_src | STATEFUL | Rolling Window | Connections with same service and source IP in window. |
| UNSW-NB15 | ct_state_ttl | STATEFUL | Rolling Window | Connections with matching state and TTL in window. |
| UNSW-NB15 | ct_dst_ltm | STATEFUL | Rolling Window | Connections to destination IP in window. |
| UNSW-NB15 | ct_src_dport_ltm | STATEFUL | Rolling Window | Connections from source IP to destination port in window. |
| UNSW-NB15 | ct_dst_sport_ltm | STATEFUL | Rolling Window | Connections to destination IP with source port in window. |
| UNSW-NB15 | ct_dst_src_ltm | STATEFUL | Rolling Window | Connections between same source and destination IP in window. |
| UNSW-NB15 | is_ftp_login | DERIVED | DPI | 1 if both USER and PASS observed in session, else 0. |
| UNSW-NB15 | ct_ftp_cmd | DERIVED | DPI | Count of FTP commands observed in session. |
| UNSW-NB15 | ct_flw_http_mthd | DERIVED | DPI | Count of HTTP request methods observed (GET, POST, etc.). |
| UNSW-NB15 | ct_src_ltm | STATEFUL | Rolling Window | Connections from source IP in window. |
| UNSW-NB15 | ct_srv_dst | STATEFUL | Rolling Window | Connections with same service to destination IP in window. |
| UNSW-NB15 | is_sm_ips_ports | DERIVED | Flow Key | 1 if `src_ip == dst_ip && src_port == dst_port` (LAND attack), else 0. |
| UNSW-NB15 | attack_cat | UNAVAILABLE | N/A | Ground-truth attack taxonomy label provided by dataset authors. |
| UNSW-NB15 | label | UNAVAILABLE | N/A | Ground-truth binary classification label (0=normal, 1=attack). |

---

## 3. NSL-KDD Feature Mapping

| Dataset | Feature | Status | Source | Calculation |
| :--- | :--- | :--- | :--- | :--- |
| NSL-KDD | duration | DERIVED | Packet Timestamps | Duration of connection in integer seconds. |
| NSL-KDD | protocol_type | DIRECT | IP Header | Protocol name: `tcp`, `udp`, `icmp`. |
| NSL-KDD | service | DERIVED | Port & DPI | Network service on destination (http, ftp, smtp, dns, private, etc.). |
| NSL-KDD | flag | STATEFUL | TCP State Machine | Connection flag: `SF`, `S0`, `S1`, `S2`, `S3`, `REJ`, `RSTO`, `RSTR`, `RSTOS0`, `SH`, `OTH`. |
| NSL-KDD | src_bytes | DERIVED | Flow Tracker | Bytes sent from source to destination (payload data). |
| NSL-KDD | dst_bytes | DERIVED | Flow Tracker | Bytes sent from destination to source (payload data). |
| NSL-KDD | land | DERIVED | IP/Port Headers | 1 if connection is from/to the same host/port; 0 otherwise. |
| NSL-KDD | wrong_fragment | DERIVED | IP Header | Count of erroneous fragment offset / overlapping fragment packets. |
| NSL-KDD | urgent | DIRECT | TCP Header | Count of urgent packets (URG flag or urgent pointer > 0). |
| NSL-KDD | count | STATEFUL | Rolling Window | Connections to same destination host in past 2s/configured window. |
| NSL-KDD | srv_count | STATEFUL | Rolling Window | Connections to same destination service in past window. |
| NSL-KDD | serror_rate | STATEFUL | Rolling Window | % of connections to same host with SYN errors (flags S0/S1/S2/S3). |
| NSL-KDD | srv_serror_rate | STATEFUL | Rolling Window | % of connections to same service with SYN errors. |
| NSL-KDD | rerror_rate | STATEFUL | Rolling Window | % of connections to same host with REJ errors. |
| NSL-KDD | srv_rerror_rate | STATEFUL | Rolling Window | % of connections to same service with REJ errors. |
| NSL-KDD | same_srv_rate | STATEFUL | Rolling Window | % of connections to same host that are to the same service. |
| NSL-KDD | diff_srv_rate | STATEFUL | Rolling Window | % of connections to same host that are to different services. |
| NSL-KDD | srv_diff_host_rate | STATEFUL | Rolling Window | % of connections to same service that are to different hosts. |
| NSL-KDD | dst_host_count | STATEFUL | Host Table | Count of past connections having same destination host. |
| NSL-KDD | dst_host_srv_count | STATEFUL | Host Table | Count of past connections having same destination host and service. |
| NSL-KDD | dst_host_same_srv_rate | STATEFUL | Host Table | % of connections to same host that are to the same service. |
| NSL-KDD | dst_host_diff_srv_rate | STATEFUL | Host Table | % of connections to same host that are to different services. |
| NSL-KDD | dst_host_same_src_port_rate | STATEFUL | Host Table | % of connections to same host coming from the same source port. |
| NSL-KDD | dst_host_srv_diff_host_rate | STATEFUL | Host Table | % of connections to same service coming from different hosts. |
| NSL-KDD | dst_host_serror_rate | STATEFUL | Host Table | % of connections to same host that had SYN errors. |
| NSL-KDD | dst_host_srv_serror_rate | STATEFUL | Host Table | % of connections to same service that had SYN errors. |
| NSL-KDD | dst_host_rerror_rate | STATEFUL | Host Table | % of connections to same host that had REJ errors. |
| NSL-KDD | dst_host_srv_rerror_rate | STATEFUL | Host Table | % of connections to same service that had REJ errors. |
| NSL-KDD | hot | UNAVAILABLE | Host OS Audit | Number of "hot" indicators (access to system files, directory entering) requires local host OS instrumentation. |
| NSL-KDD | num_failed_logins | UNAVAILABLE | Host Auth Logs | Number of failed login attempts requires application-level decrypted credential auditing or OS syslog. |
| NSL-KDD | logged_in | UNAVAILABLE | Host Auth Logs | 1 if successfully logged in; requires application-level authentication session tracking. |
| NSL-KDD | num_compromised | UNAVAILABLE | Host OS Audit | Number of compromised conditions requires local OS audit trails. |
| NSL-KDD | root_shell | UNAVAILABLE | Host OS Process Table | 1 if root shell is obtained; requires monitoring system process execution. |
| NSL-KDD | su_attempted | UNAVAILABLE | Host OS Auth Logs | 1 if `su root` command attempted; requires local shell/audit logs. |
| NSL-KDD | num_root | UNAVAILABLE | Host OS Audit | Number of root accesses requires local host file access logs. |
| NSL-KDD | num_file_creations | UNAVAILABLE | Host Filesystem | Number of file creation operations requires local filesystem audit. |
| NSL-KDD | num_shells | UNAVAILABLE | Host OS Process Table | Number of shell prompts requires monitoring local process descriptors. |
| NSL-KDD | num_access_files | UNAVAILABLE | Host Filesystem | Number of operations on access control files requires local OS audit. |
| NSL-KDD | num_outbound_cmds | UNAVAILABLE | Host Process Table | Number of outbound commands in an ftp session (always 0 in NSL-KDD). |
| NSL-KDD | is_host_login | UNAVAILABLE | Host Auth Logs | 1 if login belongs to the "host" list; requires local OS user database. |
| NSL-KDD | is_guest_login | UNAVAILABLE | Host Auth Logs | 1 if login is a guest account; requires local OS user database. |

---

## 4. Summary of Feature Coverage

| Dataset | Total Target Features | Implemented Directly / Derived / Stateful | Unavailable (Host/OS / Ground Truth) |
| :--- | :--- | :--- | :--- |
| **CICIDS2017** | 71 | **71 (100%)** | 0 (0%) |
| **UNSW-NB15** | 44 | **42 (95.5%)** | 2 (`attack_cat`, `label`) |
| **NSL-KDD** | 41 | **28 (68.3%)** | 13 (host/OS audit features) |
