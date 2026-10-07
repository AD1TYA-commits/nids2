use std::fs;
use std::path::Path;
use pcap::{Active, Capture, Device, Offline};

pub enum CaptureSource {
    Live(Capture<Active>),
    Offline(Capture<Offline>),
}

#[derive(Debug, Clone)]
pub struct InterfaceInfo {
    pub name: String,
    pub description: Option<String>,
    pub addresses: Vec<String>,
    pub is_loopback: bool,
    pub is_up: bool,
    pub is_running: bool,
}

/// Query and list available network interfaces.
/// Falls back to `/sys/class/net` if netlink permissions are restricted in non-root environments.
pub fn list_interfaces() -> Result<Vec<InterfaceInfo>, pcap::Error> {
    match Device::list() {
        Ok(devices) => {
            let mut infos = Vec::with_capacity(devices.len());
            for dev in devices {
                let mut addresses = Vec::new();
                for addr in &dev.addresses {
                    addresses.push(addr.addr.to_string());
                }

                let is_loopback = dev.flags.is_loopback();
                let is_up = dev.flags.is_up();
                let is_running = dev.flags.is_running();

                infos.push(InterfaceInfo {
                    name: dev.name,
                    description: dev.desc,
                    addresses,
                    is_loopback,
                    is_up,
                    is_running,
                });
            }
            Ok(infos)
        }
        Err(e) => {
            // Graceful fallback reading /sys/class/net
            if let Ok(sys_infos) = list_interfaces_from_sysfs() {
                if !sys_infos.is_empty() {
                    return Ok(sys_infos);
                }
            }
            Err(e)
        }
    }
}

fn list_interfaces_from_sysfs() -> std::io::Result<Vec<InterfaceInfo>> {
    let mut infos = Vec::new();
    let net_dir = Path::new("/sys/class/net");
    if !net_dir.exists() {
        return Ok(infos);
    }

    for entry in fs::read_dir(net_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();

        let operstate_path = entry.path().join("operstate");
        let is_up = fs::read_to_string(&operstate_path)
            .map(|s| s.trim() == "up" || s.trim() == "unknown")
            .unwrap_or(false);

        let is_loopback = name == "lo";

        let mut addresses = Vec::new();
        let mac_path = entry.path().join("address");
        if let Ok(mac) = fs::read_to_string(&mac_path) {
            let mac_str = mac.trim();
            if !mac_str.is_empty() && mac_str != "00:00:00:00:00:00" {
                addresses.push(format!("MAC: {}", mac_str));
            }
        }

        infos.push(InterfaceInfo {
            name,
            description: None,
            addresses,
            is_loopback,
            is_up,
            is_running: is_up,
        });
    }

    infos.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(infos)
}

impl CaptureSource {
    /// Open a live network interface for packet capture.
    pub fn open_live(
        interface: &str,
        promisc: bool,
        snaplen: i32,
        bpf_filter: Option<&str>,
    ) -> Result<Self, pcap::Error> {
        let mut cap = Capture::from_device(interface)?
            .promisc(promisc)
            .snaplen(snaplen)
            .timeout(100) // 100ms timeout for non-blocking loop / signal handling
            .open()?;

        if let Some(filter) = bpf_filter {
            cap.filter(filter, true)?;
        }

        Ok(CaptureSource::Live(cap))
    }

    /// Open an offline PCAP file for analysis.
    pub fn open_offline(
        pcap_path: &Path,
        bpf_filter: Option<&str>,
    ) -> Result<Self, pcap::Error> {
        let mut cap = Capture::from_file(pcap_path)?;

        if let Some(filter) = bpf_filter {
            cap.filter(filter, true)?;
        }

        Ok(CaptureSource::Offline(cap))
    }

    /// Retrieve the next packet from the capture source.
    /// Returns `Ok(Some((timestamp_micros, packet_data)))` or `Ok(None)` on EOF/timeout.
    pub fn next_packet<'a>(&'a mut self) -> Result<Option<(u64, &'a [u8])>, pcap::Error> {
        match self {
            CaptureSource::Live(cap) => {
                match cap.next_packet() {
                    Ok(packet) => {
                        let sec = packet.header.ts.tv_sec as u64;
                        let usec = packet.header.ts.tv_usec as u64;
                        let ts_micros = sec * 1_000_000 + usec;
                        Ok(Some((ts_micros, packet.data)))
                    }
                    Err(pcap::Error::TimeoutExpired) => Ok(None),
                    Err(e) => Err(e),
                }
            }
            CaptureSource::Offline(cap) => {
                match cap.next_packet() {
                    Ok(packet) => {
                        let sec = packet.header.ts.tv_sec as u64;
                        let usec = packet.header.ts.tv_usec as u64;
                        let ts_micros = sec * 1_000_000 + usec;
                        Ok(Some((ts_micros, packet.data)))
                    }
                    Err(pcap::Error::NoMorePackets) => Ok(None),
                    Err(e) => Err(e),
                }
            }
        }
    }
}
