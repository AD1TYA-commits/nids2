"""
scripts/test_live_packet_to_alert.py
====================================
End-to-end verification of TRUE LIVE CAPTURE pipeline:
  Actual network packets on virbr0 (to/from VM 192.168.122.227)
        ↓
  Rust network-analyzer (-i virbr0)
        ↓
  JSONL flow record on stdout
        ↓
  pipe_to_backend.py
        ↓
  FastAPI /api/classify
        ↓
  adapter.py -> NIDSDetector -> RiskEngine
        ↓
  WebSocket broadcast (/ws)
"""

import asyncio
import json
import subprocess
import time
import socket
import urllib.request
import websockets

TARGET_IP = "192.168.122.227"
INTERFACE = "virbr0"

async def listen_for_alert(ws_uri: str, alert_queue: asyncio.Queue, stop_event: asyncio.Event):
    async with websockets.connect(ws_uri) as ws:
        # Ignore initial state
        await ws.recv()
        while not stop_event.is_set():
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                d = json.loads(msg)
                if d.get("type") == "flow_alert":
                    alert_queue.put_nowait(d["data"])
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                break


async def main():
    print("=" * 70)
    print("STARTING TRUE LIVE CAPTURE END-TO-END VERIFICATION")
    print(f"Target VM: {TARGET_IP} | Interface: {INTERFACE}")
    print("=" * 70)

    # 1. Connect WebSocket client
    alert_queue = asyncio.Queue()
    stop_event = asyncio.Event()
    ws_task = asyncio.create_task(listen_for_alert("ws://127.0.0.1:8000/ws", alert_queue, stop_event))
    await asyncio.sleep(0.5)
    print("1. WebSocket listener connected to ws://127.0.0.1:8000/ws")

    # 2. Launch Rust network-analyzer piped into pipe_to_backend.py
    # Command: ./target/release/network-analyzer -i virbr0 --flow-timeout 1 -f jsonl | python3 scripts/pipe_to_backend.py
    cmd = (
        f"./target/release/network-analyzer -i {INTERFACE} --flow-timeout 1 -f jsonl "
        f"| python3 scripts/pipe_to_backend.py"
    )
    print(f"2. Launching live capture bridge: {cmd}")
    bridge_proc = subprocess.Popen(cmd, shell=True, preexec_fn=None)
    await asyncio.sleep(1.5)  # Wait for pcap initialization

    # 3. Generate actual network packets targeting the VM 192.168.122.227
    print(f"3. Generating real network traffic targeting {TARGET_IP} on {INTERFACE}...")
    # Send TCP probes
    for p in [80, 22, 8080]:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.2)
        try:
            s.connect((TARGET_IP, p))
        except Exception:
            pass
        finally:
            s.close()

    # Also send ICMP pings
    subprocess.run(["ping", "-c", "3", TARGET_IP], capture_output=True)
    print("   Packets transmitted across virbr0!")

    # 4. Wait for flow timeout (1s) and bridge ingestion
    print("4. Waiting for flow engine expiration & WebSocket alert reception...")
    received_alert = None
    for _ in range(8):
        try:
            received_alert = await asyncio.wait_for(alert_queue.get(), timeout=1.0)
            # Check if this alert corresponds to target IP or virbr0 subnet
            if TARGET_IP in [received_alert.get("src_ip"), received_alert.get("dst_ip")]:
                print(f"   Received flow matching target IP: {received_alert.get('flow_id')}")
                break
        except asyncio.TimeoutError:
            pass

    # Stop bridge and listener
    stop_event.set()
    bridge_proc.terminate()
    try:
        bridge_proc.wait(timeout=2)
    except Exception:
        bridge_proc.kill()
    ws_task.cancel()

    print("\n" + "=" * 70)
    if received_alert:
        print("VERIFICATION SUCCESSFUL: COMPLETE REAL PACKET-TO-ALERT RECORD")
        print("=" * 70)
        print(f"Flow ID:            {received_alert.get('flow_id')}")
        print(f"Timestamp:          {received_alert.get('timestamp')}")
        print(f"Source:             {received_alert.get('src_ip')}:{received_alert.get('src_port')}")
        print(f"Destination:        {received_alert.get('dst_ip')}:{received_alert.get('dst_port')}")
        print(f"Protocol:           {received_alert.get('protocol')}")
        print(f"Duration:           {received_alert.get('flow_duration_secs')}s")
        print(f"Packets / Bytes:    {received_alert.get('total_packets')} pkts / {received_alert.get('total_bytes')} bytes")
        print(f"Classifier Pred:    {received_alert.get('prediction')} (confidence: {received_alert.get('confidence')*100:.1f}%)")
        print(f"Anomaly Score:      {received_alert.get('anomaly_score')} (threshold: {received_alert.get('threshold')})")
        print(f"Risk Score:         {received_alert.get('risk_score')} / 100")
        print(f"Decision:           {received_alert.get('decision')}")
        print(f"Action:             {received_alert.get('action')}")
        print(f"Audit Reason:       {received_alert.get('reason')}")
        print("=" * 70)
    else:
        print("FAILED: No live alert received within timeout.")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
