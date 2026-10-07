"""
scripts/simulate_vm_traffic.py
==============================
Simulates the demonstration traffic against the isolated target VM (192.168.122.227):
  Phase 1: Normal HTTP / Network traffic
  Phase 2: Controlled Reconnaissance / Port Scan against target VM
"""

import socket
import time
import subprocess
import urllib.request
import json

TARGET_IP = "192.168.122.227"

def send_http_request(port=80):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        s.connect((TARGET_IP, port))
        s.sendall(b"GET / HTTP/1.1\r\nHost: 192.168.122.227\r\nUser-Agent: Mozilla/5.0\r\n\r\n")
        try:
            s.recv(1024)
        except Exception:
            pass
        s.close()
    except Exception:
        pass


def run_port_scan(ports=[21, 22, 23, 25, 80, 110, 135, 139, 443, 445, 1433, 3306, 8080]):
    print(f"[*] Scanning {len(ports)} ports on target VM {TARGET_IP}...")
    for p in ports:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.1)
            s.connect((TARGET_IP, p))
            s.close()
        except Exception:
            pass


def main():
    print("=" * 60)
    print(f"VM TRAFFIC GENERATOR -> TARGET VM: {TARGET_IP}")
    print("=" * 60)

    # Ensure backend is in LIVE mode
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/api/mode?mode=live", data=b"", method="POST")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"[+] Active backend mode: {data['mode_name']} (Interface: {data['interface']})")
    except Exception as e:
        print(f"[-] Backend connection notice: {e}")

    print("\n>>> PHASE 1: GENERATING NORMAL HTTP TRAFFIC <<<")
    for i in range(5):
        print(f"  Sending HTTP request #{i+1} to {TARGET_IP}:80...")
        send_http_request(80)
        time.sleep(0.3)

    print("\n>>> PHASE 2: GENERATING CONTROLLED PORT-SCAN RECONNAISSANCE <<<")
    run_port_scan()

    print("\n[*] Waiting 2 seconds for flow timeout and processing...")
    time.sleep(2.0)

    # Check recent alerts
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/api/history?limit=5") as resp:
            alerts = json.loads(resp.read().decode())
            print(f"\n[+] Recent live alerts captured ({len(alerts)}):")
            for a in alerts:
                print(f"    - Flow: {a['src_ip']}:{a['src_port']} -> {a['dst_ip']}:{a['dst_port']} | Pred: {a['prediction']} | Decision: {a['decision']} | Risk: {a['risk_score']}/100")
    except Exception as e:
        print(f"[-] Error querying alerts: {e}")

    print("\n" + "=" * 60)
    print("DEMO TRAFFIC SIMULATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
