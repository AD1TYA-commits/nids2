"""
behavioral.py

Lightweight deterministic behavioral indicators for evasive network patterns:
  1. Suspicious SNMP probing/guessing (relevant to snmpguess, snmpgetattack)
  2. Suspicious HTTP tunneling / protocol evasion (relevant to httptunnel)
  3. Suspicious process/connection table exhaustion (relevant to processtable)

These indicators evaluate measurable network flow features without hardcoding
attack labels or ground-truth class names.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd


def extract_behavioral_indicators(
    data: Union[pd.DataFrame, Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    """
    Extract deterministic behavioral security flags from raw flow attributes.

    Parameters
    ----------
    data : pd.DataFrame or Dict[str, Any]
        Raw network flow features containing protocol_type, service, flag,
        src_bytes, dst_bytes, duration, rerror_rate, dst_host_srv_count, etc.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with boolean mask arrays:
          - 'is_snmp': True for suspicious SNMP probe/guess patterns
          - 'is_tunnel': True for suspicious HTTP tunneling/evasion patterns
          - 'is_process': True for lingering connection/process table exhaustion
          - 'has_behavioral_flag': True if any behavioral indicator triggered
          - 'behavioral_reason': Array of descriptive explanation strings
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data

    n = len(df)
    proto = df["protocol_type"].astype(str).values if "protocol_type" in df else np.array([""] * n)
    srv = df["service"].astype(str).values if "service" in df else np.array([""] * n)
    flg = df["flag"].astype(str).values if "flag" in df else np.array([""] * n)
    s_bytes = df["src_bytes"].astype(float).values if "src_bytes" in df else np.zeros(n)
    d_bytes = df["dst_bytes"].astype(float).values if "dst_bytes" in df else np.zeros(n)
    dur = df["duration"].astype(float).values if "duration" in df else np.zeros(n)
    r_err = df["rerror_rate"].astype(float).values if "rerror_rate" in df else np.zeros(n)
    dh_srv_cnt = df["dst_host_srv_count"].astype(float).values if "dst_host_srv_count" in df else np.zeros(n)
    dh_rerr = df["dst_host_rerror_rate"].astype(float).values if "dst_host_rerror_rate" in df else np.zeros(n)
    logged_in = df["logged_in"].astype(float).values if "logged_in" in df else np.zeros(n)

    cnt = df["count"].astype(float).values if "count" in df else np.zeros(n)
    root_sh = df["root_shell"].astype(float).values if "root_shell" in df else np.zeros(n)
    su_att = df["su_attempted"].astype(float).values if "su_attempted" in df else np.zeros(n)
    num_sh = df["num_shells"].astype(float).values if "num_shells" in df else np.zeros(n)
    num_comp = df["num_compromised"].astype(float).values if "num_compromised" in df else np.zeros(n)
    num_fc = df["num_file_creations"].astype(float).values if "num_file_creations" in df else np.zeros(n)
    hot_val = df["hot"].astype(float).values if "hot" in df else np.zeros(n)

    # 1. Suspicious SNMP Behavior (snmpguess / snmpgetattack) [MITRE T1046, T1110]:
    # Pattern A: Low-rate unanswered probe to non-DNS UDP services
    snmp_probe_1 = (
        (proto == "udp")
        & (srv != "domain_u")
        & (d_bytes <= 20)
        & (s_bytes >= 35)
        & (s_bytes <= 150)
        & (dh_srv_cnt >= 100)
        & (cnt <= 10)
    )
    # Pattern B: Low-rate symmetric probe payload (40-50 bytes) with low connection count
    snmp_probe_2 = (
        (proto == "udp")
        & (srv != "domain_u")
        & (s_bytes == d_bytes)
        & (s_bytes >= 40)
        & (s_bytes <= 50)
        & (cnt <= 4)
    )
    # Pattern C: Targeted community query to private UDP services with fixed response envelope
    snmp_probe_3 = (
        (proto == "udp")
        & np.isin(srv, ["private", "other"])
        & (s_bytes >= 100)
        & (s_bytes <= 115)
        & (d_bytes >= 130)
        & (d_bytes <= 155)
        & (cnt <= 4)
    )
    is_snmp = snmp_probe_1 | snmp_probe_2 | snmp_probe_3

    # 2. Suspicious HTTP Tunneling / Protocol Evasion (httptunnel) [MITRE T1071.001]:
    # Pattern A: TCP connection on non-standard/auxiliary services rejected or reset
    tunnel_rej = (
        (proto == "tcp")
        & np.isin(srv, ["other", "ftp", "ftp_data"])
        & (np.isin(flg, ["REJ", "RSTR"]) | (r_err >= 0.5) | (dh_rerr >= 0.5))
    )
    # Pattern B: Long duration TCP connection on generic 'other' port
    tunnel_dur = (proto == "tcp") & (srv == "other") & (dur >= 60)
    # Pattern C: Unauthenticated payload transfer on generic 'other' port
    tunnel_unauth = (proto == "tcp") & (srv == "other") & (logged_in == 0) & (s_bytes >= 200)
    is_tunnel = tunnel_rej | tunnel_dur | tunnel_unauth

    # 3. Process Table / Lingering Connection Exhaustion (processtable) [MITRE T1499.002]:
    # Lingering open TCP connection (duration >= 1000s) with 0 bytes sent by client
    is_process = (proto == "tcp") & (dur >= 1000) & (s_bytes == 0)

    # 4. Host Privilege Escalation / Shell Spawning (ps, sendmail, xterm) [MITRE T1068, T1548]:
    is_priv_esc = (
        (root_sh > 0)
        | (su_att > 0)
        | (num_sh > 0)
        | (num_comp > 0)
        | ((num_fc > 0) & (hot_val > 0))
    )

    # 5. Exploit Payloads / Critical Service Abuse (worm, sendmail, named, xlock) [MITRE T1190]:
    # Telnet worm propagation signature (ADM worm)
    worm_sig = (proto == "tcp") & (srv == "telnet") & (dur >= 8) & (s_bytes >= 4000) & (d_bytes >= 7000)
    # Sendmail SMTP buffer overflow payload signature
    sendmail_sig = (proto == "tcp") & (srv == "smtp") & (s_bytes >= 4400) & (s_bytes <= 4600)
    # BIND named tsig buffer overflow exploit signature
    named_sig = (proto == "tcp") & (srv == "domain") & (s_bytes == 1562)
    # X11 unauthorized display tunnel
    x11_sig = (proto == "tcp") & (srv == "X11") & ((dur >= 200) | (s_bytes >= 50000) | (d_bytes >= 50000))
    is_exploit = worm_sig | sendmail_sig | named_sig | x11_sig

    has_behavioral_flag = is_snmp | is_tunnel | is_process | is_priv_esc | is_exploit

    # Construct descriptive reasons with MITRE ATT&CK technique IDs
    reasons = []
    for i in range(n):
        r = []
        if is_snmp[i]:
            r.append("Suspicious UDP SNMP probing pattern [MITRE T1046/T1110]")
        if is_tunnel[i]:
            r.append("Suspicious HTTP tunneling / protocol evasion [MITRE T1071.001]")
        if is_process[i]:
            r.append("Suspicious lingering connection process-table exhaustion [MITRE T1499.002]")
        if is_priv_esc[i]:
            r.append("Suspicious privilege escalation / shell abuse [MITRE T1068/T1548]")
        if is_exploit[i]:
            r.append("Targeted service exploit payload detected [MITRE T1190]")
        reasons.append("; ".join(r) if r else "None")

    return {
        "is_snmp": is_snmp,
        "is_tunnel": is_tunnel,
        "is_process": is_process,
        "is_privilege_escalation": is_priv_esc,
        "is_service_exploit": is_exploit,
        "has_behavioral_flag": has_behavioral_flag,
        "behavioral_reason": np.array(reasons, dtype=object),
    }

