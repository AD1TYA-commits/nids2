"""
scripts/diagnostic_comparison.py
================================
Runs diagnostic comparison before and after adapter semantic fixes on:
  1. 100 known NORMAL benchmark samples (from working_dataset_all77.parquet)
  2. 20 live_flows.jsonl samples
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

NIDS_ROOT = Path("/home/adityabohra/nids")
NIDS2_ROOT = Path("/home/adityabohra/nids2")
AE_DSP_ROOT = NIDS2_ROOT / "nids_m" / "nids2"

for p in [NIDS_ROOT, AE_DSP_ROOT, NIDS2_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ml.experiments.feature_engineer import RAW_77_FEATURES, add_engineered_features
from ml.inference.predict import NIDSDetector
from nids_ae_dsp.core.risk_engine import RiskEngine

# Import current adapter (WITH semantic fixes)
import backend.adapter as adapter


def rust_flow_to_raw_77_BEFORE(flow: dict) -> dict:
    """Pre-fix implementation: no flag clipping, no 5s active/idle threshold."""
    def parse_proto(raw):
        p = str(raw.get("protocol", raw.get("Protocol", ""))).strip().upper()
        if p == "TCP" or p == "6.0" or p == "6": return 6.0
        if p == "UDP" or p == "17.0" or p == "17": return 17.0
        if "ICMP" in p or p == "1.0" or p == "1": return 1.0
        return 0.0

    return {
        "Protocol": parse_proto(flow),
        "Flow Duration": float(flow.get("flow_duration_micros", flow.get("Flow Duration", 0))),
        "Total Fwd Packets": float(flow.get("total_fwd_packets", flow.get("Total Fwd Packets", 0))),
        "Total Backward Packets": float(flow.get("total_bwd_packets", flow.get("Total Backward Packets", 0))),
        "Fwd Packets Length Total": float(flow.get("total_fwd_bytes", flow.get("Fwd Packets Length Total", 0))),
        "Bwd Packets Length Total": float(flow.get("total_bwd_bytes", flow.get("Bwd Packets Length Total", 0))),
        "Fwd Packet Length Max": float(flow.get("fwd_pkt_len_max", flow.get("Fwd Packet Length Max", 0.0))),
        "Fwd Packet Length Min": float(flow.get("fwd_pkt_len_min", flow.get("Fwd Packet Length Min", 0.0))),
        "Fwd Packet Length Mean": float(flow.get("fwd_pkt_len_mean", flow.get("Fwd Packet Length Mean", 0.0))),
        "Fwd Packet Length Std": float(flow.get("fwd_pkt_len_std", flow.get("Fwd Packet Length Std", 0.0))),
        "Bwd Packet Length Max": float(flow.get("bwd_pkt_len_max", flow.get("Bwd Packet Length Max", 0.0))),
        "Bwd Packet Length Min": float(flow.get("bwd_pkt_len_min", flow.get("Bwd Packet Length Min", 0.0))),
        "Bwd Packet Length Mean": float(flow.get("bwd_pkt_len_mean", flow.get("Bwd Packet Length Mean", 0.0))),
        "Bwd Packet Length Std": float(flow.get("bwd_pkt_len_std", flow.get("Bwd Packet Length Std", 0.0))),
        "Flow Bytes/s": float(flow.get("flow_bytes_per_sec", flow.get("Flow Bytes/s", 0.0))),
        "Flow Packets/s": float(flow.get("flow_packets_per_sec", flow.get("Flow Packets/s", 0.0))),
        "Flow IAT Mean": float(flow.get("flow_iat_mean", flow.get("Flow IAT Mean", 0.0))),
        "Flow IAT Std": float(flow.get("flow_iat_std", flow.get("Flow IAT Std", 0.0))),
        "Flow IAT Max": float(flow.get("flow_iat_max", flow.get("Flow IAT Max", 0.0))),
        "Flow IAT Min": float(flow.get("flow_iat_min", flow.get("Flow IAT Min", 0.0))),
        "Fwd IAT Total": float(flow.get("fwd_iat_total", flow.get("Fwd IAT Total", 0.0))),
        "Fwd IAT Mean": float(flow.get("fwd_iat_mean", flow.get("Fwd IAT Mean", 0.0))),
        "Fwd IAT Std": float(flow.get("fwd_iat_std", flow.get("Fwd IAT Std", 0.0))),
        "Fwd IAT Max": float(flow.get("fwd_iat_max", flow.get("Fwd IAT Max", 0.0))),
        "Fwd IAT Min": float(flow.get("fwd_iat_min", flow.get("Fwd IAT Min", 0.0))),
        "Bwd IAT Total": float(flow.get("bwd_iat_total", flow.get("Bwd IAT Total", 0.0))),
        "Bwd IAT Mean": float(flow.get("bwd_iat_mean", flow.get("Bwd IAT Mean", 0.0))),
        "Bwd IAT Std": float(flow.get("bwd_iat_std", flow.get("Bwd IAT Std", 0.0))),
        "Bwd IAT Max": float(flow.get("bwd_iat_max", flow.get("Bwd IAT Max", 0.0))),
        "Bwd IAT Min": float(flow.get("bwd_iat_min", flow.get("Bwd IAT Min", 0.0))),
        "Fwd PSH Flags": float(flow.get("fwd_psh_flags", flow.get("Fwd PSH Flags", 0))),
        "Bwd PSH Flags": float(flow.get("bwd_psh_flags", flow.get("Bwd PSH Flags", 0))),
        "Fwd URG Flags": float(flow.get("fwd_urg_flags", flow.get("Fwd URG Flags", 0))),
        "Bwd URG Flags": float(flow.get("bwd_urg_flags", flow.get("Bwd URG Flags", 0))),
        "Fwd Header Length": float(flow.get("fwd_header_length", flow.get("Fwd Header Length", 0))),
        "Bwd Header Length": float(flow.get("bwd_header_length", flow.get("Bwd Header Length", 0))),
        "Fwd Packets/s": float(flow.get("fwd_packets_per_sec", flow.get("Fwd Packets/s", 0.0))),
        "Bwd Packets/s": float(flow.get("bwd_packets_per_sec", flow.get("Bwd Packets/s", 0.0))),
        "Packet Length Min": float(flow.get("min_packet_length", flow.get("Packet Length Min", 0.0))),
        "Packet Length Max": float(flow.get("max_packet_length", flow.get("Packet Length Max", 0.0))),
        "Packet Length Mean": float(flow.get("packet_length_mean", flow.get("Packet Length Mean", 0.0))),
        "Packet Length Std": float(flow.get("packet_length_std", flow.get("Packet Length Std", 0.0))),
        "Packet Length Variance": float(flow.get("packet_length_variance", flow.get("Packet Length Variance", 0.0))),
        "FIN Flag Count": float(flow.get("fin_flag_count", flow.get("FIN Flag Count", 0))),
        "SYN Flag Count": float(flow.get("syn_flag_count", flow.get("SYN Flag Count", 0))),
        "RST Flag Count": float(flow.get("rst_flag_count", flow.get("RST Flag Count", 0))),
        "PSH Flag Count": float(flow.get("psh_flag_count", flow.get("PSH Flag Count", 0))),
        "ACK Flag Count": float(flow.get("ack_flag_count", flow.get("ACK Flag Count", 0))),
        "URG Flag Count": float(flow.get("urg_flag_count", flow.get("URG Flag Count", 0))),
        "CWE Flag Count": float(flow.get("cwe_flag_count", flow.get("CWE Flag Count", 0))),
        "ECE Flag Count": float(flow.get("ece_flag_count", flow.get("ECE Flag Count", 0))),
        "Down/Up Ratio": float(flow.get("down_up_ratio", flow.get("Down/Up Ratio", 0.0))),
        "Avg Packet Size": float(flow.get("average_packet_size", flow.get("Avg Packet Size", 0.0))),
        "Avg Fwd Segment Size": float(flow.get("avg_fwd_segment_size", flow.get("Avg Fwd Segment Size", 0.0))),
        "Avg Bwd Segment Size": float(flow.get("avg_bwd_segment_size", flow.get("Avg Bwd Segment Size", 0.0))),
        "Fwd Avg Bytes/Bulk": float(flow.get("fwd_avg_bytes_bulk", flow.get("Fwd Avg Bytes/Bulk", 0.0))),
        "Fwd Avg Packets/Bulk": float(flow.get("fwd_avg_packets_bulk", flow.get("Fwd Avg Packets/Bulk", 0.0))),
        "Fwd Avg Bulk Rate": float(flow.get("fwd_avg_bulk_rate", flow.get("Fwd Avg Bulk Rate", 0.0))),
        "Bwd Avg Bytes/Bulk": float(flow.get("bwd_avg_bytes_bulk", flow.get("Bwd Avg Bytes/Bulk", 0.0))),
        "Bwd Avg Packets/Bulk": float(flow.get("bwd_avg_packets_bulk", flow.get("Bwd Avg Packets/Bulk", 0.0))),
        "Bwd Avg Bulk Rate": float(flow.get("bwd_avg_bulk_rate", flow.get("Bwd Avg Bulk Rate", 0.0))),
        "Subflow Fwd Packets": float(flow.get("subflow_fwd_packets", flow.get("Subflow Fwd Packets", flow.get("total_fwd_packets", 0)))),
        "Subflow Fwd Bytes": float(flow.get("subflow_fwd_bytes", flow.get("Subflow Fwd Bytes", flow.get("total_fwd_bytes", 0)))),
        "Subflow Bwd Packets": float(flow.get("subflow_bwd_packets", flow.get("Subflow Bwd Packets", flow.get("total_bwd_packets", 0)))),
        "Subflow Bwd Bytes": float(flow.get("subflow_bwd_bytes", flow.get("Subflow Bwd Bytes", flow.get("total_bwd_bytes", 0)))),
        "Init Fwd Win Bytes": float(flow.get("init_win_bytes_forward", flow.get("Init Fwd Win Bytes", -1))),
        "Init Bwd Win Bytes": float(flow.get("init_win_bytes_backward", flow.get("Init Bwd Win Bytes", -1))),
        "Fwd Act Data Packets": float(flow.get("act_data_pkt_fwd", flow.get("Fwd Act Data Packets", 0))),
        "Fwd Seg Size Min": float(flow.get("min_seg_size_forward", flow.get("Fwd Seg Size Min", 0))),
        "Active Mean": float(flow.get("active_mean", flow.get("Active Mean", 0.0))),
        "Active Std": float(flow.get("active_std", flow.get("Active Std", 0.0))),
        "Active Max": float(flow.get("active_max", flow.get("Active Max", 0.0))),
        "Active Min": float(flow.get("active_min", flow.get("Active Min", 0.0))),
        "Idle Mean": float(flow.get("idle_mean", flow.get("Idle Mean", 0.0))),
        "Idle Std": float(flow.get("idle_std", flow.get("Idle Std", 0.0))),
        "Idle Max": float(flow.get("idle_max", flow.get("Idle Max", 0.0))),
        "Idle Min": float(flow.get("idle_min", flow.get("Idle Min", 0.0))),
    }


def evaluate_batch(samples, mapper_fn, detector, risk_engine):
    scores = []
    predictions = []
    decisions = []
    risks = []

    for s in samples:
        raw_dict = mapper_fn(s)
        df_raw = pd.DataFrame([raw_dict])[RAW_77_FEATURES]
        m_res = detector.predict(df_raw)[0]

        pred_label = str(m_res["prediction"])
        conf = float(m_res["confidence"])
        is_anom = bool(m_res["anomaly"])
        anom_score = float(m_res["anomaly_score"])

        is_unknown = (pred_label == "UNKNOWN")
        clf_hypo = "Normal" if is_unknown else pred_label

        r_dec = risk_engine.evaluate_sample(
            pred_class=clf_hypo,
            confidence=conf,
            anomaly_score=anom_score,
            threshold=detector.threshold,
            is_anomaly=is_anom,
            ae_z=anom_score,
            dsp_z=-2.5 if is_unknown else 0.5,
            nearest_proto_id=0,
            nearest_proto_dist=8.0 if is_unknown else 2.0,
            is_novel_pattern=is_unknown,
        )

        scores.append(anom_score)
        predictions.append(pred_label)
        decisions.append(r_dec.decision.value)
        risks.append(r_dec.risk_score)

    return {
        "scores": np.array(scores),
        "predictions": pd.Series(predictions).value_counts().to_dict(),
        "decisions": pd.Series(decisions).value_counts().to_dict(),
        "mean_risk": float(np.mean(risks)),
    }


def evaluate_batch_pipeline(samples, nids):
    scores = []
    predictions = []
    decisions = []
    risks = []

    for s in samples:
        res = nids.process_flow(s)
        scores.append(res["anomaly_score"])
        predictions.append(res["prediction"])
        decisions.append(res["decision"])
        risks.append(res["risk_score"])

    return {
        "scores": np.array(scores),
        "predictions": pd.Series(predictions).value_counts().to_dict(),
        "decisions": pd.Series(decisions).value_counts().to_dict(),
        "mean_risk": float(np.mean(risks)),
    }


def main():
    print("Loading models and datasets...")
    detector = NIDSDetector(model_dir="/home/adityabohra/nids/ml/models/production")
    risk_engine = RiskEngine()
    nids = adapter.IntegratedNIDS()

    # 1. 100 Known NORMAL samples from benchmark
    df_bm = pd.read_parquet("/home/adityabohra/nids/ml/data/working_dataset_all77.parquet")
    normal_samples = df_bm[df_bm["StandardLabel"] == "Normal"].iloc[:100].to_dict(orient="records")

    # 2. 20 Live flows samples
    live_samples = []
    with open("/home/adityabohra/nids2/live_flows.jsonl", "r") as f:
        for i, line in enumerate(f):
            if i >= 20:
                break
            live_samples.append(json.loads(line))

    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1: 100 KNOWN NORMAL BENCHMARK SAMPLES")
    print("=" * 70)
    res_norm_before = evaluate_batch(normal_samples, rust_flow_to_raw_77_BEFORE, detector, risk_engine)
    res_norm_after = evaluate_batch_pipeline(normal_samples, nids)

    s_b = res_norm_before["scores"]
    s_a = res_norm_after["scores"]
    print(f"BEFORE: Score Mean={s_b.mean():.4f}, Median={np.median(s_b):.4f}, Min={s_b.min():.4f}, Max={s_b.max():.4f}")
    print(f"        Above Threshold ({detector.threshold:.4f}): {np.sum(s_b > detector.threshold)} / 100 ({np.mean(s_b > detector.threshold)*100:.1f}%)")
    print(f"        Predictions: {res_norm_before['predictions']}")
    print(f"        Decisions:   {res_norm_before['decisions']}")
    print(f"        Mean Risk:   {res_norm_before['mean_risk']:.1f}")

    print(f"\nAFTER:  Score Mean={s_a.mean():.4f}, Median={np.median(s_a):.4f}, Min={s_a.min():.4f}, Max={s_a.max():.4f}")
    print(f"        Above Threshold ({detector.threshold:.4f}): {np.sum(s_a > detector.threshold)} / 100 ({np.mean(s_a > detector.threshold)*100:.1f}%)")
    print(f"        Predictions: {res_norm_after['predictions']}")
    print(f"        Decisions:   {res_norm_after['decisions']}")
    print(f"        Mean Risk:   {res_norm_after['mean_risk']:.1f}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2: 20 LIVE_FLOWS.JSONL SAMPLES")
    print("=" * 70)
    res_live_before = evaluate_batch(live_samples, rust_flow_to_raw_77_BEFORE, detector, risk_engine)
    res_live_after = evaluate_batch_pipeline(live_samples, nids)

    sl_b = res_live_before["scores"]
    sl_a = res_live_after["scores"]
    print(f"BEFORE: Score Mean={sl_b.mean():.4f}, Median={np.median(sl_b):.4f}, Min={sl_b.min():.4f}, Max={sl_b.max():.4f}")
    print(f"        Above Threshold ({detector.threshold:.4f}): {np.sum(sl_b > detector.threshold)} / 20 ({np.mean(sl_b > detector.threshold)*100:.1f}%)")
    print(f"        Predictions: {res_live_before['predictions']}")
    print(f"        Decisions:   {res_live_before['decisions']}")
    print(f"        Mean Risk:   {res_live_before['mean_risk']:.1f}")

    print(f"\nAFTER:  Score Mean={sl_a.mean():.4f}, Median={np.median(sl_a):.4f}, Min={sl_a.min():.4f}, Max={sl_a.max():.4f}")
    print(f"        Above Threshold ({detector.threshold:.4f}): {np.sum(sl_a > detector.threshold)} / 20 ({np.mean(sl_a > detector.threshold)*100:.1f}%)")
    print(f"        Predictions: {res_live_after['predictions']}")
    print(f"        Decisions:   {res_live_after['decisions']}")
    print(f"        Mean Risk:   {res_live_after['mean_risk']:.1f}")

    print("\n" + "=" * 70)
    print("DETAIL FOR 20 LIVE FLOWS (AFTER FIXES)")
    print("=" * 70)
    print(f"{'Flow #':<7} | {'Proto':<5} | {'Duration (s)':<12} | {'Pkts':<5} | {'Bytes':<6} | {'Clf Pred':<10} | {'AE Score':<9} | {'Decision':<14} | {'Risk':<5}")
    print("-" * 80)
    for i, s in enumerate(live_samples):
        res = adapter.IntegratedNIDS().process_flow(s)
        print(f"#{i+1:<6} | {res['protocol']:<5} | {res['flow_duration_secs']:<12.3f} | {res['total_packets']:<5} | {res['total_bytes']:<6} | {res['prediction']:<10} | {res['anomaly_score']:<9.4f} | {res['decision']:<14} | {res['risk_score']:<5.1f}")


if __name__ == "__main__":
    main()
