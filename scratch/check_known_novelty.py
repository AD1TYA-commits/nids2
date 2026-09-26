import numpy as np
import pandas as pd
from nids_ae_dsp.bundle_io import load_aedsp_bundle
from nids_ae_dsp.core.classifier import AEDSPClassifier
from nids_ae_dsp.core.anomaly_detector import AEDSPAnomalyDetector
from nids_ae_dsp.core.risk_engine import RiskEngine, SecurityDecision
from nids_ae_dsp.core.behavioral import extract_behavioral_indicators

bundle = load_aedsp_bundle('AE-DSP/results/nsl_kdd/model_bundle_seed42')
cols = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]
train_df = pd.read_csv('AE-DSP/data/nsl_kdd/KDDTrain+.txt', header=None, names=cols)
test_df = pd.read_csv('AE-DSP/data/nsl_kdd/KDDTest+.txt', header=None, names=cols)
train_labels = train_df['label'].astype(str).str.strip().str.rstrip('.')
test_labels = test_df['label'].astype(str).str.strip().str.rstrip('.')
train_classes = set(train_labels) - {'normal'}
unseen_classes = sorted(list((set(test_labels) - {'normal'}) - train_classes))
is_known = test_labels.isin(train_classes).values
is_unseen = test_labels.isin(unseen_classes).values
is_normal = (test_labels == 'normal').values

feature_cols = [c for c in cols if c not in ['label', 'difficulty']]
X_raw = test_df[feature_cols]

rep = bundle.extract_representation(X_raw)
detector = AEDSPAnomalyDetector(threshold=bundle.threshold, alpha=bundle.alpha, gamma=bundle.gamma, proto_dist_p95=10.0)
anom = detector.detect(rep)
print('Total known attacks:', sum(is_known))
print('Known attacks with is_anomaly:', sum(anom.is_anomaly[is_known]))
print('Known attacks with is_novel_pattern:', sum(anom.is_novel_pattern[is_known]))
print('Unseen attacks with is_novel_pattern:', sum(anom.is_novel_pattern[is_unseen]))
