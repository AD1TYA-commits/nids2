import numpy as np
import pandas as pd
import torch
from nids_ae_dsp.bundle_io import load_aedsp_bundle
from nids_ae_dsp.core.classifier import AEDSPClassifier
from nids_ae_dsp.core.anomaly_detector import AEDSPAnomalyDetector

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
is_unseen = test_labels.isin(unseen_classes).values
is_known = test_labels.isin(train_classes).values

feature_cols = [c for c in cols if c not in ['label', 'difficulty']]
X_raw = test_df[feature_cols]

rep = bundle.extract_representation(X_raw)

clf_state = torch.load('AE-DSP/results/nsl_kdd/model_bundle_seed42/classifier.pt', map_location='cpu')
clf = AEDSPClassifier(representation_dim=rep.features.shape[1], class_names=bundle.classifier_classes if hasattr(bundle, 'classifier_classes') else ['normal', 'dos', 'probe', 'r2l', 'u2r'])
try:
    clf.load_state_dict(clf_state)
except Exception:
    pass

clf_res = clf.predict(rep)
print('Unseen attacks total:', sum(is_unseen))
unseen_preds = pd.Series(clf_res.predicted_class[is_unseen]).value_counts()
print('Unseen attacks predicted classes:\n', unseen_preds)
print('Unseen attacks average confidence:', np.mean(clf_res.confidence[is_unseen]))
known_preds = pd.Series(clf_res.predicted_class[is_known]).value_counts()
print('Known attacks predicted classes:\n', known_preds)
