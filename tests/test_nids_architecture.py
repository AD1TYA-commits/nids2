"""
test_nids_architecture.py

Comprehensive test suite verifying the AE-DSP NIDS Architecture:
1. AutoEncoder & AE-DSP Representation extraction
2. Classifier (Component 1)
3. Anomaly Detector (Component 2)
4. Risk Engine (NORMAL, SUSPICIOUS, MALICIOUS, UNKNOWN ATTACK)
5. End-to-end pipeline execution and persistence
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from nids_ae_dsp.core.ae_dsp_model import AutoEncoder, PretrainedAEDSP, AEDSPRepresentation
from nids_ae_dsp.core.classifier import AEDSPClassifier
from nids_ae_dsp.core.anomaly_detector import AEDSPAnomalyDetector
from nids_ae_dsp.core.risk_engine import RiskEngine, SecurityDecision
from nids_ae_dsp.pipeline import AEDSP_NIDS
from nids_ae_dsp.trainer import build_aedsp_nids


class TestAEDSPNIDSArchitecture(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

        self.num_samples = 400
        self.num_features = 40
        self.topk = 25
        self.latent_dim = 8
        self.proto_k = 6

        # Synthetic feature matrix
        # Normal traffic: standard normal around 0
        self.X_normal = np.random.randn(250, self.num_features).astype(np.float32)
        # Known Attack traffic: shifted clusters
        self.X_attack_dos = np.random.randn(80, self.num_features).astype(np.float32) + 2.5
        self.X_attack_probe = np.random.randn(70, self.num_features).astype(np.float32) - 2.5

        self.X = np.vstack([self.X_normal, self.X_attack_dos, self.X_attack_probe])
        self.y_binary = np.array([0] * 250 + [1] * 150, dtype=np.int64)
        self.y_labels = ["Normal"] * 250 + ["DoS"] * 80 + ["Probe"] * 70

        # Unseen Zero-day attack: strong unexpected deviation in orthogonal dimensions
        self.X_zeroday = np.random.randn(50, self.num_features).astype(np.float32) * 2.0 + 7.0

    def test_01_autoencoder_dimensions(self):
        """Verify AutoEncoder architecture matches AE-DSP specifications."""
        ae = AutoEncoder(input_dim=self.topk, latent_dim=self.latent_dim, hidden_dim=64)
        x_dummy = torch.randn(16, self.topk)
        recon, z = ae(x_dummy)

        self.assertEqual(recon.shape, (16, self.topk))
        self.assertEqual(z.shape, (16, self.latent_dim))

    def test_02_representation_extraction(self):
        """Verify PretrainedAEDSP extracts full feature representation."""
        top_idx = np.arange(self.topk)
        ae_scaler = StandardScaler().fit(self.X_normal[:, top_idx])
        ae_model = AutoEncoder(input_dim=self.topk, latent_dim=self.latent_dim)

        kmeans = KMeans(n_clusters=self.proto_k, random_state=42).fit(self.X_attack_dos[:, top_idx])

        aedsp = PretrainedAEDSP(
            autoencoder=ae_model,
            ae_scaler=ae_scaler,
            top_idx=top_idx,
            mu_normal=self.X_normal[:, top_idx].mean(axis=0),
            std_normal=self.X_normal[:, top_idx].std(axis=0),
            dsp_kmeans=kmeans,
            dsp_sigma=2.0,
            ae_med=0.5,
            ae_scale=0.2,
            dsp_med=0.1,
            dsp_scale=0.05,
            threshold=1.5,
            alpha=0.8,
            gamma=1.8,
        )

        rep = aedsp.extract_representation(self.X, is_already_shap=True)
        self.assertIsInstance(rep, AEDSPRepresentation)
        self.assertEqual(len(rep), len(self.X))
        self.assertEqual(rep.latent.shape, (len(self.X), self.latent_dim))
        self.assertEqual(rep.prototype_similarities.shape, (len(self.X), self.proto_k))
        self.assertEqual(rep.fused_features.shape, (len(self.X), aedsp.representation_dim))

    def test_03_classifier_component(self):
        """Verify Classifier operates on AE-DSP representation."""
        rep_dim = self.latent_dim + self.proto_k + 4
        classifier = AEDSPClassifier(
            representation_dim=rep_dim,
            class_names=["Normal", "DoS", "Probe"],
        )

        dummy_rep = np.random.randn(60, rep_dim).astype(np.float32)
        dummy_labels = ["Normal"] * 20 + ["DoS"] * 20 + ["Probe"] * 20

        classifier.fit(dummy_rep, dummy_labels, epochs=15)
        res = classifier.predict(dummy_rep)

        self.assertEqual(len(res.predicted_class), 60)
        self.assertTrue(np.all(res.confidence >= 0.0) and np.all(res.confidence <= 1.0))
        self.assertEqual(res.probabilities.shape, (60, 3))
        self.assertTrue(np.all(res.entropy >= 0.0))

    def test_04_anomaly_detector_component(self):
        """Verify Anomaly Detector identifies anomalies and zero-day novelty."""
        detector = AEDSPAnomalyDetector(
            threshold=2.0,
            alpha=0.8,
            gamma=1.8,
            proto_dist_p95=3.0,
            novelty_sensitivity=1.2,
        )

        # Create mock AEDSPRepresentation with normal, known attack, and zero-day
        n = 3
        rep = AEDSPRepresentation(
            latent=np.zeros((n, 8), dtype=np.float32),
            reconstruction=np.zeros((n, 10), dtype=np.float32),
            residuals_squared=np.zeros((n, 10), dtype=np.float32),
            ae_score=np.array([0.1, 1.5, 4.0]),
            ae_z=np.array([0.1, 2.0, 5.0]),  # Normal, Known attack, Zero-day
            delta_shap=np.zeros((n, 10), dtype=np.float32),
            prototype_distances=np.zeros((n, 4), dtype=np.float32),
            prototype_similarities=np.array([[0.1, 0.1, 0.1, 0.1], [0.9, 0.2, 0.1, 0.1], [0.01, 0.01, 0.01, 0.01]]),
            nearest_proto_id=np.array([0, 0, 2]),
            nearest_proto_dist=np.array([4.0, 1.2, 6.5]),  # Zero-day has massive distance to attack prototypes
            dsp_score=np.array([0.1, 0.9, 0.01]),
            dsp_z=np.array([0.0, 2.5, -0.5]),
            fusion_score=np.array([0.08, 6.1, 3.1]),  # Normal is below th, Attack and Zero-day are above th=2.0
            fused_features=np.zeros((n, 16), dtype=np.float32),
        )

        res = detector.detect(rep)
        self.assertFalse(res.is_anomaly[0])  # Normal
        self.assertTrue(res.is_anomaly[1])   # Known attack
        self.assertTrue(res.is_anomaly[2])   # Zero-day anomaly
        self.assertTrue(res.is_novel_pattern[2])  # Zero-day flagged as novel

    def test_05_risk_engine_decisions(self):
        """Verify Risk Engine outputs all 4 required security decisions."""
        engine = RiskEngine()

        # 1. NORMAL
        d1 = engine.evaluate_sample(
            pred_class="Normal",
            confidence=0.95,
            anomaly_score=0.4,
            threshold=2.0,
            is_anomaly=False,
            ae_z=0.2,
            dsp_z=0.1,
            nearest_proto_id=0,
            nearest_proto_dist=4.0,
            is_novel_pattern=False,
        )
        self.assertEqual(d1.decision, SecurityDecision.NORMAL)
        self.assertLessEqual(d1.risk_score, 25.0)

        # 2. MALICIOUS
        d2 = engine.evaluate_sample(
            pred_class="DoS",
            confidence=0.92,
            anomaly_score=5.5,
            threshold=2.0,
            is_anomaly=True,
            ae_z=2.5,
            dsp_z=2.0,
            nearest_proto_id=1,
            nearest_proto_dist=0.8,
            is_novel_pattern=False,
        )
        self.assertEqual(d2.decision, SecurityDecision.MALICIOUS)
        self.assertGreaterEqual(d2.risk_score, 75.0)

        # 3. UNKNOWN ATTACK (Evasion / Zero-Day)
        # Anomaly score is high, but classifier thought it was 'Normal'
        d3 = engine.evaluate_sample(
            pred_class="Normal",
            confidence=0.85,
            anomaly_score=4.2,
            threshold=2.0,
            is_anomaly=True,
            ae_z=3.8,
            dsp_z=-0.2,
            nearest_proto_id=3,
            nearest_proto_dist=5.5,
            is_novel_pattern=True,
        )
        self.assertEqual(d3.decision, SecurityDecision.UNKNOWN_ATTACK)
        self.assertTrue(d3.is_unknown_attack)
        self.assertGreaterEqual(d3.risk_score, 85.0)

        # 4. SUSPICIOUS (Borderline anomaly)
        d4 = engine.evaluate_sample(
            pred_class="Normal",
            confidence=0.55,
            anomaly_score=1.85,  # Close to threshold 2.0
            threshold=2.0,
            is_anomaly=False,
            ae_z=1.2,
            dsp_z=0.4,
            nearest_proto_id=0,
            nearest_proto_dist=2.5,
            is_novel_pattern=False,
        )
        self.assertEqual(d4.decision, SecurityDecision.SUSPICIOUS)
        self.assertGreaterEqual(d4.risk_score, 35.0)
        self.assertLessEqual(d4.risk_score, 74.0)

    def test_06_end_to_end_training_and_pipeline(self):
        """Verify full builder, training, evaluation, saving and loading."""
        pipeline = build_aedsp_nids(
            X_train_pre=self.X,
            y_train_binary=self.y_binary,
            y_train_labels=self.y_labels,
            shap_train=self.X,  # Using pre-simulated space
            topk=self.topk,
            proto_k=self.proto_k,
            latent_dim=self.latent_dim,
            epochs=15,
            classifier_epochs=15,
            verbose=False,
        )

        self.assertIsInstance(pipeline, AEDSP_NIDS)

        # Test on Normal traffic
        normal_decisions = pipeline.predict(self.X_normal[:10], is_already_shap=True)
        self.assertEqual(len(normal_decisions), 10)
        # Most normal should be NORMAL (at least 70% given 75th percentile threshold)
        normal_count = sum(1 for d in normal_decisions if d.decision == SecurityDecision.NORMAL)
        self.assertGreaterEqual(normal_count, 7)

        # Test on Known DoS attack
        dos_decisions = pipeline.predict(self.X_attack_dos[:10], is_already_shap=True)
        dos_count = sum(1 for d in dos_decisions if d.decision == SecurityDecision.MALICIOUS)
        self.assertGreater(dos_count, 6)

        # Test on Zero-day traffic
        zeroday_decisions = pipeline.predict(self.X_zeroday[:10], is_already_shap=True)
        # Zero-day traffic should trigger UNKNOWN ATTACK
        unknown_count = sum(1 for d in zeroday_decisions if d.decision == SecurityDecision.UNKNOWN_ATTACK)
        self.assertGreater(unknown_count, 5)

        # Test Pipeline Serialization & Deserialization
        tmp_dir = tempfile.mkdtemp()
        try:
            pipeline.save(tmp_dir)
            reloaded = AEDSP_NIDS.load(tmp_dir)
            self.assertEqual(reloaded.ae_dsp.topk, pipeline.ae_dsp.topk)
            self.assertEqual(reloaded.classifier.class_names, pipeline.classifier.class_names)

            reloaded_decisions = reloaded.predict(self.X_zeroday[:5], is_already_shap=True)
            self.assertEqual(len(reloaded_decisions), 5)
        finally:
            shutil.rmtree(tmp_dir)

    def test_07_load_bundle_from_official_script(self):
        """Verify loading an AE-DSP bundle generated by the official script."""
        bundle_path = os.path.join("AE-DSP", "results", "nsl_kdd", "model_bundle_seed42")
        if not os.path.exists(bundle_path):
            self.skipTest(f"Bundle not found at {bundle_path}")

        pipeline = AEDSP_NIDS.from_aedsp_bundle(
            bundle_dir=bundle_path,
            class_names=["Normal", "DoS", "Probe", "R2L", "U2R"],
        )
        self.assertEqual(pipeline.ae_dsp.topk, 50)
        self.assertEqual(pipeline.ae_dsp.proto_k, 12)
        self.assertEqual(pipeline.ae_dsp.latent_dim, 8)
        self.assertAlmostEqual(pipeline.ae_dsp.alpha, 0.8)
        self.assertAlmostEqual(pipeline.ae_dsp.gamma, 1.8)


if __name__ == "__main__":
    unittest.main()
