import unittest
import numpy as np
from src.svr_pnn import SVRPNN


class TestSVRPNN(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        # 3 classes, 15 samples each, 6 features
        self.X = rng.standard_normal((45, 6))
        self.y = np.array([0] * 15 + [1] * 15 + [2] * 15)

    def test_rbf_kernel_properties(self):
        model = SVRPNN(sigma=1.0)
        K = model.rbf_kernel(self.X, self.X)
        self.assertEqual(K.shape, (45, 45))
        self.assertTrue(np.allclose(K, K.T))
        self.assertTrue(np.allclose(np.diag(K), 1.0))
        self.assertTrue(np.all(K >= 0.0))
        self.assertTrue(np.all(K <= 1.0))

    def test_one_hot_mapping_mode(self):
        model = SVRPNN(gamma=1.0, sigma=1.0, mapping_mode="one_hot")
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        probs = model.predict_proba(self.X)

        self.assertEqual(len(preds), 45)
        self.assertEqual(probs.shape, (45, 3))
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue(np.allclose(np.sum(probs, axis=1), 1.0))

    def test_ordinal_mapping_mode(self):
        model = SVRPNN(gamma=1.0, sigma=1.0, mapping_mode="ordinal")
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        probs = model.predict_proba(self.X)

        self.assertEqual(len(preds), 45)
        self.assertEqual(probs.shape, (45, 3))
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue(np.allclose(np.sum(probs, axis=1), 1.0))

    def test_duplicate_samples_stability(self):
        # Create dataset with duplicate samples that could make raw kernel ill-conditioned
        X_dup = np.vstack([self.X[:10], self.X[:10]])
        y_dup = np.concatenate([self.y[:10], self.y[:10]])
        model = SVRPNN(gamma=1.0, sigma=1.0, mapping_mode="one_hot")
        model.fit(X_dup, y_dup)
        probs = model.predict_proba(X_dup)
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue(np.allclose(np.sum(probs, axis=1), 1.0))

    def test_small_sigma_underflow_stability(self):
        # Very small sigma could cause underflow in exp(-dist / (2 sigma^2))
        model = SVRPNN(gamma=1.0, sigma=0.01, mapping_mode="one_hot")
        model.fit(self.X, self.y)
        probs = model.predict_proba(self.X)
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue(np.allclose(np.sum(probs, axis=1), 1.0))

    def test_resource_estimation(self):
        res = SVRPNN.estimate_resource_requirements(500)
        self.assertEqual(res["n_samples"], 500)
        self.assertIn("kernel_memory_mb", res)
        self.assertTrue(res["is_feasible_exact"])

        res_large = SVRPNN.estimate_resource_requirements(18846)
        self.assertFalse(res_large["is_feasible_exact"])
        self.assertGreater(res_large["kernel_memory_mb"], 2000)


if __name__ == "__main__":
    unittest.main()
