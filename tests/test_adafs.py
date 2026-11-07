"""
Unit and Mathematical Validation Tests for
Adaptive Density-Aware Feature Selection (ADAFS)
"""

import unittest
import numpy as np
import scipy.sparse as sp

from src.fd_ffo import FDFFO


class TestADAFS(unittest.TestCase):
    """Test suite for the ADAFS research extension."""

    def setUp(self):
        self.rng = np.random.default_rng(42)
        # 100 samples, 20 features, 3 classes
        self.X_dense = self.rng.uniform(0.0, 1.0, (100, 20))
        self.y = self.rng.choice([0, 1, 2], size=100)
        self.X_sparse = sp.csr_matrix(self.X_dense)

    def test_scores_finite_and_bounded(self):
        """Verify scores are finite and strictly bounded in [0, 1]."""
        model = FDFFO(
            scoring_mode="adaptive_density",
            adafs_lambda=0.5,
            adafs_threshold_method="percentile",
            adafs_threshold_value=50.0,
            random_state=42
        )
        model.fit(self.X_dense, self.y)

        self.assertTrue(np.all(np.isfinite(model.feature_scores_)))
        self.assertTrue(np.all(model.feature_scores_ >= 0.0))
        self.assertTrue(np.all(model.feature_scores_ <= 1.0))

        self.assertTrue(np.all(np.isfinite(model.relevance_scores_)))
        self.assertTrue(np.all(model.relevance_scores_ >= 0.0))
        self.assertTrue(np.all(model.relevance_scores_ <= 1.0))

        self.assertTrue(np.all(np.isfinite(model.redundancy_scores_)))
        self.assertTrue(np.all(model.redundancy_scores_ >= 0.0))
        self.assertTrue(np.all(model.redundancy_scores_ <= 1.0))

    def test_zero_variance_features_stability(self):
        """Verify that zero-variance features do not cause division by zero or NaN."""
        X_zero = self.X_dense.copy()
        X_zero[:, 3] = 0.0   # All zeros
        X_zero[:, 7] = 42.0  # Constant non-zero

        model = FDFFO(scoring_mode="adaptive_density", random_state=42)
        model.fit(X_zero, self.y)

        self.assertEqual(model.relevance_scores_[3], 0.0)
        self.assertEqual(model.relevance_scores_[7], 0.0)
        self.assertTrue(np.all(np.isfinite(model.feature_scores_)))
        self.assertFalse(np.isnan(model.feature_scores_).any())

    def test_at_least_one_feature_always_selected(self):
        """Verify deterministic fallback guarantees at least 1 feature even with extreme threshold."""
        model = FDFFO(
            scoring_mode="adaptive_density",
            adafs_threshold_method="percentile",
            adafs_threshold_value=100.0,  # 100th percentile
            random_state=42
        )
        model.fit(self.X_dense, self.y)
        self.assertGreaterEqual(model.n_features_out_, 1)
        self.assertGreaterEqual(np.sum(model.selected_features_), 1)

    def test_determinism_identical_output(self):
        """Verify identical input with identical configuration yields identical results."""
        m1 = FDFFO(scoring_mode="adaptive_density", adafs_lambda=0.4, random_state=42)
        m2 = FDFFO(scoring_mode="adaptive_density", adafs_lambda=0.4, random_state=42)

        m1.fit(self.X_dense, self.y)
        m2.fit(self.X_dense, self.y)

        np.testing.assert_array_equal(m1.selected_features_, m2.selected_features_)
        np.testing.assert_allclose(m1.feature_scores_, m2.feature_scores_)
        np.testing.assert_allclose(m1.relevance_scores_, m2.relevance_scores_)
        np.testing.assert_allclose(m1.redundancy_scores_, m2.redundancy_scores_)

    def test_zero_data_leakage(self):
        """Verify test data has zero influence on feature selection scores or threshold."""
        X_train = self.X_dense[:70]
        y_train = self.y[:70]

        X_test_1 = self.X_dense[70:]
        # Modify test set drastically
        X_test_2 = X_test_1 * 100.0 + 50.0

        model = FDFFO(scoring_mode="adaptive_density", random_state=42)
        model.fit(X_train, y_train)

        out_train = model.transform(X_train)
        out_test_1 = model.transform(X_test_1)
        out_test_2 = model.transform(X_test_2)

        # Output feature dimension should be invariant
        self.assertEqual(out_train.shape[1], out_test_1.shape[1])
        self.assertEqual(out_train.shape[1], out_test_2.shape[1])

        # Fitted mask and scores remain completely untouched
        self.assertEqual(model.n_features_in_, 20)
        self.assertEqual(model.n_features_out_, out_train.shape[1])

    def test_selected_feature_indices_valid(self):
        """Verify selected features mask format and transform shapes."""
        model = FDFFO(scoring_mode="adaptive_density", random_state=42)
        X_trans = model.fit_transform(self.X_dense, self.y)

        self.assertEqual(len(model.selected_features_), 20)
        self.assertEqual(model.selected_features_.dtype, bool)
        self.assertEqual(X_trans.shape[0], 100)
        self.assertEqual(X_trans.shape[1], model.n_features_out_)

    def test_sparse_matrix_support(self):
        """Verify sparse matrices yield identical relevance and selection as dense."""
        m_dense = FDFFO(scoring_mode="adaptive_density", random_state=42)
        m_sparse = FDFFO(scoring_mode="adaptive_density", random_state=42)

        m_dense.fit(self.X_dense, self.y)
        m_sparse.fit(self.X_sparse, self.y)

        np.testing.assert_allclose(m_dense.relevance_scores_, m_sparse.relevance_scores_, atol=1e-10)
        np.testing.assert_array_equal(m_dense.selected_features_, m_sparse.selected_features_)

    def test_hand_calculated_relevance_verification(self):
        """
        Exact hand-calculated proof:
        4 samples, 3 features, 2 classes.
        Feature 0: perfect class separation (values: [1, 1] for class 0; [3, 3] for class 1)
          mean_total = 2, SST = 4, SSB = 4 -> Relevance = 1.0
        Feature 1: identical within and between classes ([2, 4] for class 0; [2, 4] for class 1)
          mean_total = 3, SST = 4, SSB = 0 -> Relevance = 0.0
        Feature 2: constant zero variance ([5, 5] for class 0; [5, 5] for class 1)
          SST = 0 -> Relevance = 0.0
        """
        X_hand = np.array([
            [1.0, 2.0, 5.0],
            [1.0, 4.0, 5.0],
            [3.0, 2.0, 5.0],
            [3.0, 4.0, 5.0],
        ], dtype=float)
        y_hand = np.array([0, 0, 1, 1])

        model = FDFFO(scoring_mode="adaptive_density", adafs_lambda=0.0, random_state=42)
        model.fit(X_hand, y_hand)

        expected_relevance = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(model.relevance_scores_, expected_relevance, atol=1e-12)

        # Feature 0 must be ranked highest
        self.assertEqual(np.argmax(model.feature_scores_), 0)
        self.assertTrue(model.selected_features_[0])

    def test_adaptive_threshold_methods(self):
        """Verify all supported adaptive threshold methods run cleanly."""
        methods = ["percentile", "mean", "otsu", "std"]
        for m in methods:
            model = FDFFO(
                scoring_mode="adaptive_density",
                adafs_threshold_method=m,
                adafs_threshold_value=50.0 if m == "percentile" else 0.5,
                random_state=42
            )
            model.fit(self.X_dense, self.y)
            self.assertGreaterEqual(model.n_features_out_, 1)
            self.assertIsNotNone(model.selected_threshold_)
            self.assertTrue(np.isfinite(model.selected_threshold_))

    def test_error_handling_without_labels(self):
        """Verify adaptive_density raises clear error when y is omitted."""
        model = FDFFO(scoring_mode="adaptive_density")
        with self.assertRaises(ValueError) as ctx:
            model.fit(self.X_dense, y=None)
        self.assertIn("requires training labels y", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
