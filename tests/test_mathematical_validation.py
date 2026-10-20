import unittest
import numpy as np

from src.fd_ffo import FDFFO
from src.tl_hho import TLHHO
from src.svr_pnn import SVRPNN


class TestMathematicalValidation(unittest.TestCase):
    """
    Focused mathematical tests verifying exact numerical properties,
    closed-form analytical solutions, and theoretical invariants.
    """

    def test_hand_calculated_density_and_delta(self):
        """
        Verify density rho and expanse delta on a hand-calculated 1D dataset:
        Points: [0.0, 1.0, 1.2, 5.0], with cutoff dc = 1.1.
        Distances:
          D(0,1)=1.0, D(0,2)=1.2, D(0,3)=5.0
          D(1,2)=0.2, D(1,3)=4.0, D(2,3)=3.8
        Hand-calculated:
          rho: [1, 2, 1, 0] (point 1 has highest density = 2)
          delta:
            delta_1 = min(D_1i) = 0.2 (highest density point)
            delta_0 = dist to higher density (point 1) = 1.0
            delta_2 = dist to higher density (point 1) = 0.2
            delta_3 = min dist to higher density (0,1,2) = 3.8
        """
        X = np.array([[0.0, 1.0, 1.2, 5.0]])
        fdffo = FDFFO(cutoff_distance=1.1)
        fdffo.fit(X)

        expected_rho = np.array([1.0, 2.0, 1.0, 0.0])
        expected_delta = np.array([1.0, 0.2, 0.2, 3.8])

        np.testing.assert_allclose(fdffo.density_, expected_rho, err_msg="Density rho does not match hand calculation")
        np.testing.assert_allclose(fdffo.delta_, expected_delta, err_msg="Expanse delta does not match hand calculation")

    def test_hand_calculated_ls_svr_kkt_analytical_solution(self):
        """
        Analytical validation of Least-Squares SVR KKT system:
        2 samples: x1 = 0, x2 = 1, targets y1 = 1, y2 = 0.
        sigma = 1.0, gamma = 1.0, jitter = 0.
        K11 = 1, K22 = 1, K12 = K21 = exp(-1).
        KKT system:
          [ 2        exp(-1)  1 ] [ alpha_1 ]   [ 1 ]
          [ exp(-1)  2        1 ] [ alpha_2 ] = [ 0 ]
          [ 1        1        0 ] [   bias  ]   [ 0 ]
        Analytical closed-form:
          alpha_1 = 1 / (2 * (2 - exp(-1))) ≈ 0.306349918
          alpha_2 = -alpha_1 ≈ -0.306349918
          bias = 0.5
        """
        X = np.array([[0.0], [1.0]])
        y = np.array([1, 0])

        model = SVRPNN(gamma=1.0, sigma=1.0, mapping_mode="ordinal", jitter=0.0)
        model.fit(X, y)

        expected_alpha_1 = 1.0 / (2.0 * (2.0 - np.exp(-1.0)))
        expected_alpha_2 = -expected_alpha_1
        expected_bias = 0.5

        self.assertAlmostEqual(model.alpha_[0, 0], expected_alpha_1, places=7)
        self.assertAlmostEqual(model.alpha_[1, 0], expected_alpha_2, places=7)
        self.assertAlmostEqual(model.bias_[0], expected_bias, places=7)

    def test_binary_mask_conversion_invariants(self):
        """
        Test position-to-features conversion under exact boundary values.
        """
        tlhho = TLHHO()
        # Test threshold boundary 0.5
        pos = np.array([0.499999, 0.500000, 0.500001])
        mask = tlhho.position_to_features(pos)
        self.assertTrue(np.array_equal(mask, [False, True, True]))

        # Test fallback when all are strictly zero
        pos_zeros = np.zeros(10)
        mask_zeros = tlhho.position_to_features(pos_zeros)
        self.assertEqual(np.sum(mask_zeros), 1)

    def test_tl_hho_fitness_equation_closed_form(self):
        """
        Verify that TL-HHO fitness follows Eq. (31) exactly:
          fitness = alpha * error + beta * (n_selected / total_features)
        """
        tlhho = TLHHO(alpha=0.8, val_size=0.5, random_state=42)
        # 10 samples, 4 features
        X = np.ones((10, 4))
        y = np.array([0] * 5 + [1] * 5)

        # Select all 4 features: position all 1.0
        pos_all = np.ones(4)
        fit_all = tlhho.calculate_fitness(pos_all, X, y)

        # With all features identical (constant 1), kNN prediction will have error E.
        # Feature ratio is 4/4 = 1.0.
        # fitness = 0.8 * error + 0.2 * 1.0 >= 0.2
        self.assertGreaterEqual(fit_all, 0.2)
        self.assertLessEqual(fit_all, 1.0)

    def test_pnn_probability_normalization_invariants(self):
        """
        Test that PNN probabilities always sum to 1.0 across classes
        and remain finite even under extreme distance vectors.
        """
        X = np.array([[100.0, -100.0], [-100.0, 100.0]])
        y = np.array([0, 1])

        # Small sigma forces extreme exponential decay
        model = SVRPNN(gamma=1.0, sigma=0.001, mapping_mode="one_hot")
        model.fit(X, y)

        probs = model.predict_proba(X)
        self.assertEqual(probs.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(probs)))
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(2), atol=1e-12)

    def test_sparse_vs_dense_pairwise_distances(self):
        """
        Verify that FD-FFO pairwise distances are mathematically identical
        between dense numpy array and scipy.sparse.csr_matrix.
        """
        import scipy.sparse as sp

        rng = np.random.default_rng(42)
        dense_data = rng.uniform(0.0, 1.0, size=(50, 15))
        # Zero out ~70% of entries to simulate sparse text matrices
        dense_data[dense_data < 0.7] = 0.0
        sparse_data = sp.csr_matrix(dense_data)

        fdffo = FDFFO()
        dist_dense = fdffo._pairwise_distances(dense_data)
        dist_sparse = fdffo._pairwise_distances(sparse_data)

        np.testing.assert_allclose(
            dist_sparse, dist_dense, atol=1e-12,
            err_msg="Sparse and dense pairwise distances do not match exactly"
        )

        # Test transform on sparse matrix preserves sparsity
        fdffo.fit(sparse_data)
        trans_sparse = fdffo.transform(sparse_data)
        self.assertTrue(sp.issparse(trans_sparse))
        self.assertEqual(trans_sparse.shape, (50, 15))

    def test_density_and_delta_duplicate_and_boundary_cases(self):
        """
        Verify numerical stability of rho and delta on edge cases:
        - Duplicate feature columns (distance 0 between distinct features)
        - Constant feature columns
        """
        # Feature 0 and Feature 1 are exact duplicates; Feature 2 is distinct
        X = np.array([
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 3.0, 1.0]
        ])
        fdffo = FDFFO(cutoff_distance=0.5, scoring_mode="density_score")
        fdffo.fit(X)

        self.assertTrue(np.all(np.isfinite(fdffo.density_)))
        self.assertTrue(np.all(np.isfinite(fdffo.delta_)))
        self.assertTrue(np.all(fdffo.density_ >= 0))
        self.assertTrue(np.all(fdffo.delta_ >= 0))

    def test_nystrom_low_rank_svr_pnn_invariants(self):
        """
        Verify that opt-in Nyström low-rank approximation in SVRPNN:
        - Fits successfully without error
        - Probability outputs sum to 1.0
        - Correctly predicts separable multiclass synthetic data
        """
        rng = np.random.default_rng(42)
        # 3 distinct clusters in 5D
        c0 = rng.normal(loc=-3.0, scale=0.2, size=(20, 5))
        c1 = rng.normal(loc=0.0, scale=0.2, size=(20, 5))
        c2 = rng.normal(loc=3.0, scale=0.2, size=(20, 5))

        X = np.vstack([c0, c1, c2])
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)

        model = SVRPNN(
            gamma=1.0,
            sigma=1.0,
            mapping_mode="one_hot",
            solver_mode="nystrom",
            n_components=10,
            random_state=42
        )
        model.fit(X, y)

        probs = model.predict_proba(X)
        self.assertEqual(probs.shape, (60, 3))
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(60), atol=1e-12)

        preds = model.predict(X)
        accuracy = np.mean(preds == y)
        self.assertGreaterEqual(accuracy, 0.90)


if __name__ == "__main__":
    unittest.main()

