import unittest
import numpy as np
from src.fd_ffo import FDFFO


class TestFDFFO(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        # 30 samples, 10 features
        self.X = rng.standard_normal((30, 10))

    def test_pairwise_distances(self):
        fdffo = FDFFO()
        dists = fdffo._pairwise_distances(self.X)
        self.assertEqual(dists.shape, (10, 10))
        self.assertTrue(np.all(dists >= 0.0))
        self.assertTrue(np.allclose(dists, dists.T))
        self.assertTrue(np.allclose(np.diag(dists), 0.0))

    def test_density_and_delta(self):
        fdffo = FDFFO(cutoff_distance=5.0)
        dists = fdffo._pairwise_distances(self.X)
        density = fdffo._calculate_density(dists)
        delta = fdffo._calculate_delta(dists, density)

        self.assertEqual(len(density), 10)
        self.assertEqual(len(delta), 10)
        self.assertTrue(np.all(density >= 0.0))
        self.assertTrue(np.all(delta >= 0.0))

    def test_passthrough_mode(self):
        fdffo = FDFFO(scoring_mode="passthrough")
        X_out = fdffo.fit_transform(self.X)
        self.assertEqual(X_out.shape, self.X.shape)
        self.assertEqual(fdffo.n_features_out_, 10)
        self.assertEqual(fdffo.removal_rate_, 0.0)

    def test_density_score_mode(self):
        fdffo = FDFFO(scoring_mode="density_score", density_threshold=0.3)
        X_out = fdffo.fit_transform(self.X)
        self.assertLessEqual(X_out.shape[1], self.X.shape[1])
        self.assertGreaterEqual(X_out.shape[1], 1)
        self.assertEqual(len(fdffo.selected_features_), 10)

    def test_fixed_ratio_mode(self):
        fdffo = FDFFO(scoring_mode="fixed_ratio", removal_ratio=0.5)
        X_out = fdffo.fit_transform(self.X)
        self.assertEqual(X_out.shape[1], 5)

    def test_reproducibility(self):
        f1 = FDFFO(random_state=123).fit(self.X)
        f2 = FDFFO(random_state=123).fit(self.X)
        self.assertEqual(f1.best_index_, f2.best_index_)
        self.assertTrue(np.allclose(f1.density_, f2.density_))
        self.assertTrue(np.allclose(f1.delta_, f2.delta_))


if __name__ == "__main__":
    unittest.main()
