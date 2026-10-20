import unittest
import numpy as np
from src.tl_hho import TLHHO


class TestTLHHO(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        # 40 samples, 8 features, 2 classes
        self.X = rng.standard_normal((40, 8))
        self.y = np.array([0] * 20 + [1] * 20)

    def test_position_to_features(self):
        tlhho = TLHHO()
        pos = np.array([0.1, 0.6, 0.4, 0.9])
        mask = tlhho.position_to_features(pos)
        self.assertEqual(len(mask), 4)
        self.assertTrue(np.array_equal(mask, [False, True, False, True]))

        # Test edge case: all below 0.5
        pos_all_low = np.array([0.1, 0.2, 0.05])
        mask_fallback = tlhho.position_to_features(pos_all_low)
        self.assertEqual(np.sum(mask_fallback), 1)
        self.assertTrue(mask_fallback[1])

        # Test edge case: all zeros
        pos_zeros = np.zeros(5)
        mask_zeros = tlhho.position_to_features(pos_zeros)
        self.assertEqual(np.sum(mask_zeros), 1)

    def test_fitness_minimization(self):
        tlhho = TLHHO(population_size=6, max_iterations=3, random_state=42)
        selected_mask = tlhho.fit(self.X, self.y)
        self.assertEqual(len(selected_mask), self.X.shape[1])
        self.assertGreaterEqual(np.sum(selected_mask), 1)
        # Ensure best fitness is recorded and non-negative
        self.assertGreaterEqual(tlhho.best_fitness, 0.0)
        self.assertLessEqual(tlhho.best_fitness, 1.0)
        # History should decrease or stay constant
        self.assertLessEqual(tlhho.fitness_history_[-1], tlhho.fitness_history_[0])

    def test_deterministic_behavior(self):
        t1 = TLHHO(population_size=4, max_iterations=2, random_state=99)
        m1 = t1.fit(self.X, self.y)

        t2 = TLHHO(population_size=4, max_iterations=2, random_state=99)
        m2 = t2.fit(self.X, self.y)

        self.assertTrue(np.array_equal(m1, m2))
        self.assertAlmostEqual(t1.best_fitness, t2.best_fitness)

    def test_calculate_fitness_direct_determinism(self):
        tlhho = TLHHO(random_state=42)
        pos = np.array([0.8, 0.2, 0.9, 0.1, 0.7, 0.3, 0.6, 0.4])
        fit1 = tlhho.calculate_fitness(pos, self.X, self.y)
        fit2 = tlhho.calculate_fitness(pos, self.X, self.y)
        self.assertEqual(fit1, fit2)

    def test_transform_shape(self):
        tlhho = TLHHO(population_size=4, max_iterations=2, random_state=42)
        tlhho.fit(self.X, self.y)
        X_trans = tlhho.transform(self.X)
        self.assertEqual(X_trans.shape[0], self.X.shape[0])
        self.assertEqual(X_trans.shape[1], np.sum(tlhho.selected_features_))


if __name__ == "__main__":
    unittest.main()
