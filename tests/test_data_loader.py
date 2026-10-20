import unittest
import numpy as np
from src.data_loader import load_20_newsgroups, get_stratified_sample, load_web_snippets


class TestDataLoader(unittest.TestCase):

    def test_load_20_newsgroups_offline(self):
        docs, labels, names = load_20_newsgroups(subset="train")
        self.assertGreater(len(docs), 0)
        self.assertEqual(len(docs), len(labels))
        self.assertEqual(len(names), 20)
        self.assertEqual(len(np.unique(labels)), 20)

    def test_stratified_sampling(self):
        docs, labels, names = load_20_newsgroups(subset="train")
        n_per_class = 4
        s_docs, s_labels = get_stratified_sample(
            docs, labels, n_samples_per_class=n_per_class, random_state=42
        )
        self.assertEqual(len(s_docs), 20 * n_per_class)
        self.assertEqual(len(s_labels), 20 * n_per_class)
        unique, counts = np.unique(s_labels, return_counts=True)
        self.assertEqual(len(unique), 20)
        self.assertTrue(np.all(counts == n_per_class))

    def test_web_snippets_loader_if_present(self):
        try:
            docs, labels, names = load_web_snippets(subset="test")
            self.assertGreater(len(docs), 0)
            self.assertEqual(len(docs), len(labels))
            self.assertLessEqual(len(names), 8)
        except FileNotFoundError:
            # Archive might not be in test path, skip gracefully
            pass


if __name__ == "__main__":
    unittest.main()
