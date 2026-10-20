import unittest
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score

from src.feature_extraction import get_tfidf_vectorizer
from src.fd_ffo import FDFFO
from src.tl_hho import TLHHO
from src.svr_pnn import SVRPNN


class TestPipelineIntegration(unittest.TestCase):

    def setUp(self):
        # 60 synthetic documents across 3 classes
        self.raw_documents = [
            f"space astronomy planet galaxy orbit telescope launch rocket mission {i}" for i in range(20)
        ] + [
            f"baseball hockey team score game player league season goal championship {i}" for i in range(20)
        ] + [
            f"computer software hardware memory processor disk driver system kernel {i}" for i in range(20)
        ]
        self.labels = np.array([0] * 20 + [1] * 20 + [2] * 20)

    def test_leakage_free_pipeline(self):
        # 1. Train / Test Split FIRST
        X_tr_text, X_te_text, y_tr, y_te = train_test_split(
            self.raw_documents,
            self.labels,
            test_size=0.3,
            random_state=42,
            stratify=self.labels
        )

        # 2. Fit TF-IDF on Train ONLY, transform Test
        vectorizer = get_tfidf_vectorizer(max_features=25)
        X_tr_tfidf = vectorizer.fit_transform(X_tr_text).toarray()
        X_te_tfidf = vectorizer.transform(X_te_text).toarray()

        self.assertEqual(X_tr_tfidf.shape[1], X_te_tfidf.shape[1])

        # 3. Fit FD-FFO on Train ONLY, transform Test
        fdffo = FDFFO(scoring_mode="density_score", density_threshold=0.1)
        X_tr_fdffo = fdffo.fit_transform(X_tr_tfidf)
        X_te_fdffo = fdffo.transform(X_te_tfidf)

        self.assertEqual(X_tr_fdffo.shape[1], X_te_fdffo.shape[1])

        # 4. Fit TL-HHO on Train ONLY, transform Test
        tlhho = TLHHO(population_size=4, max_iterations=2, random_state=42)
        X_tr_tlhho = tlhho.fit_transform(X_tr_fdffo, y_tr)
        X_te_tlhho = tlhho.transform(X_te_fdffo)

        self.assertEqual(X_tr_tlhho.shape[1], X_te_tlhho.shape[1])

        # 5. Fit SVR-PNN on Train ONLY
        model = SVRPNN(gamma=1.0, sigma=1.0, mapping_mode="one_hot")
        model.fit(X_tr_tlhho, y_tr)

        # 6. Evaluate on Test (Unseen)
        predictions = model.predict(X_te_tlhho)
        probabilities = model.predict_proba(X_te_tlhho)

        self.assertEqual(len(predictions), len(y_te))
        self.assertEqual(probabilities.shape, (len(y_te), 3))

        acc = accuracy_score(y_te, predictions)
        nmi = normalized_mutual_info_score(y_te, predictions)
        ari = adjusted_rand_score(y_te, predictions)

        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)
        self.assertGreaterEqual(nmi, 0.0)
        self.assertLessEqual(nmi, 1.0)


if __name__ == "__main__":
    unittest.main()
