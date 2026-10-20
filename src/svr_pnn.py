import numpy as np


class SVRPNN:
    """
    Support Vector Regression - Probabilistic Neural Network (SVR-PNN).

    Paper Reference:
    "An optimal document clustering method using hybrid optimal feature selection
     and efficient soft computing classifier", Section 4.3, Eqs. (32)-(43).

    Important Specification & Ambiguity Note:
    Equations (32)-(40) formulate standard Least-Squares SVR (LS-SVR).
    However, the paper NEVER defines:
      1. What numerical target value to assign in a multiclass setting (e.g. 20 Newsgroups).
      2. How the scalar SVR output connects mathematically to the vector activations in PNN.

    To resolve this without inventing arbitrary equations, two modes are supported:
      - 'one_hot' (default, recommended): Solves LS-SVR with a one-hot class indicator
        matrix target. This maps documents into a C-dimensional continuous class embedding
        without imposing any artificial ordinal ordering on nominal categories.
      - 'ordinal' (baseline): Uses scalar class indices (0, 1, ..., C-1) as the 1D regression
        target, matching the previous baseline implementation.
    """

    def __init__(
        self,
        gamma=1.0,
        sigma=1.0,
        mapping_mode="one_hot",
        solver_mode="exact",
        n_components=200,
        batch_size=500,
        random_state=42,
        jitter=1e-8
    ):
        self.gamma = gamma
        self.sigma = sigma
        self.mapping_mode = mapping_mode
        self.solver_mode = solver_mode
        self.n_components = n_components
        self.batch_size = batch_size
        self.random_state = random_state
        self.jitter = jitter

        self.alpha_ = None
        self.bias_ = None
        self.weights_ = None
        self.landmarks_ = None
        self.nystrom_projection_ = None

        self.X_train_ = None
        self.y_train_ = None

        self.classes_ = None
        self.class_to_index_ = None

    @staticmethod
    def estimate_resource_requirements(n_samples, n_components=200):
        """
        Estimate memory and compute requirements for exact and Nyström kernel solves.

        Returns:
            dict containing estimated RAM and FLOPs for exact vs Nyström solvers.
        """
        exact_mb = (n_samples ** 2 * 8) / (1024 * 1024)
        exact_flops = (1.0 / 3.0) * (n_samples ** 3)
        nystrom_m = min(n_samples, n_components)
        nystrom_mb = (n_samples * nystrom_m * 8 + nystrom_m ** 2 * 8) / (1024 * 1024)
        nystrom_flops = n_samples * (nystrom_m ** 2) + (1.0 / 3.0) * (nystrom_m ** 3)
        return {
            "n_samples": n_samples,
            "kernel_memory_mb": round(exact_mb, 2),
            "approx_flops": f"{exact_flops:.2e}",
            "is_feasible_standard_pc": n_samples <= 3000,
            "exact_kkt_memory_mb": round(exact_mb, 2),
            "exact_approx_flops": f"{exact_flops:.2e}",
            "is_feasible_exact": n_samples <= 3000,
            "nystrom_components": nystrom_m,
            "nystrom_memory_mb": round(nystrom_mb, 2),
            "nystrom_approx_flops": f"{nystrom_flops:.2e}",
            "is_feasible_nystrom": True
        }

    # ==========================================================
    # Eq. (41) - Gaussian RBF Kernel
    # ==========================================================

    def rbf_kernel(self, X1, X2):
        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)

        sq_norm_1 = np.sum(X1 ** 2, axis=1, keepdims=True)
        sq_norm_2 = np.sum(X2 ** 2, axis=1, keepdims=True).T
        dist_sq = np.maximum(sq_norm_1 + sq_norm_2 - 2.0 * (X1 @ X2.T), 0.0)

        # Paper Eq. (41): exp(- ||y - y_j||^2 / sigma^2)
        bandwidth = self.sigma ** 2
        return np.exp(-dist_sq / bandwidth)

    # ==========================================================
    # SVR Training via KKT System (Eqs. 33-40) / Nystrom Primal
    # ==========================================================

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive.")

        self.X_train_ = X
        self.y_train_ = y

        self.classes_ = np.unique(y)
        self.class_to_index_ = {lbl: i for i, lbl in enumerate(self.classes_)}
        n_samples = len(X)
        n_classes = len(self.classes_)

        # Prepare target representation
        if self.mapping_mode == "one_hot":
            target_matrix = np.zeros((n_samples, n_classes), dtype=float)
            for idx, label in enumerate(y):
                target_matrix[idx, self.class_to_index_[label]] = 1.0
        elif self.mapping_mode == "ordinal":
            target_matrix = np.array(
                [self.class_to_index_[lbl] for lbl in y],
                dtype=float
            ).reshape(-1, 1)
        else:
            raise ValueError(f"Unknown mapping_mode: '{self.mapping_mode}'")

        if self.solver_mode == "exact":
            # Compute full kernel matrix K
            K = self.rbf_kernel(X, X)

            # Construct KKT Linear System (Eqs. 37-39):
            # [ K + I/gamma   1 ] [ alpha ] = [ Y ]
            # [     1^T       0 ] [ bias  ]   [ 0 ]
            reg_matrix = K + (1.0 / self.gamma) * np.eye(n_samples)
            if self.jitter > 0:
                reg_matrix += self.jitter * np.eye(n_samples)

            kkt_matrix = np.zeros((n_samples + 1, n_samples + 1), dtype=float)
            kkt_matrix[:n_samples, :n_samples] = reg_matrix
            kkt_matrix[:n_samples, n_samples] = 1.0
            kkt_matrix[n_samples, :n_samples] = 1.0

            target_kkt = np.zeros((n_samples + 1, target_matrix.shape[1]), dtype=float)
            target_kkt[:n_samples, :] = target_matrix

            try:
                solution = np.linalg.solve(kkt_matrix, target_kkt)
            except np.linalg.LinAlgError:
                solution, _, _, _ = np.linalg.lstsq(kkt_matrix, target_kkt, rcond=1e-7)

            self.alpha_ = solution[:n_samples, :]
            self.bias_ = solution[n_samples, :]

        elif self.solver_mode == "nystrom":
            # Opt-in Nystrom low-rank approximation for scalable training
            m = min(self.n_components, n_samples)
            rng = np.random.default_rng(self.random_state)
            landmark_indices = rng.choice(n_samples, size=m, replace=False)
            self.landmarks_ = X[landmark_indices]

            # Compute kernel matrix between landmarks W: (m, m)
            W = self.rbf_kernel(self.landmarks_, self.landmarks_)
            W += 1e-10 * np.eye(m)
            evals, evecs = np.linalg.eigh(W)
            evals = np.maximum(evals, 1e-12)
            self.nystrom_projection_ = evecs / np.sqrt(evals)[None, :]  # shape: (m, m)

            # Empirical feature map Phi(X) = K(X, landmarks) @ projection: (N, m)
            C = self.rbf_kernel(X, self.landmarks_)
            Phi = C @ self.nystrom_projection_

            # Primal ridge regression for [w; bias]:
            # [ Phi.T @ Phi + (1/gamma) I_m    Phi.T @ 1 ] [ w    ] = [ Phi.T @ Y ]
            # [         1.T @ Phi                  N     ] [ bias ]   [   1.T @ Y ]
            A = np.zeros((m + 1, m + 1), dtype=float)
            A[:m, :m] = Phi.T @ Phi + (1.0 / self.gamma) * np.eye(m)
            if self.jitter > 0:
                A[:m, :m] += self.jitter * np.eye(m)
            sum_Phi = Phi.sum(axis=0)
            A[:m, m] = sum_Phi
            A[m, :m] = sum_Phi
            A[m, m] = float(n_samples)

            B = np.zeros((m + 1, target_matrix.shape[1]), dtype=float)
            B[:m, :] = Phi.T @ target_matrix
            B[m, :] = target_matrix.sum(axis=0)

            try:
                solution = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                solution, _, _, _ = np.linalg.lstsq(A, B, rcond=1e-7)

            self.weights_ = solution[:m, :]  # shape: (m, n_targets)
            self.bias_ = solution[m, :]      # shape: (n_targets,)

        else:
            raise ValueError(f"Unknown solver_mode: '{self.solver_mode}'")

        return self

    # ==========================================================
    # Eq. (40) - SVR Decision Function
    # ==========================================================

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        if self.solver_mode == "exact":
            if self.alpha_ is None:
                raise RuntimeError("SVRPNN must be fitted before decision_function.")
            K = self.rbf_kernel(X, self.X_train_)
            return K @ self.alpha_ + self.bias_
        elif self.solver_mode == "nystrom":
            if self.weights_ is None:
                raise RuntimeError("SVRPNN must be fitted before decision_function.")
            C = self.rbf_kernel(X, self.landmarks_)
            Phi = C @ self.nystrom_projection_
            return Phi @ self.weights_ + self.bias_
        else:
            raise ValueError(f"Unknown solver_mode: '{self.solver_mode}'")

    # ==========================================================
    # Eqs. (42)-(43) - PNN Activations & Probabilities
    # ==========================================================

    def predict_proba(self, X):
        if (self.solver_mode == "exact" and self.alpha_ is None) or \
           (self.solver_mode == "nystrom" and self.weights_ is None):
            raise RuntimeError("SVRPNN must be fitted before prediction.")
        X = np.asarray(X, dtype=float)
        n_test = len(X)
        n_classes = len(self.classes_)

        mapped_train = self.decision_function(self.X_train_)  # shape: (N_train, D_mapped)
        mapped_X = self.decision_function(X)                  # shape: (N_test, D_mapped)

        probabilities = np.zeros((n_test, n_classes), dtype=float)
        batch_size = max(1, self.batch_size)

        train_by_class = []
        for class_idx, class_lbl in enumerate(self.classes_):
            mask = (self.y_train_ == class_lbl)
            m_tr_c = mapped_train[mask]
            sq_tr_c = np.sum(m_tr_c ** 2, axis=1, keepdims=True).T if len(m_tr_c) > 0 else None
            train_by_class.append((class_idx, m_tr_c, sq_tr_c))

        two_sigma_sq = 2.0 * (self.sigma ** 2)

        for start_idx in range(0, n_test, batch_size):
            end_idx = min(start_idx + batch_size, n_test)
            m_batch = mapped_X[start_idx:end_idx]
            sq_batch = np.sum(m_batch ** 2, axis=1, keepdims=True)

            for class_idx, m_tr_c, sq_tr_c in train_by_class:
                if m_tr_c is None or len(m_tr_c) == 0:
                    continue
                dist_sq = np.maximum(sq_batch + sq_tr_c - 2.0 * (m_batch @ m_tr_c.T), 0.0)
                activations = np.exp(-dist_sq / two_sigma_sq)
                probabilities[start_idx:end_idx, class_idx] = np.sum(activations, axis=1)

        # Normalize with numerical stability
        row_sums = probabilities.sum(axis=1, keepdims=True)
        zero_mask = (row_sums.ravel() == 0.0)
        if np.any(zero_mask):
            probabilities[zero_mask, :] = 1.0 / n_classes
            row_sums[zero_mask] = 1.0

        probabilities /= row_sums
        return probabilities

    def predict(self, X):
        probs = self.predict_proba(X)
        best_indices = np.argmax(probs, axis=1)
        return self.classes_[best_indices]