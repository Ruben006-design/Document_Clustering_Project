import numpy as np


class FDFFO:
    """
    Fuzzy Density Fruit Fly Optimization (FD-FFO) Preprocessing.

    Paper Reference:
    "An optimal document clustering method using hybrid optimal feature selection
     and efficient soft computing classifier", Section 4.1, Eqs. (1)-(10).

    Important Specification Note:
    Section 4.1 states:
        "Fuzzy density score below a threshold of 0.4 are eliminated.
         Using fuzzy density fruit fly optimization algorithm 70 % of redundant
         contents are removed."
    However, the paper provides NO mathematical formula defining the "fuzzy density score",
    nor does it explain how 2D fruit fly coordinates select or filter features.
    To avoid fabricating equations while supporting reproducible research, this class
    exposes a configurable `scoring_mode`:
      - 'passthrough' (default): Faithful baseline. Retains all features because the score formula
        is undefined in the paper.
      - 'density_score': Normalized density-delta heuristic (score = norm_rho * norm_delta >= threshold).
      - 'fixed_ratio': Drops the lowest scored features to match the paper's reported 70% removal.
    """

    def __init__(
        self,
        cutoff_distance=0.5,
        density_threshold=0.4,
        scoring_mode="passthrough",
        removal_ratio=0.70,
        population_size=None,
        max_iterations=10,
        random_state=42
    ):
        self.cutoff_distance = cutoff_distance
        self.density_threshold = density_threshold
        self.scoring_mode = scoring_mode
        self.removal_ratio = removal_ratio
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.random_state = random_state

        self.density_ = None
        self.delta_ = None
        self.smell_ = None
        self.best_index_ = None
        self.feature_scores_ = None
        self.selected_features_ = None

        self.n_features_in_ = 0
        self.n_features_out_ = 0
        self.removal_rate_ = 0.0

    # ==========================================================
    # PAIRWISE FEATURE DISTANCES
    # ==========================================================

    def _pairwise_distances(self, X):
        """
        Calculate Euclidean distance between feature columns.

        Each column of X represents one feature across samples.
        Supports both dense NumPy arrays and scipy.sparse matrices
        without unnecessary full-dataset densification.
        Returns a (n_features, n_features) symmetric distance matrix.
        """
        import scipy.sparse as sp

        if sp.issparse(X):
            # Compute column squared norms: (n_features,)
            squared_norms = np.asarray(X.power(2).sum(axis=0)).ravel().astype(float)
            # Compute gram matrix of features: X.T @ X -> (n_features, n_features)
            gram = (X.T @ X).toarray().astype(float)
        else:
            X = np.asarray(X, dtype=float)
            squared_norms = np.sum(X ** 2, axis=0)
            gram = X.T @ X

        distances_squared = (
            squared_norms[:, None]
            + squared_norms[None, :]
            - 2.0 * gram
        )
        distances_squared = np.maximum(distances_squared, 0.0)
        np.fill_diagonal(distances_squared, 0.0)
        return np.sqrt(distances_squared)

    # ==========================================================
    # EQUATIONS 1 & 2: Local Density rho
    # ==========================================================

    def _calculate_density(self, distances):
        """
        Calculate local density rho.
        Paper Eq. (1): rho_i = sum_j Y(D_ji - D_b)
        Paper Eq. (2): Y(y) = 1 if y < 0 else 0
        Note: D_ji - d_c < 0 <=> D_ji < d_c (cutoff distance).
        """
        # Vectorized density calculation: count neighbors within cutoff (excluding self)
        within_cutoff = (distances < self.cutoff_distance).astype(float)
        np.fill_diagonal(within_cutoff, 0.0)
        return np.sum(within_cutoff, axis=0)

    # ==========================================================
    # EQUATION 3: Minimum distance delta to higher density
    # ==========================================================

    def _calculate_delta(self, distances, density):
        """
        Calculate expanse delta according to Eq. (3).
        delta_j = min_{i: rho_i > rho_j} D_ji
        For the point of highest density: min distance to any other point.
        """
        n_points = len(density)
        delta = np.zeros(n_points, dtype=float)

        for j in range(n_points):
            higher_density_indices = np.where(density > density[j])[0]
            if len(higher_density_indices) > 0:
                delta[j] = np.min(distances[j, higher_density_indices])
            else:
                other_distances = np.delete(distances[j], j)
                delta[j] = np.min(other_distances) if len(other_distances) > 0 else 0.0

        return delta

    # ==========================================================
    # EQUATION 4: Fruit Fly Initialization
    # ==========================================================

    def _initialize_fruit_flies(self, n_points):
        """
        Initialize the X/Y coordinates of fruit flies.
        Eq. (4): Y_j = Y_axis + Random, X_j = X_axis + Random
        """
        rng = np.random.default_rng(self.random_state)
        x_axis = 0.0
        y_axis = 0.0

        X_pos = x_axis + rng.uniform(-1.0, 1.0, n_points)
        Y_pos = y_axis + rng.uniform(-1.0, 1.0, n_points)
        return X_pos, Y_pos, rng

    # ==========================================================
    # EQUATIONS 5-10: Distance, Concentration, Smell, Updates
    # ==========================================================

    def _calculate_distance_from_origin(self, X_pos, Y_pos):
        return np.sqrt(X_pos ** 2 + Y_pos ** 2)

    def _calculate_concentration(self, distance):
        safe_dist = np.maximum(distance, 1e-12)
        return 1.0 / safe_dist

    def _calculate_smell(self, concentration):
        return concentration.copy()

    def _select_best_fly(self, smell):
        return int(np.argmax(smell))

    def _update_position(self, X_pos, Y_pos, best_idx):
        best_x = X_pos[best_idx]
        best_y = Y_pos[best_idx]
        return np.full_like(X_pos, best_x), np.full_like(Y_pos, best_y)

    # ==========================================================
    # FIT
    # ==========================================================

    def fit(self, X, y=None):
        """
        Execute FD-FFO feature analysis and selection.
        Supports both dense NumPy arrays and scipy.sparse matrices.
        """
        import scipy.sparse as sp

        if not sp.issparse(X):
            X = np.asarray(X, dtype=float)
            if X.ndim != 2:
                raise ValueError("X must be a 2D matrix.")
        else:
            if len(X.shape) != 2:
                raise ValueError("X must be a 2D sparse matrix.")

        if X.shape[1] < 2:
            raise ValueError("FD-FFO requires at least two features.")
        if self.cutoff_distance <= 0:
            raise ValueError("cutoff_distance must be positive.")

        self.n_features_in_ = X.shape[1]

        # 1. Pairwise distances between features
        distances = self._pairwise_distances(X)

        # 2. Local density and delta calculation (Eqs. 1-3)
        self.density_ = self._calculate_density(distances)
        self.delta_ = self._calculate_delta(distances, self.density_)

        # 3. Fruit Fly Search Loop (Eqs. 4-10)
        n_features = X.shape[1]
        X_pos, Y_pos, rng = self._initialize_fruit_flies(n_features)

        best_index = 0
        best_smell = -np.inf

        for _ in range(self.max_iterations):
            dist = self._calculate_distance_from_origin(X_pos, Y_pos)
            concentration = self._calculate_concentration(dist)
            smell = self._calculate_smell(concentration)

            current_best_idx = self._select_best_fly(smell)
            current_best_smell = smell[current_best_idx]

            if current_best_smell > best_smell:
                best_smell = current_best_smell
                best_index = current_best_idx

            X_pos, Y_pos = self._update_position(X_pos, Y_pos, current_best_idx)
            X_pos += rng.uniform(-0.1, 0.1, n_features)
            Y_pos += rng.uniform(-0.1, 0.1, n_features)

        self.smell_ = smell.copy()
        self.best_index_ = best_index

        # 4. Feature Selection Handling
        if self.scoring_mode == "passthrough":
            # Undefined score in paper; faithful baseline keeps all features
            self.feature_scores_ = np.ones(n_features, dtype=float)
            self.selected_features_ = np.ones(n_features, dtype=bool)

        elif self.scoring_mode == "density_score":
            # Heuristic decision graph score: norm_rho * norm_delta
            max_rho = np.max(self.density_) if np.max(self.density_) > 0 else 1.0
            max_delta = np.max(self.delta_) if np.max(self.delta_) > 0 else 1.0
            norm_rho = self.density_ / max_rho
            norm_delta = self.delta_ / max_delta
            self.feature_scores_ = norm_rho * norm_delta

            mask = self.feature_scores_ >= self.density_threshold
            if not np.any(mask):
                # Ensure at least the top feature is retained
                mask[np.argmax(self.feature_scores_)] = True
            self.selected_features_ = mask

        elif self.scoring_mode == "fixed_ratio":
            # Retain top (1 - removal_ratio) features based on density * delta
            max_rho = np.max(self.density_) if np.max(self.density_) > 0 else 1.0
            max_delta = np.max(self.delta_) if np.max(self.delta_) > 0 else 1.0
            scores = (self.density_ / max_rho) * (self.delta_ / max_delta)
            self.feature_scores_ = scores

            n_keep = max(1, int(np.round(n_features * (1.0 - self.removal_ratio))))
            ranked_indices = np.argsort(scores)[::-1]
            mask = np.zeros(n_features, dtype=bool)
            mask[ranked_indices[:n_keep]] = True
            self.selected_features_ = mask

        else:
            raise ValueError(f"Unknown scoring_mode: '{self.scoring_mode}'")

        self.n_features_out_ = int(np.sum(self.selected_features_))
        self.removal_rate_ = 1.0 - (self.n_features_out_ / self.n_features_in_)

        return self

    def transform(self, X):
        """
        Apply learned feature selection mask to X.
        Supports dense NumPy arrays and scipy.sparse matrices.
        """
        if self.selected_features_ is None:
            raise RuntimeError("FDFFO must be fitted before transform.")
        import scipy.sparse as sp
        if sp.issparse(X):
            return X[:, self.selected_features_]
        X = np.asarray(X, dtype=float)
        return X[:, self.selected_features_]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
