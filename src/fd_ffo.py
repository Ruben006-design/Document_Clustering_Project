
import numpy as np


class FDFFO:
    """
    Fuzzy Density Fruit Fly Optimization (FD-FFO)

    Paper-based implementation of the explicitly described
    density, delta, and fruit-fly search operations.

    Important:
    The paper states that fuzzy-density scores below 0.4
    are eliminated, but does not provide an explicit equation
    defining that fuzzy-density score. Therefore this class
    exposes density_, delta_, smell_, and best_index_ rather
    than inventing a fuzzy-score equation.
    """

    def __init__(
        self,
        cutoff_distance=0.5,
        density_threshold=0.4,
        population_size=None,
        max_iterations=10,
        random_state=42
    ):
        self.cutoff_distance = cutoff_distance
        self.density_threshold = density_threshold
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.random_state = random_state

        self.density_ = None
        self.delta_ = None
        self.smell_ = None
        self.best_index_ = None

        self.selected_features_ = None
        self.feature_scores_ = None

    # ==========================================================
    # PAIRWISE FEATURE DISTANCES
    # ==========================================================

    def _pairwise_distances(self, X):
        """
        Calculate Euclidean distance between TF-IDF feature vectors.

        Each column of X represents one feature.
        Therefore the distance matrix is feature x feature.
        """

        X = np.asarray(X, dtype=float)

        squared_norms = np.sum(X ** 2, axis=0)

        distances_squared = (
            squared_norms[:, None]
            + squared_norms[None, :]
            - 2.0 * (X.T @ X)
        )

        distances_squared = np.maximum(
            distances_squared,
            0.0
        )

        return np.sqrt(distances_squared)

    # ==========================================================
    # EQUATION 1 + EQUATION 2
    # ==========================================================

    def _calculate_density(self, distances):
        """
        Calculate local density rho.

        Paper Eq. (1):
            rho_i = sum_j Y(D_ji - D_b)

        Paper Eq. (2):
            Y(y) = 1, if y < 0
                   0, otherwise

        This implementation uses a positive cutoff distance
        because distances themselves are non-negative.
        """

        n_points = distances.shape[0]

        density = np.zeros(
            n_points,
            dtype=float
        )

        for i in range(n_points):

            for j in range(n_points):

                if i == j:
                    continue

                if (
                    distances[j, i]
                    < self.cutoff_distance
                ):
                    density[i] += 1.0

        return density

    # ==========================================================
    # EQUATION 3
    # ==========================================================

    def _calculate_delta(self, distances, density):
        """
        Calculate delta according to Eq. (3).

        For a point that has another point with higher density:
            delta_j = minimum distance to a higher-density point.

        For the highest-density point:
            delta_j = minimum distance to another point.
        """

        n_points = len(density)

        delta = np.zeros(
            n_points,
            dtype=float
        )

        for j in range(n_points):

            higher_density = np.where(
                density > density[j]
            )[0]

            if len(higher_density) > 0:

                delta[j] = np.min(
                    distances[
                        j,
                        higher_density
                    ]
                )

            else:

                other_distances = np.delete(
                    distances[j],
                    j
                )

                if len(other_distances) > 0:
                    delta[j] = np.min(
                        other_distances
                    )
                else:
                    delta[j] = 0.0

        return delta

    # ==========================================================
    # EQUATION 4
    # ==========================================================

    def _initialize_fruit_flies(self, n_points):
        """
        Initialize the X/Y search axes and fruit-fly positions.

        The paper describes:
            Y_j = Y_axis + random value
            X_j = X_axis + random value
        """

        rng = np.random.default_rng(
            self.random_state
        )

        x_axis = 0.0
        y_axis = 0.0

        X_position = (
            x_axis
            + rng.uniform(
                -1.0,
                1.0,
                n_points
            )
        )

        Y_position = (
            y_axis
            + rng.uniform(
                -1.0,
                1.0,
                n_points
            )
        )

        return (
            X_position,
            Y_position,
            rng
        )

    # ==========================================================
    # EQUATION 5
    # ==========================================================

    def _calculate_distance_from_origin(
        self,
        X_position,
        Y_position
    ):
        """
        Dist_j = sqrt(X_j^2 + Y_j^2)
        """

        distance = np.sqrt(
            X_position ** 2
            + Y_position ** 2
        )

        return distance

    # ==========================================================
    # EQUATION 6
    # ==========================================================

    def _calculate_concentration(
        self,
        distance
    ):
        """
        T_j = 1 / Dist_j
        """

        safe_distance = np.maximum(
            distance,
            1e-12
        )

        return 1.0 / safe_distance

    # ==========================================================
    # EQUATION 7
    # ==========================================================

    def _calculate_smell(
        self,
        concentration
    ):
        """
        Smell_j = fitness(T_j)

        Since the paper does not specify a separate numerical
        fitness function for this FD-FFO preprocessing step,
        concentration is retained as the smell value.
        """

        return concentration.copy()

    # ==========================================================
    # EQUATION 8
    # ==========================================================

    def _select_best_fly(self, smell):
        """
        Select the fruit fly with maximum smell.
        """

        return int(
            np.argmax(smell)
        )

    # ==========================================================
    # EQUATIONS 9 AND 10
    # ==========================================================

    def _update_position(
        self,
        X_position,
        Y_position,
        best_index
    ):
        """
        Move the search position to the best fruit fly.
        """

        best_x = X_position[
            best_index
        ]

        best_y = Y_position[
            best_index
        ]

        return (
            np.full_like(
                X_position,
                best_x
            ),
            np.full_like(
                Y_position,
                best_y
            )
        )

    # ==========================================================
    # FIT
    # ==========================================================

    def fit(self, X):
        """
        Run FD-FFO preprocessing.
        """

        X = np.asarray(
            X,
            dtype=float
        )

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D matrix."
            )

        if X.shape[1] < 2:
            raise ValueError(
                "FD-FFO requires at least two features."
            )

        if self.cutoff_distance <= 0:
            raise ValueError(
                "cutoff_distance must be greater than zero."
            )

        # ------------------------------------------------------
        # Density calculation
        # ------------------------------------------------------

        distances = self._pairwise_distances(
            X
        )

        self.density_ = (
            self._calculate_density(
                distances
            )
        )

        # ------------------------------------------------------
        # Delta calculation
        # ------------------------------------------------------

        self.delta_ = (
            self._calculate_delta(
                distances,
                self.density_
            )
        )

        # ------------------------------------------------------
        # Fruit Fly Optimization
        # ------------------------------------------------------

        n_points = X.shape[1]

        X_position, Y_position, rng = (
            self._initialize_fruit_flies(
                n_points
            )
        )

        best_index = None
        best_smell = -np.inf

        for _ in range(
            self.max_iterations
        ):

            distance = (
                self._calculate_distance_from_origin(
                    X_position,
                    Y_position
                )
            )

            concentration = (
                self._calculate_concentration(
                    distance
                )
            )

            smell = (
                self._calculate_smell(
                    concentration
                )
            )

            current_best_index = (
                self._select_best_fly(
                    smell
                )
            )

            current_best_smell = (
                smell[
                    current_best_index
                ]
            )

            if current_best_smell > best_smell:

                best_smell = (
                    current_best_smell
                )

                best_index = (
                    current_best_index
                )

            # Move toward the current best fly.
            X_position, Y_position = (
                self._update_position(
                    X_position,
                    Y_position,
                    current_best_index
                )
            )

            # Small random perturbation for the next search.
            X_position += rng.uniform(
                -0.1,
                0.1,
                n_points
            )

            Y_position += rng.uniform(
                -0.1,
                0.1,
                n_points
            )

        # Store final smell information.
        self.smell_ = smell.copy()
        self.best_index_ = best_index

        # ------------------------------------------------------
        # IMPORTANT:
        #
        # The paper states:
        #
        #   fuzzy density score < 0.4 -> eliminated
        #
        # but does not give the mathematical formula that
        # converts rho/delta/FFO output into that score.
        #
        # Therefore we DO NOT invent a formula here.
        #
        # Until that missing mapping is established, retain
        # all features rather than falsely claiming a
        # paper-derived feature score.
        # ------------------------------------------------------

        self.feature_scores_ = None

        self.selected_features_ = np.ones(
            X.shape[1],
            dtype=bool
        )

        return self

    # ==========================================================
    # TRANSFORM
    # ==========================================================

    def transform(self, X):
        """
        Apply the selected feature mask.
        """

        if self.selected_features_ is None:
            raise RuntimeError(
                "FDFFO must be fitted before transform."
            )

        X = np.asarray(
            X,
            dtype=float
        )

        return X[
            :,
            self.selected_features_
        ]

    # ==========================================================
    # FIT TRANSFORM
    # ==========================================================

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)

