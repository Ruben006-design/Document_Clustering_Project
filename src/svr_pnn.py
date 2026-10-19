import numpy as np


class SVRPNN:

    def __init__(self, gamma=1.0, sigma=1.0):
        self.gamma = gamma
        self.sigma = sigma

        self.alpha_ = None
        self.bias_ = 0.0

        self.X_train_ = None
        self.y_train_ = None

        self.classes_ = None
        self.class_to_index_ = None

    # ==========================================================
    # Eq. (41) - Gaussian kernel
    # ==========================================================

    def rbf_kernel(self, X1, X2):

        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)

        squared_distance = (
            np.sum(X1 ** 2, axis=1, keepdims=True)
            + np.sum(X2 ** 2, axis=1)
            - 2 * X1 @ X2.T
        )

        squared_distance = np.maximum(
            squared_distance,
            0.0
        )

        return np.exp(
            -squared_distance / (self.sigma ** 2)
        )

    # ==========================================================
    # SVR training
    #
    # Based on Eqs. (33)-(40)
    #
    # Eq. (33):
    #
    # Min 1/2 z'z + gamma/2 sum(f_j^2)
    #
    # Eq. (37):
    #
    # sum(alpha_j) = 0
    #
    # Eq. (38):
    #
    # alpha_j = gamma f_j
    #
    # Eq. (40):
    #
    # g(y) = sum(alpha_j L(y,y_j)) + a
    # ==========================================================

    def fit(self, X, y):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D matrix."
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        if self.sigma <= 0:
            raise ValueError(
                "sigma must be greater than zero."
            )

        if self.gamma <= 0:
            raise ValueError(
                "gamma must be greater than zero."
            )

        self.X_train_ = X
        self.y_train_ = y

        # Classes
        self.classes_ = np.unique(y)

        self.class_to_index_ = {
            label: index
            for index, label in enumerate(self.classes_)
        }

        # ------------------------------------------------------
        # Convert class labels into numerical target values.
        #
        # The paper does not specify a multiclass SVR encoding.
        # Therefore we use class indices only for the SVR mapping.
        # PNN remains responsible for class membership.
        # ------------------------------------------------------

        y_numeric = np.array(
            [
                self.class_to_index_[label]
                for label in y
            ],
            dtype=float
        )

        # ------------------------------------------------------
        # Kernel matrix
        # ------------------------------------------------------

        K = self.rbf_kernel(
            X,
            X
        )

        n_samples = len(X)

        # ------------------------------------------------------
        # KKT system derived from Eqs. (37)-(39)
        #
        # [ K + I/gamma    1 ] [alpha] = [y]
        # [     1^T        0 ] [  a  ]   [0]
        #
        # The second row enforces:
        #
        # sum(alpha_j) = 0
        # ------------------------------------------------------

        regularization_matrix = (
            K
            + (1.0 / self.gamma)
            * np.eye(n_samples)
        )

        kkt_matrix = np.zeros(
            (n_samples + 1, n_samples + 1),
            dtype=float
        )

        kkt_matrix[:n_samples, :n_samples] = (
            regularization_matrix
        )

        kkt_matrix[:n_samples, n_samples] = 1.0
        kkt_matrix[n_samples, :n_samples] = 1.0

        target_vector = np.zeros(
            n_samples + 1,
            dtype=float
        )

        target_vector[:n_samples] = y_numeric

        solution = np.linalg.solve(
            kkt_matrix,
            target_vector
        )

        self.alpha_ = solution[:n_samples]
        self.bias_ = solution[n_samples]

        return self

    # ==========================================================
    # Eq. (40) - SVR regression mapping
    # ==========================================================

    def decision_function(self, X):

        if self.alpha_ is None:
            raise RuntimeError(
                "SVRPNN must be fitted before prediction."
            )

        X = np.asarray(X, dtype=float)

        K = self.rbf_kernel(
            X,
            self.X_train_
        )

        return (
            K @ self.alpha_
            + self.bias_
        )

    # ==========================================================
    # Eq. (42) - PNN Gaussian activation
    #
    # f(w_j) =
    # exp(
    #   -(z_j-y)'(z_j-y) / (2 sigma^2)
    # )
    #
    # Here the SVR output is used as the mapped representation.
    # ==========================================================

    def pnn_activation(self, X):

        if self.alpha_ is None:
            raise RuntimeError(
                "SVRPNN must be fitted before prediction."
            )

        X = np.asarray(X, dtype=float)

        # SVR maps feature space to the representation
        # used by the PNN.
        mapped_X = self.decision_function(X)

        mapped_train = self.decision_function(
            self.X_train_
        )

        difference = (
            mapped_X[:, np.newaxis]
            - mapped_train[np.newaxis, :]
        )

        activations = np.exp(
            -(difference ** 2)
            / (2 * self.sigma ** 2)
        )

        return activations

    # ==========================================================
    # Eq. (43) - PNN class probability
    # ==========================================================

    def predict_proba(self, X):

        X = np.asarray(X, dtype=float)

        activations = self.pnn_activation(X)

        probabilities = np.zeros(
            (
                len(X),
                len(self.classes_)
            ),
            dtype=float
        )

        for class_index, class_label in enumerate(
            self.classes_
        ):

            class_mask = (
                self.y_train_ == class_label
            )

            probabilities[:, class_index] = (
                activations[:, class_mask]
                .sum(axis=1)
            )

        row_sums = probabilities.sum(
            axis=1,
            keepdims=True
        )

        row_sums[row_sums == 0] = 1.0

        probabilities = (
            probabilities
            / row_sums
        )

        return probabilities

    # ==========================================================
    # Final cluster assignment
    # ==========================================================

    def predict(self, X):

        probabilities = self.predict_proba(
            X
        )
        indices = np.argmax(
            probabilities,
            axis=1
        )

        return self.classes_[indices]