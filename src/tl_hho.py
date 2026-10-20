import math
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class TLHHO:
    """
    Teaching-Learning-Based Harris Hawks Optimization (TL-HHO).

    Paper Reference:
    "An optimal document clustering method using hybrid optimal feature selection
     and efficient soft computing classifier", Section 4.2, Eqs. (12)-(31).

    Algorithmic Workflow:
    1. Teaching Phase: Learners learn from the current best teacher (Eqs. 12-14).
    2. Learner Phase: Learners interact with peers to improve fitness (Eqs. 15-16).
    3. HHO Exploitation / Exploration: Hawks update positions via soft/hard besiege
       and rapid dives with Lévy flight (Eqs. 17-30).
    4. Multi-objective Fitness: Minimizes Eq. (31):
       fitness = alpha * kNN_error + (1 - alpha) * (n_selected / total_features)
    """

    def __init__(
        self,
        population_size=10,
        max_iterations=10,
        alpha=0.9,
        val_size=0.2,
        n_neighbors=3,
        random_state=42
    ):
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.val_size = val_size
        self.n_neighbors = n_neighbors
        self.random_state = random_state

        self.rng = np.random.default_rng(random_state)

        self.best_position = None
        self.best_fitness = None
        self.selected_features_ = None

        self.fitness_history_ = []

    # ---------------------------------------------------------
    # Position to Binary Feature Mask
    # ---------------------------------------------------------
    def position_to_features(self, position):
        selection = position >= 0.5
        # Ensure at least one feature is selected
        if not np.any(selection):
            selection[np.argmax(position)] = True
        return selection

    # ---------------------------------------------------------
    # Multi-Objective Fitness Function (Eq. 31)
    # fitness = alpha * classification_error + beta * (selected / total)
    # ---------------------------------------------------------
    def calculate_fitness(self, position, X, y):
        selected = self.position_to_features(position)
        selected_count = np.sum(selected)
        total_features = X.shape[1]

        X_selected = X[:, selected]

        # Robust internal train/validation split
        unique_classes, counts = np.unique(y, return_counts=True)
        can_stratify = len(unique_classes) > 1 and np.min(counts) >= 2
        stratify = y if can_stratify else None

        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_selected,
                y,
                test_size=self.val_size,
                random_state=self.random_state,
                stratify=stratify
            )
        except ValueError:
            # Fallback if split fails due to sparsity
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_selected,
                y,
                test_size=self.val_size,
                random_state=self.random_state,
                stratify=None
            )

        n_neighbors = min(self.n_neighbors, max(1, len(X_tr) - 1))
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        classifier.fit(X_tr, y_tr)
        predictions = classifier.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        classification_error = 1.0 - accuracy
        feature_ratio = selected_count / total_features

        fitness = self.alpha * classification_error + self.beta * feature_ratio
        return fitness

    # ---------------------------------------------------------
    # Teaching Phase (Eqs. 12-14)
    # ---------------------------------------------------------
    def teaching_phase(self, population, fitness, X, y):
        teacher_idx = np.argmin(fitness)
        teacher = population[teacher_idx]
        mean_position = np.mean(population, axis=0)

        new_population = population.copy()
        new_fitness = fitness.copy()

        for i in range(self.population_size):
            learning_factor = self.rng.random()
            teaching_factor = self.rng.integers(1, 3)  # Round[1 + Rand(0,1)] in {1, 2}

            difference_mean = learning_factor * (teacher - teaching_factor * mean_position)
            candidate = np.clip(population[i] + difference_mean, 0.0, 1.0)
            candidate_fitness = self.calculate_fitness(candidate, X, y)

            if candidate_fitness < new_fitness[i]:
                new_population[i] = candidate
                new_fitness[i] = candidate_fitness

        return new_population, new_fitness

    # ---------------------------------------------------------
    # Learner Phase (Eqs. 15-16)
    # ---------------------------------------------------------
    def learner_phase(self, population, fitness, X, y):
        new_population = population.copy()
        new_fitness = fitness.copy()

        for i in range(self.population_size):
            # Select a random peer learner p != i
            peer_candidates = [idx for idx in range(self.population_size) if idx != i]
            p = self.rng.choice(peer_candidates)

            learning_factor = self.rng.random()

            # If learner i is better than peer p, move away from p towards i
            if fitness[i] < fitness[p]:
                candidate = population[i] + learning_factor * (population[i] - population[p])
            else:
                candidate = population[i] + learning_factor * (population[p] - population[i])

            candidate = np.clip(candidate, 0.0, 1.0)
            candidate_fitness = self.calculate_fitness(candidate, X, y)

            if candidate_fitness < new_fitness[i]:
                new_population[i] = candidate
                new_fitness[i] = candidate_fitness

        return new_population, new_fitness

    # ---------------------------------------------------------
    # Lévy Flight (Eq. 26)
    # ---------------------------------------------------------
    def levy_flight(self, dimension, beta=1.5):
        sigma = (
            (math.gamma(1 + beta) * math.sin(math.pi * beta / 2))
            / (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))
        ) ** (1 / beta)

        u = self.rng.normal(0, sigma, dimension)
        v = self.rng.normal(0, 1, dimension)
        step = u / (np.abs(v) ** (1 / beta) + 1e-12)
        return 0.01 * step

    # ---------------------------------------------------------
    # HHO Exploration Phase (Eq. 17)
    # ---------------------------------------------------------
    def hho_exploration(self, hawk, population, rabbit, lower_bound=0.0, upper_bound=1.0):
        random_hawk_idx = self.rng.integers(0, self.population_size)
        random_hawk = population[random_hawk_idx]

        r1 = self.rng.random()
        r2 = self.rng.random()
        r3 = self.rng.random()
        r4 = self.rng.random()
        prob = self.rng.random()

        if prob >= 0.5:
            candidate = random_hawk - r1 * np.abs(random_hawk - 2.0 * r2 * hawk)
        else:
            mean_pos = np.mean(population, axis=0)
            candidate = (rabbit - mean_pos) - r3 * (lower_bound + r4 * (upper_bound - lower_bound))

        return np.clip(candidate, lower_bound, upper_bound)

    # ---------------------------------------------------------
    # HHO Exploitation Phase (Eqs. 20-30)
    # ---------------------------------------------------------
    def hho_exploitation(self, hawk, rabbit, energy, lower_bound=0.0, upper_bound=1.0):
        prob = self.rng.random()
        jump_strength = 2.0 * (1.0 - self.rng.random())
        delta = rabbit - hawk

        # Soft besiege
        if prob >= 0.5 and abs(energy) >= 0.5:
            candidate = rabbit - energy * np.abs(jump_strength * rabbit - hawk)
            return np.clip(candidate, lower_bound, upper_bound)

        # Hard besiege
        if prob >= 0.5 and abs(energy) < 0.5:
            candidate = rabbit - energy * np.abs(delta)
            return np.clip(candidate, lower_bound, upper_bound)

        # Rapid dives with Lévy flight (prob < 0.5)
        # Candidate X (Eq. 24)
        X_cand = rabbit - energy * np.abs(jump_strength * rabbit - hawk)
        levy = self.levy_flight(len(hawk))
        # Candidate K (Eq. 25)
        K_cand = X_cand + self.rng.random() * levy

        X_cand = np.clip(X_cand, lower_bound, upper_bound)
        K_cand = np.clip(K_cand, lower_bound, upper_bound)
        return X_cand, K_cand

    # ---------------------------------------------------------
    # Fit Method (Feature Selection)
    # ---------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        dimension = X.shape[1]

        # Initialize population in [0, 1]^D
        population = self.rng.uniform(0.0, 1.0, size=(self.population_size, dimension))
        fitness = np.array([self.calculate_fitness(ind, X, y) for ind in population])

        best_idx = np.argmin(fitness)
        self.best_position = population[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.fitness_history_ = [self.best_fitness]

        for iteration in range(self.max_iterations):
            # 1. Teaching phase
            population, fitness = self.teaching_phase(population, fitness, X, y)

            # 2. Learner phase
            population, fitness = self.learner_phase(population, fitness, X, y)

            # 3. Rabbit tracking (best solution so far)
            rabbit_idx = np.argmin(fitness)
            rabbit = population[rabbit_idx].copy()

            # 4. HHO Phase
            for i in range(self.population_size):
                hawk = population[i].copy()
                F0 = self.rng.uniform(-1.0, 1.0)
                energy = 2.0 * F0 * (1.0 - ((iteration + 1) / self.max_iterations))

                if abs(energy) >= 1.0:
                    # Exploration
                    candidate = self.hho_exploration(hawk, population, rabbit)
                    candidate_fitness = self.calculate_fitness(candidate, X, y)
                    if candidate_fitness < fitness[i]:
                        population[i] = candidate
                        fitness[i] = candidate_fitness
                else:
                    # Exploitation
                    result = self.hho_exploitation(hawk, rabbit, energy)
                    if isinstance(result, tuple):
                        cand_x, cand_k = result
                        fit_x = self.calculate_fitness(cand_x, X, y)
                        fit_k = self.calculate_fitness(cand_k, X, y)

                        # Select candidate with minimal fitness
                        best_cand, best_cand_fit = (cand_x, fit_x) if fit_x <= fit_k else (cand_k, fit_k)
                        if best_cand_fit < fitness[i]:
                            population[i] = best_cand
                            fitness[i] = best_cand_fit
                    else:
                        cand_fit = self.calculate_fitness(result, X, y)
                        if cand_fit < fitness[i]:
                            population[i] = result
                            fitness[i] = cand_fit

            # Global best update
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = fitness[current_best_idx]
                self.best_position = population[current_best_idx].copy()

            self.fitness_history_.append(self.best_fitness)

        self.selected_features_ = self.position_to_features(self.best_position)
        return self.selected_features_

    def transform(self, X):
        if self.selected_features_ is None:
            raise RuntimeError("TLHHO must be fitted before transform.")
        X = np.asarray(X, dtype=float)
        return X[:, self.selected_features_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)