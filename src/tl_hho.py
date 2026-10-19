import math
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class TLHHO:

    def __init__(
        self,
        population_size=10,
        max_iterations=10,
        alpha=0.9,
        random_state=42
    ):
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.random_state = random_state

        self.rng = np.random.default_rng(random_state)

        self.best_position = None
        self.best_fitness = None

        self.X = None
        self.y = None

    # ---------------------------------------------------------
    # Convert continuous position into binary feature selection
    # ---------------------------------------------------------
    def position_to_features(self, position):
        selection = position >= 0.5

        # At least one feature must be selected
        if not np.any(selection):
            selection[np.argmax(position)] = True

        return selection

    # ---------------------------------------------------------
    # Fitness function - Eq. (31)
    # Fitness = alpha * classification error
    #         + beta * selected feature ratio
    # ---------------------------------------------------------
    def calculate_fitness(self, position, X, y):

        selected = self.position_to_features(position)

        selected_count = np.sum(selected)
        total_features = X.shape[1]

        X_selected = X[:, selected]

        stratify = y if len(np.unique(y)) > 1 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X_selected,
            y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=stratify
        )

        n_neighbors = min(3, len(X_train))

        classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors
        )

        classifier.fit(X_train, y_train)

        predictions = classifier.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        classification_error = 1.0 - accuracy

        feature_ratio = (
            selected_count / total_features
        )

        fitness = (
            self.alpha * classification_error
            + self.beta * feature_ratio
        )

        return fitness

    # ---------------------------------------------------------
    # Teaching phase - Eqs. (12)-(14)
    # ---------------------------------------------------------
    def teaching_phase(self, population, fitness):

        teacher_index = np.argmin(fitness)

        teacher = population[teacher_index]

        mean_position = np.mean(
            population,
            axis=0
        )

        new_population = population.copy()

        for i in range(self.population_size):

            learning_factor = self.rng.random()

            teaching_factor = self.rng.integers(
                1,
                3
            )

            difference_mean = (
                learning_factor
                * (
                    teacher
                    - teaching_factor * mean_position
                )
            )

            candidate = (
                population[i]
                + difference_mean
            )

            candidate = np.clip(
                candidate,
                0.0,
                1.0
            )

            candidate_fitness = self.calculate_fitness(
                candidate,
                self.X,
                self.y
            )

            if candidate_fitness < fitness[i]:

                new_population[i] = candidate
                fitness[i] = candidate_fitness

        return new_population, fitness

    # ---------------------------------------------------------
    # Learner phase - Eqs. (15)-(16)
    # ---------------------------------------------------------
    def learner_phase(self, population, fitness):

        new_population = population.copy()

        for i in range(self.population_size):

            indices = list(range(self.population_size))
            indices.remove(i)

            p, q = self.rng.choice(
                indices,
                size=2,
                replace=False
            )

            learner_p = population[p]
            learner_q = population[q]

            learning_factor = self.rng.random()

            if fitness[q] < fitness[p]:

                candidate = (
                    learner_q
                    + learning_factor
                    * (learner_q - learner_p)
                )

            else:

                candidate = (
                    learner_q
                    + learning_factor
                    * (learner_p - learner_q)
                )

            candidate = np.clip(
                candidate,
                0.0,
                1.0
            )

            candidate_fitness = self.calculate_fitness(
                candidate,
                self.X,
                self.y
            )

            if candidate_fitness < fitness[q]:

                new_population[q] = candidate
                fitness[q] = candidate_fitness

        return new_population, fitness

    # ---------------------------------------------------------
    # Lévy flight used in progressive rapid dives
    # ---------------------------------------------------------
    def levy_flight(self, dimension, beta=1.5):

        sigma = (
            (
                math.gamma(1 + beta)
                * math.sin(math.pi * beta / 2)
            )
            /
            (
                math.gamma((1 + beta) / 2)
                * beta
                * 2 ** ((beta - 1) / 2)
            )
        ) ** (1 / beta)

        u = self.rng.normal(
            0,
            sigma,
            dimension
        )

        v = self.rng.normal(
            0,
            1,
            dimension
        )

        step = (
            u
            / (np.abs(v) ** (1 / beta))
        )

        return 0.01 * step

    # ---------------------------------------------------------
    # HHO exploration phase - Eq. (17)
    # ---------------------------------------------------------
    def hho_exploration(
        self,
        hawk,
        population,
        rabbit,
        lower_bound=0.0,
        upper_bound=1.0
    ):

        random_hawk_index = self.rng.integers(
            0,
            self.population_size
        )

        random_hawk = population[
            random_hawk_index
        ]

        r1 = self.rng.random()
        r2 = self.rng.random()
        r3 = self.rng.random()
        r4 = self.rng.random()

        probability = self.rng.random()

        if probability >= 0.5:

            candidate = (
                random_hawk
                - r1
                * np.abs(
                    random_hawk
                    - 2.0 * r2 * hawk
                )
            )

        else:

            mean_position = np.mean(
                population,
                axis=0
            )

            candidate = (
                rabbit
                - mean_position
                - r3
                * (
                    lower_bound
                    + r4
                    * (
                        upper_bound
                        - lower_bound
                    )
                )
            )

        return np.clip(
            candidate,
            lower_bound,
            upper_bound
        )

    # ---------------------------------------------------------
    # HHO exploitation phase
    # Eqs. (20)-(30)
    # ---------------------------------------------------------
    def hho_exploitation(
        self,
        hawk,
        rabbit,
        energy,
        lower_bound=0.0,
        upper_bound=1.0
    ):

        probability = self.rng.random()

        jump_strength = (
            2.0
            * (
                1.0
                - self.rng.random()
            )
        )

        delta = rabbit - hawk

        # -----------------------------------------------------
        # Soft besiege
        # -----------------------------------------------------
        if probability >= 0.5 and abs(energy) >= 0.5:

            candidate = (
                rabbit
                - energy
                * np.abs(
                    jump_strength * rabbit
                    - hawk
                )
            )

            return np.clip(
                candidate,
                lower_bound,
                upper_bound
            )

        # -----------------------------------------------------
        # Hard besiege
        # -----------------------------------------------------
        if probability >= 0.5 and abs(energy) < 0.5:

            candidate = (
                rabbit
                - energy
                * np.abs(delta)
            )

            return np.clip(
                candidate,
                lower_bound,
                upper_bound
            )

        # -----------------------------------------------------
        # Progressive rapid dives
        # -----------------------------------------------------
        if probability < 0.5:

            # Candidate X - Eq. (24)/(29)
            X = (
                rabbit
                - energy
                * np.abs(
                    jump_strength * rabbit
                    - hawk
                )
            )

            # Lévy flight
            levy = self.levy_flight(
                len(hawk)
            )

            # Candidate K - Eq. (25)/(30)
            K = X + self.rng.random() * levy

            X = np.clip(
                X,
                lower_bound,
                upper_bound
            )

            K = np.clip(
                K,
                lower_bound,
                upper_bound
            )

            # The actual fitness comparison is performed
            # outside this method.
            return X, K

        return np.clip(
            rabbit - energy * np.abs(delta),
            lower_bound,
            upper_bound
        )

    # ---------------------------------------------------------
    # Main TL-HHO optimization
    # ---------------------------------------------------------
    def fit(self, X, y):

        self.X = np.asarray(
            X,
            dtype=float
        )

        self.y = np.asarray(y)

        if self.X.ndim != 2:
            raise ValueError(
                "X must be a 2D matrix."
            )

        if len(self.X) != len(self.y):
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        dimension = self.X.shape[1]

        # -----------------------------------------------------
        # Initialize hawk population
        # -----------------------------------------------------
        population = self.rng.random(
            (
                self.population_size,
                dimension
            )
        )

        fitness = np.array([
            self.calculate_fitness(
                position,
                self.X,
                self.y
            )
            for position in population
        ])

        best_index = np.argmin(fitness)

        self.best_position = (
            population[best_index].copy()
        )

        self.best_fitness = fitness[
            best_index
        ]

        print("\nStarting TL-HHO...")
        print(
            "Population size:",
            self.population_size
        )
        print(
            "Iterations:",
            self.max_iterations
        )

        # -----------------------------------------------------
        # Main TL-HHO loop
        # -----------------------------------------------------
        for iteration in range(
            self.max_iterations
        ):

            # -------------------------------------------------
            # Teaching phase
            # -------------------------------------------------
            population, fitness = (
                self.teaching_phase(
                    population,
                    fitness
                )
            )

            # -------------------------------------------------
            # Learner phase
            # -------------------------------------------------
            population, fitness = (
                self.learner_phase(
                    population,
                    fitness
                )
            )

            # -------------------------------------------------
            # Find current rabbit
            # -------------------------------------------------
            rabbit_index = np.argmin(
                fitness
            )

            rabbit = (
                population[
                    rabbit_index
                ].copy()
            )

            rabbit_fitness = fitness[
                rabbit_index
            ]

            # -------------------------------------------------
            # HHO update for every hawk
            # -------------------------------------------------
            for i in range(
                self.population_size
            ):

                hawk = population[i].copy()

                # Initial energy F0 in [-1, 1]
                F0 = self.rng.uniform(
                    -1.0,
                    1.0
                )

                # Eq. (19)
                energy = (
                    2.0
                    * F0
                    * (
                        1.0
                        - (
                            (iteration + 1)
                            / self.max_iterations
                        )
                    )
                )

                # ---------------------------------------------
                # Exploration
                # ---------------------------------------------
                if abs(energy) >= 1.0:

                    candidate = (
                        self.hho_exploration(
                            hawk,
                            population,
                            rabbit
                        )
                    )

                    candidate_fitness = (
                        self.calculate_fitness(
                            candidate,
                            self.X,
                            self.y
                        )
                    )

                    if (
                        candidate_fitness
                        < fitness[i]
                    ):

                        population[i] = candidate
                        fitness[i] = (
                            candidate_fitness
                        )

                # ---------------------------------------------
                # Exploitation
                # ---------------------------------------------
                else:

                    result = (
                        self.hho_exploitation(
                            hawk,
                            rabbit,
                            energy
                        )
                    )

                    # Progressive rapid dive
                    if isinstance(
                        result,
                        tuple
                    ):

                        candidate_x, candidate_k = result

                        fitness_x = (
                            self.calculate_fitness(
                                candidate_x,
                                self.X,
                                self.y
                            )
                        )

                        fitness_k = (
                            self.calculate_fitness(
                                candidate_k,
                                self.X,
                                self.y
                            )
                        )

                        current_fitness = fitness[i]

                        if fitness_x < current_fitness:

                            population[i] = candidate_x
                            fitness[i] = fitness_x

                        elif fitness_k < current_fitness:

                            population[i] = candidate_k
                            fitness[i] = fitness_k

                    else:

                        candidate = result

                        candidate_fitness = (
                            self.calculate_fitness(
                                candidate,
                                self.X,
                                self.y
                            )
                        )

                        if (
                            candidate_fitness
                            < fitness[i]
                        ):

                            population[i] = candidate
                            fitness[i] = (
                                candidate_fitness
                            )

            # -------------------------------------------------
            # Update rabbit / global best
            # -------------------------------------------------
            rabbit_index = np.argmin(
                fitness
            )

            if (
                fitness[rabbit_index]
                < self.best_fitness
            ):

                self.best_position = (
                    population[
                        rabbit_index
                    ].copy()
                )

                self.best_fitness = fitness[
                    rabbit_index
                ]

            print(
                f"Iteration {iteration + 1}/"
                f"{self.max_iterations} - "
                f"Best fitness: "
                f"{self.best_fitness:.6f}"
            )

        return self.position_to_features(
            self.best_position
        )