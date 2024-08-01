import numpy as np
import pandas as pd


class BoundedConfidence:
    def __init__(self, start_distribution: np.array, confidence_threshold: float):
        self.confidence_threshold = confidence_threshold
        self.start_distribution = np.array(start_distribution, dtype=float)
        self.opinions = np.array(start_distribution, dtype=float)
        self.agents = range(len(self.opinions))

    def run(self, number_of_steps: int = 1):
        columns = self.agents
        results = pd.DataFrame(data=self.opinions.reshape(1, -1), columns=columns)
        for _ in range(number_of_steps):
            self.update()
            new_opinions = self.opinions.copy()
            results = pd.concat(
                [results, pd.DataFrame(new_opinions.reshape(1, -1), columns=columns)],
                ignore_index=True,
            )
        return results

    def update(self):
        post_opinions = self.opinions.copy()
        for agent in self.agents:
            neighbors = (
                np.abs(self.opinions - self.opinions[agent])
                <= self.confidence_threshold
            )
            if np.sum(neighbors) > 0:
                post_opinions[agent] = np.mean(self.opinions[neighbors])
        self.opinions = post_opinions

    def reset(self):
        self.opinions = np.array(self.start_distribution, dtype=float)


class LinearPooling:
    def __init__(self, start_profile: np.array, matrix: np.array) -> None:
        self.opinions = np.array(start_profile, dtype=float)
        self.matrix = np.array(matrix, dtype=float)
        self.agents = range(len(self.opinions))

    def run(self) -> np.array:
        run = True
        while run:
            self.update()
            if np.max(model.opinions) - np.min(model.opinions) < 10**-6:
                run = False
        return self.opinions

    def update(self) -> None:
        self.opinions = np.dot(self.matrix, self.opinions)


if __name__ == "__main__":
    matrix = np.array([[1 / 2, 1 / 2, 0], [1 / 4, 3 / 4, 0], [1 / 3, 1 / 3, 1 / 3]])
    profile = [120, 240, 90]
    model = LinearPooling(profile, matrix)
    for round in range(100):
        print(f"Round {round}: {model.opinions}")
        model.update()
        if np.max(model.opinions) - np.min(model.opinions) < 10**-6:
            break
