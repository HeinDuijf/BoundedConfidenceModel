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
            new_opinions = self.update(return_opinions=True)
            results = pd.concat(
                [results, pd.DataFrame(new_opinions.reshape(1, -1), columns=columns)],
                ignore_index=True,
            )
        return results

    def update(self, return_opinions: bool = False):
        post_opinions = self.opinions.copy()
        for agent in self.agents:
            neighbors = (
                np.abs(self.opinions - self.opinions[agent])
                <= self.confidence_threshold
            )
            if np.sum(neighbors) > 0:
                post_opinions[agent] = np.mean(self.opinions[neighbors])
        self.opinions = post_opinions
        if return_opinions:
            return self.opinions.copy()
