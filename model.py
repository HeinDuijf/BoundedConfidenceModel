import numpy as np
import pandas as pd


class HegselmannKrause:
    def __init__(self, start_distribution: np.array, confidence_threshold: float):
        self.confidence_threshold = confidence_threshold
        self.start_distribution = np.array(start_distribution, dtype=float)
        self.opinions = np.array(start_distribution, dtype=float)
        self.agents = range(len(self.opinions))

    def run(self, number_of_steps):
        result = pd.DataFrame(columns=self.agents)
        for step in number_of_steps:
            opinions_step = self.update_and_return()
            result.stack()
        pass

    def update(self):
        new_opinions = np.zeros(self.opinions.shape)
        for agent in self.agents:
            neighbors = (
                np.abs(self.opinions - self.opinions[agent])
                <= self.confidence_threshold
            )
            if np.sum(neighbors) > 0:
                new_opinions[agent] = np.mean(self.opinions[neighbors])
        self.opinions = new_opinions

    def update_and_return(self):
        self.update()
        return self.opinions
