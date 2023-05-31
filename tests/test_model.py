import numpy as np
from model import HegselmannKrause


def test_HegselmannKrause():
    model1 = HegselmannKrause(
        start_distribution=[0.2, 0.4, 0.6], confidence_threshold=0.5
    )
    model1.update()
    correct_opinions = np.array([0.4, 0.4, 0.4], dtype=float)
    margin_of_error = 0.00001
    for agent in model1.agents:
        assert abs(model1.opinions[agent] - correct_opinions[agent]) < margin_of_error
