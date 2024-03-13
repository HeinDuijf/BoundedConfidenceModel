import numpy as np
from model import HegselmannKrause


def test_HegselmannKrause():
    margin_of_error = 0.00001
    model1 = HegselmannKrause(
        start_distribution=[0.2, 0.4, 0.6], confidence_threshold=0.5
    )
    model1.update()
    correct_opinions = np.array([0.4, 0.4, 0.4], dtype=float)
    for agent in model1.agents:
        assert abs(model1.opinions[agent] - correct_opinions[agent]) < margin_of_error

    model2 = HegselmannKrause(
        start_distribution=[0.2, 0.3, 0.5, 0.7], confidence_threshold=0.25
    )
    model2.update(number_of_steps=10)
    model_10 = model2
    model2 = HegselmannKrause(
        start_distribution=[0.2, 0.3, 0.5, 0.7], confidence_threshold=0.25
    )
    model_10_steps = model2
    for _ in range(10):
        model_10_steps.update()
    assert all(
        np.abs(model_10.opinions.copy() - model_10_steps.opinions.copy())
        < margin_of_error
    )

    model = HegselmannKrause(
        start_distribution=[0.2, 0.3, 0.5, 0.7, 0.95], confidence_threshold=0.25
    )
    result_update = model.update(number_of_steps=2, return_opinions=True)
    model = HegselmannKrause(
        start_distribution=[0.2, 0.3, 0.5, 0.7, 0.95], confidence_threshold=0.25
    )
    run = model.run(number_of_steps=2)
    result_run = run.loc[run.index[-1]]
    assert all(np.abs(result_update - result_run) < margin_of_error)
