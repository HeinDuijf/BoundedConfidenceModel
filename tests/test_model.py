import numpy as np
import pytest
from scipy.special import expit, logit

from model import BoundedConfidence, LogOdds

global margin_of_error
margin_of_error = 0.00001


def test_initialize():
    model1 = BoundedConfidence(
        start_distribution=[0.2, 0.4, 0.6], confidence_threshold=0.5
    )
    assert model1.agents == range(3)
    assert set(model1.start_distribution) == {0.2, 0.4, 0.6}
    assert set(model1.opinions) == {0.2, 0.4, 0.6}


def test_update():
    model1 = BoundedConfidence(
        start_distribution=[0.2, 0.4, 0.6], confidence_threshold=0.5
    )
    model1.update()
    correct_opinions = np.array([0.4, 0.4, 0.4], dtype=float)
    assert all(
        abs(model1.opinions[agent] - correct_opinions[agent]) < margin_of_error
        for agent in model1.agents
    )
    model2 = BoundedConfidence(
        start_distribution=[0.2, 0.3, 0.5, 0.7], confidence_threshold=0.25
    )
    model2.update()
    correct_opinions = np.array([0.25, 1 / 3, 0.5, 0.6])
    assert all(
        abs(model2.opinions[agent] - correct_opinions[agent]) < margin_of_error
        for agent in model2.agents
    )


def test_run():
    model1 = BoundedConfidence(
        start_distribution=[0.2, 0.4, 0.6], confidence_threshold=0.5
    )
    steps = 5
    results1 = model1.run(number_of_steps=steps)
    assert len(results1.index) == steps + 1
    assert len(results1.columns) == len(model1.agents)

    correct_opinions = np.array([0.4, 0.4, 0.4], dtype=float)
    assert all(
        abs(model1.opinions[agent] - correct_opinions[agent]) < margin_of_error
        for agent in model1.agents
    )

    model2 = BoundedConfidence(
        start_distribution=[0.2, 0.3, 0.5, 0.7], confidence_threshold=0.25
    )
    steps = 3
    results2 = model2.run(number_of_steps=steps)
    assert len(results2.index) == steps + 1
    assert len(results2.columns) == len(model2.agents)

    correct_opinions = np.full(len(model2.agents), [0.42326388888])
    assert all(
        abs(model2.opinions[agent] - correct_opinions[agent]) < margin_of_error
        for agent in model2.agents
    )


def test_reset():
    model2 = BoundedConfidence(
        start_distribution=[0.2, 0.3, 0.5, 0.7], confidence_threshold=0.25
    )
    steps = 3
    model2.run(number_of_steps=steps)
    assert set(model2.opinions) != set(model2.start_distribution)
    model2.reset()
    assert set(model2.opinions) == set(model2.start_distribution)


def test_logodds_matches_closed_form():
    matrix = np.array([[0.5, 0.5, 0], [0.25, 0.75, 0], [1 / 3, 1 / 3, 1 / 3]])
    start = [0.2, 0.5, 0.8]
    steps = 4
    model = LogOdds(start, matrix)
    results = model.run(number_of_steps=steps)

    predicted_log_odds = np.linalg.matrix_power(matrix, steps) @ logit(np.array(start))
    predicted = expit(predicted_log_odds)
    assert np.allclose(results.iloc[-1].to_numpy(), predicted, atol=1e-8)


def test_logodds_requires_row_stochastic_matrix():
    bad_matrix = np.array([[0.5, 0.6], [0.3, 0.7]])
    with pytest.raises(ValueError):
        LogOdds(start_profile=[0.3, 0.6], matrix=bad_matrix)
