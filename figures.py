import numpy as np
from model import BoundedConfidence, LogOdds
from visualize import visualize


def create_all_figures(blackwhite=False):
    start = [20, 40, 60, 64, 64, 64, 64, 64]
    model = BoundedConfidence(start_distribution=start, confidence_threshold=22)
    visualize(
        model=model,
        steps=6,
        digits=0,
        output_file="images/openness_monotonicity_0",
        blackwhite=blackwhite,
    )

    model = BoundedConfidence(start_distribution=start, confidence_threshold=24)
    visualize(
        model=model,
        steps=6,
        digits=0,
        output_file="images/openness_monotonicity_1",
        blackwhite=blackwhite,
    )

    start = [30, 40, 80]
    model = BoundedConfidence(start_distribution=start, confidence_threshold=40)
    visualize(
        model=model,
        steps=6,
        digits=0,
        output_file="images/openness_stability_0",
        blackwhite=blackwhite,
    )
    model = BoundedConfidence(start_distribution=start, confidence_threshold=50)
    visualize(
        model=model,
        steps=6,
        digits=0,
        output_file="images/openness_stability_1",
        blackwhite=blackwhite,
    )

    start = [31, 41, 82]
    model = BoundedConfidence(start_distribution=start, confidence_threshold=50)
    visualize(
        model=model,
        steps=6,
        digits=0,
        output_file="images/responsiveness",
        blackwhite=blackwhite,
    )

    start = [30, 40, 81]
    model = BoundedConfidence(start_distribution=start, confidence_threshold=50)
    visualize(
        model=model,
        steps=6,
        digits=0,
        output_file="images/manipulation",
        blackwhite=blackwhite,
    )

    start = [20, 30, 43, 50, 75]
    model = BoundedConfidence(start_distribution=start, confidence_threshold=25)
    visualize(
        model=model,
        steps=6,
        digits=0,
        output_file="images/no_show_0",
        blackwhite=blackwhite,
    )

    start = [20, 30, 50, 75]
    model = BoundedConfidence(start_distribution=start, confidence_threshold=25)
    visualize(
        model=model,
        steps=6,
        digits=0,
        output_file="images/no_show_1",
        blackwhite=blackwhite,
    )

    # start = [0.2, 0.5, 0.9]
    # matrix = np.array([[0.5, 0.5, 0], [0.25, 0.75, 0], [1 / 3, 1 / 3, 1 / 3]])
    # model = LogOdds(start_profile=start, matrix=matrix)
    # visualize(
    #     model=model,
    #     steps=6,
    #     output_file="images/logodds_example",
    #     blackwhite=blackwhite,
    # )



if __name__ == "__main__":
    create_all_figures(blackwhite=False)
