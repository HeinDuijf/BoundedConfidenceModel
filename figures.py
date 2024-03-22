from model import BoundedConfidence
from visualize import visualize

if __name__ == "__main__":
    start = [10, 20, 30, 32, 32, 32, 32, 32]
    model = BoundedConfidence(start_distribution=start, confidence_threshold=11)
    visualize(
        model=model,
        steps=6,
        digits=0,
        output_file="images/openness_monotonicity_0",
    )

    model = BoundedConfidence(start_distribution=start, confidence_threshold=12)
    visualize(
        model=model,
        steps=6,
        digits=0,
        output_file="images/openness_monotonicity_1",
    )

    start = [30, 40, 80]
    model = BoundedConfidence(start_distribution=start, confidence_threshold=40)
    visualize(
        model=model,
        steps=6,
        digits=0,
        output_file="images/openness_stability_0",
    )
    model = BoundedConfidence(start_distribution=start, confidence_threshold=50)
    visualize(
        model=model,
        steps=6,
        digits=0,
        output_file="images/openness_stability_1",
    )

    start = [31, 41, 82]
    model = BoundedConfidence(start_distribution=start, confidence_threshold=50)
    visualize(
        model=model,
        steps=6,
        digits=0,
        output_file="images/responsiveness",
    )

    start = [30, 40, 81]
    model = BoundedConfidence(start_distribution=start, confidence_threshold=50)
    visualize(
        model=model,
        steps=6,
        digits=0,
        output_file="images/manipulation",
    )

    start = [20, 30, 43, 50, 75]
    model = BoundedConfidence(start_distribution=start, confidence_threshold=25)
    visualize(
        model=model,
        steps=6,
        digits=0,
        output_file="images/no_show_0",
    )

    start = [20, 30, 50, 75]
    model = BoundedConfidence(start_distribution=start, confidence_threshold=25)
    visualize(
        model=model,
        steps=6,
        digits=0,
        output_file="images/no_show_1",
    )
