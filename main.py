from model import BoundedConfidence
from visualize import visualize

if __name__ == "__main__":
    start = [0.1, 0.2, 0.3, 0.35]
    model = BoundedConfidence(start, confidence_threshold=0.1)
    visualize(model, 5)
