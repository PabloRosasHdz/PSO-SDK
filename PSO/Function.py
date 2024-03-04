import numpy as np


class Function:
    def evaluate(positions: np.ndarray) -> float:
        x, y = positions
        if x < -5 or x > 5 or y < -5 or y > 5:
            return float("inf")
        return x**2 + y**2
