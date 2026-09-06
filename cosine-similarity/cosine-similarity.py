import numpy as np


def cosine_similarity(a: list, b: list) -> float:
    """Returns the cosine similarity as a Python float."""
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)

    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(va, vb) / (norm_a * norm_b))