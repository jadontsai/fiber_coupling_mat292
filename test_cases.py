import numpy as np

def quadratic(theta: np.ndarray) -> float:

    return -np.sum(theta**2)


def offset_quadratic(theta: np.ndarray) -> float:

    c = np.array([1.0, -0.5, 2.0, 0.5])[:len(theta)]
    return -np.sum((theta - c)**2)


def multimodal_cosine(theta: np.ndarray) -> float:

    return np.sum(np.cos(theta))


def synthetic_coupling(theta: np.ndarray) -> float:

    offsets = np.array([0.7, -1.1, 0.4, 2.2])[:len(theta)]
    val = (
        0.6 * np.cos(theta[0] - offsets[0])
        + 0.9 * np.cos(2.0 * (theta[1] - offsets[1]))
        + 0.7 * np.cos(theta[2] - offsets[2])
        + 1.0 * np.cos(3.0 * (theta[3] - offsets[3]))
    )
    val += 0.15 * np.cos((theta[0] - theta[2])) + 0.1 * np.cos((theta[1] + theta[3]))
    return float(val + 4.0)