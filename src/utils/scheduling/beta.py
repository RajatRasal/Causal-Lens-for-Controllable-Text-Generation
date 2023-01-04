import numpy as np


def beta_cycle_in_range(
    n_iter: int,
    start: float = 0.0,
    stop: float = 1.0,
    n_cycle: int = 4,
    ratio_increase: float = 0.5,
    ratio_zero: float = 0.3,
) -> np.ndarray:
    L = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    # linear schedule
    step = (stop - start) / (period * ratio_increase)

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            if i < period * ratio_zero:
                L[int(i + c * period)] = start
            else:
                L[int(i + c * period)] = v
                v += step
            i += 1

    return L
