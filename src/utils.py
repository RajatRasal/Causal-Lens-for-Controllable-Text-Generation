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
    step = (stop - start) / (period * ratio_increase)  # linear schedule

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


if __name__ == "__main__":
    xx = beta_cycle_in_range(85174, 0, 1, 10, 0.25, 0.25)
    print(xx)
