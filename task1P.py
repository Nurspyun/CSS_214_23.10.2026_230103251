import os
import time
import numpy as np
from numba import njit, prange, set_num_threads

STUDENT_ID = 230103251
N = 8000
REPEATS = 5

LOGICAL = os.cpu_count()
try:
    import psutil
    PHYSICAL = psutil.cpu_count(logical=False) or LOGICAL
except ImportError:
    PHYSICAL = LOGICAL


@njit(parallel=True)
def sum_row_major(a, nw):
    n_rows, n_cols = a.shape
    partial = np.zeros(nw)
    for w in prange(nw):
        r0 = w * n_rows // nw
        r1 = (w + 1) * n_rows // nw
        s = 0.0
        for i in range(r0, r1):
            for j in range(n_cols):
                s += a[i, j]
        partial[w] = s
    return partial.sum()


@njit(parallel=True)
def sum_col_major(a, nw):
    n_rows, n_cols = a.shape
    partial = np.zeros(nw)
    for w in prange(nw):
        c0 = w * n_cols // nw
        c1 = (w + 1) * n_cols // nw
        s = 0.0
        for j in range(c0, c1):
            for i in range(n_rows):
                s += a[i, j]
        partial[w] = s
    return partial.sum()


def timed(fn, a, nw):
    best = float("inf")
    result = 0.0
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        result = fn(a, nw)
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best, result


def main():
    set_num_threads(PHYSICAL)

    warm = np.ones((64, 64))
    sum_row_major(warm, PHYSICAL)
    sum_col_major(warm, PHYSICAL)

    rng = np.random.default_rng(STUDENT_ID)
    a = rng.random((N, N), dtype=np.float64)

    t_a, s_a = timed(sum_row_major, a, PHYSICAL)
    t_b, s_b = timed(sum_col_major, a, PHYSICAL)
    penalty = t_b / t_a

    lines = [
        f"Student ID:            {STUDENT_ID}",
        f"Logical cores:         {LOGICAL}",
        f"Physical cores used:   {PHYSICAL}",
        f"Matrix:                {N} x {N} float64 ({a.nbytes / 1e6:.0f} MB)",
        f"Mode A (row-major):    {t_a:.2f} ms   sum={s_a:.6f}",
        f"Mode B (col-major):    {t_b:.2f} ms   sum={s_b:.6f}",
        f"Cache penalty factor:  {penalty:.2f}x  (Mode B / Mode A)",
        f"Sums match:            {np.isclose(s_a, s_b, rtol=1e-9)}",
    ]
    out = "\n".join(lines)
    print(out)
    with open("task1_output.txt", "w") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()