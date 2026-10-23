import os
import sys
import time
import numpy as np
from numba import njit, prange, set_num_threads

STUDENT_ID = 230103251
N = 2_000_000
REPEATS = 5

LOGICAL = os.cpu_count()
try:
    import psutil
    THREADS = psutil.cpu_count(logical=False) or LOGICAL
except ImportError:
    THREADS = LOGICAL


@njit
def serial_scan(a):
    out = np.empty_like(a)
    s = 0
    for i in range(a.shape[0]):
        s += a[i]
        out[i] = s
    return out


@njit(parallel=True)
def parallel_scan(a, nchunks):
    n = a.shape[0]
    out = np.empty_like(a)
    sums = np.zeros(nchunks, dtype=np.int64)

    for c in prange(nchunks):
        lo = c * n // nchunks
        hi = (c + 1) * n // nchunks
        s = 0
        for i in range(lo, hi):
            s += a[i]
            out[i] = s
        sums[c] = s

    offsets = np.zeros(nchunks, dtype=np.int64)
    for c in range(1, nchunks):
        offsets[c] = offsets[c - 1] + sums[c - 1]

    for c in prange(nchunks):
        lo = c * n // nchunks
        hi = (c + 1) * n // nchunks
        off = offsets[c]
        for i in range(lo, hi):
            out[i] += off

    return out


def timed(fn, *args):
    best = float("inf")
    result = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        result = fn(*args)
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best, result


def main():
    set_num_threads(THREADS)

    warm = np.ones(1000, dtype=np.int64)
    serial_scan(warm)
    parallel_scan(warm, THREADS)

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.integers(1, 11, size=N, dtype=np.int64)

    t_s, out_s = timed(serial_scan, data)
    t_p, out_p = timed(parallel_scan, data, THREADS)
    speedup = t_s / t_p

    try:
        assert out_p[-1] == out_s[-1]
        assert np.array_equal(out_p, out_s)
        status = "PASS"
    except AssertionError:
        status = "FAIL"

    lines = [
        f"Student ID:        {STUDENT_ID}",
        f"Array size:        {N}",
        f"Threads/chunks:    {THREADS}",
        f"Last element:      serial={out_s[-1]}  parallel={out_p[-1]}",
        f"Serial Time (ms):  {t_s:.3f}",
        f"Parallel Time (ms): {t_p:.3f}",
        f"Speedup:           {speedup:.2f}x",
        f"Verification:      {status}",
    ]
    out = "\n".join(lines)
    print(out)
    with open("task3_output.txt", "w") as f:
        f.write(out + "\n")

    if status != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()