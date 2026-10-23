import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from numba import njit

STUDENT_ID = 230103251
NUM_TASKS = 10_000
ROUNDS = 500
BUF_SIZE = 1024
REPEATS = 2

CORES = os.cpu_count()


@njit(nogil=True)
def hash_task(buf, seed):
    h = (2166136261 ^ seed) & 0xFFFFFFFF
    for r in range(ROUNDS):
        for i in range(buf.shape[0]):
            h = ((h ^ buf[i]) * 16777619) & 0xFFFFFFFF
        h = (h + r) & 0xFFFFFFFF
    return h


def run_batch(pool_size, buf):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=pool_size) as ex:
        futures = [ex.submit(hash_task, buf, i) for i in range(NUM_TASKS)]
        total = 0
        for f in futures:
            total ^= f.result()
    ms = (time.perf_counter() - t0) * 1000.0
    return ms, total


def main():
    rng = np.random.default_rng(STUDENT_ID)
    buf = rng.integers(0, 256, size=BUF_SIZE, dtype=np.int64)

    hash_task(buf, 0)

    sizes = sorted({1, max(1, CORES // 2), CORES, CORES * 2, CORES * 4, CORES * 16, 256})

    results = []
    for size in sizes:
        best = float("inf")
        for _ in range(REPEATS):
            ms, _ = run_batch(size, buf)
            best = min(best, ms)
        results.append((size, best))

    base = results[0][1]
    best_idx = min(range(len(results)), key=lambda i: results[i][1])
    best_time = results[best_idx][1]

    statuses = []
    flagged = False
    for i, (size, ms) in enumerate(results):
        if i < best_idx:
            statuses.append("SCALING")
        elif i == best_idx:
            statuses.append("BEST")
        elif ms <= best_time * 1.03:
            statuses.append("PLATEAU")
        elif not flagged:
            statuses.append("<< WORSENING STARTS")
            flagged = True
        else:
            statuses.append("DEGRADED")

    lines = [
        f"Student ID: {STUDENT_ID}",
        f"Logical cores: {CORES}   Tasks: {NUM_TASKS}   Rounds/task: {ROUNDS}   Buffer: {BUF_SIZE} bytes",
        f"{'Pool Size':>10} | {'Execution Time (ms)':>20} | {'Speedup vs 1 thread':>20} | Status",
    ]
    for (size, ms), status in zip(results, statuses):
        lines.append(f"{size:>10} | {ms:>20.2f} | {base / ms:>19.2f}x | {status}")

    out = "\n".join(lines)
    print(out)
    with open("task5_output.txt", "w") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()