import time
import random
import threading

STUDENT_ID = 230103251
TOTAL = 40_000_000
THREADS = 4
PER_THREAD = TOTAL // THREADS


def worker_shared(n, rng, lock, hits, idx):
    rand = rng.random
    count = 0
    for _ in range(n):
        with lock:
            x = rand()
            y = rand()
        if x * x + y * y <= 1.0:
            count += 1
    hits[idx] = count


def worker_independent(n, seed, hits, idx):
    rand = random.Random(seed).random
    count = 0
    for _ in range(n):
        x = rand()
        y = rand()
        if x * x + y * y <= 1.0:
            count += 1
    hits[idx] = count


def run_shared():
    rng = random.Random(STUDENT_ID)
    lock = threading.Lock()
    hits = [0] * THREADS
    threads = [
        threading.Thread(target=worker_shared, args=(PER_THREAD, rng, lock, hits, i))
        for i in range(THREADS)
    ]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ms = (time.perf_counter() - t0) * 1000.0
    pi = 4.0 * sum(hits) / (PER_THREAD * THREADS)
    return pi, ms


def run_independent():
    hits = [0] * THREADS
    threads = [
        threading.Thread(
            target=worker_independent,
            args=(PER_THREAD, STUDENT_ID * 10 + i, hits, i),
        )
        for i in range(THREADS)
    ]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ms = (time.perf_counter() - t0) * 1000.0
    pi = 4.0 * sum(hits) / (PER_THREAD * THREADS)
    return pi, ms


def main():
    import math

    pi_s, t_s = run_shared()
    pi_i, t_i = run_independent()
    speedup = t_s / t_i

    lines = [
        f"Student ID:             {STUDENT_ID}",
        f"Samples / threads:      {PER_THREAD * THREADS} / {THREADS}",
        f"Shared PRNG:            pi={pi_s:.8f}  error={abs(pi_s - math.pi):.8f}  time={t_s:.2f} ms",
        f"Independent PRNG:       pi={pi_i:.8f}  error={abs(pi_i - math.pi):.8f}  time={t_i:.2f} ms",
        f"Speedup (shared/indep): {speedup:.2f}x",
    ]
    out = "\n".join(lines)
    print(out)
    with open("task2_output.txt", "w") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()