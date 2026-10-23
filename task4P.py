import time
import numpy as np
from multiprocessing import Process, Pipe, shared_memory

STUDENT_ID = 230103251
SIZES = [100_000, 1_000_000, 10_000_000]
REPEATS = 5


def child(conn):
    while True:
        msg = conn.recv()
        if msg is None:
            break
        if isinstance(msg, tuple):
            _, name, n = msg
            shm = shared_memory.SharedMemory(name=name)
            arr = np.ndarray((n,), dtype=np.float64, buffer=shm.buf)
            conn.send(int(arr.shape[0]))
            del arr
            shm.close()
        else:
            conn.send(int(msg.shape[0]))


def best_ms(fn):
    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def main():
    parent_conn, child_conn = Pipe()
    proc = Process(target=child, args=(child_conn,))
    proc.start()

    warm = np.ones(1000)
    parent_conn.send(warm)
    parent_conn.recv()

    rng = np.random.default_rng(STUDENT_ID)
    rows = []

    for n in SIZES:
        shm = shared_memory.SharedMemory(create=True, size=n * 8)
        arr = np.ndarray((n,), dtype=np.float64, buffer=shm.buf)
        arr[:] = rng.random(n)

        def path_a():
            parent_conn.send(arr)
            parent_conn.recv()

        def path_b():
            parent_conn.send(("shm", shm.name, n))
            parent_conn.recv()

        t_a = best_ms(path_a)
        t_b = best_ms(path_b)
        rows.append((n, t_a, t_b))

        del arr
        shm.close()
        shm.unlink()

    parent_conn.send(None)
    proc.join()

    lines = [
        f"Student ID: {STUDENT_ID}",
        f"{'N':>12} | {'Path A pipe/pickle (ms)':>24} | {'Path B shared mem (ms)':>23} | {'A / B':>10}",
    ]
    for n, t_a, t_b in rows:
        ratio = t_a / t_b if t_b > 0 else float("inf")
        lines.append(f"{n:>12,} | {t_a:>24.3f} | {t_b:>23.3f} | {ratio:>9.1f}x")
    out = "\n".join(lines)
    print(out)
    with open("task4_output.txt", "w") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()