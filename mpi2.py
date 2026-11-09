import os
import sys
import time
from collections import Counter

from mpi4py import MPI


def load_query(path):
    with open(path, "r", encoding="utf-8") as f:
        return sorted({line.strip().lower() for line in f if line.strip()})


def count_files(files, target):
    """Same tokenization as the baseline: split() + lower()."""
    freq = Counter()
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                for w in line.lower().split():
                    if w in target:
                        freq[w] += 1
    return freq


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

    comm.Barrier()
    t_start = MPI.Wtime()

    words, files = None, None
    if rank == 0:
        words = load_query(os.path.join(dataset, "consulta.txt"))
        names = sorted(n for n in os.listdir(dataset)
                       if n.startswith("file_") and n.endswith(".txt"))
        all_files = [os.path.join(dataset, n) for n in names]
        # LPT greedy: sort by size (bytes ~ work), give each file to the
        # process with the smallest accumulated load
        sized = sorted(((os.path.getsize(p), p) for p in all_files), reverse=True)
        load = [0] * size
        files = [[] for _ in range(size)]
        for sz, p in sized:
            r = load.index(min(load))
            files[r].append(p)
            load[r] += sz

    words = comm.bcast(words, root=0)           # step 2: broadcast query
    my_files = comm.scatter(files, root=0)      # step 4: static assignment
    target = set(words)

    t0 = time.perf_counter()
    local = count_files(my_files, target)       # step 5: local counting
    t_local = time.perf_counter() - t0

    # step 6: gather partial results (+ stats) in rank 0
    gathered = comm.gather((dict(local), len(my_files), t_local), root=0)
    comm.Barrier()
    t_total = MPI.Wtime() - t_start

    if rank == 0:
        total = Counter()
        for d, _, _ in gathered:
            total.update(d)
        times = [g[2] for g in gathered]
        mean_t = sum(times) / len(times)
        print(f"MPI version 2 | np={size}")
        for r, (_, nf, t) in enumerate(gathered):
            print(f"  rank {r}: files={nf} local_time={t:.6f} s")
        print(f"LI={max(times) / mean_t:.4f}  "
              f"spread={(max(times) - min(times)) / mean_t:.4f}")
        print(f"Total occurrences: {sum(total.values())}")
        print("Top 10 palabras de consulta en el corpus:")
        for w, c in total.most_common(10):
            print(f"  {w}: {c}")
        print(f"EXECUTION_TIME={t_total:.6f}")


if __name__ == "__main__":
    main()
