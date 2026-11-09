# Lab 2 – Parallel Word Counting in a Text Corpus with MPI

Kevin Torregrosa

## 1. Problem Description
Given `dataset/consulta.txt` (one query word per line) and a corpus of 3000 files `file_XXXX.txt`, count how many times each query word appears in the corpus (case-insensitive, whitespace tokenization) and report the top 10. File sizes in the corpus are very uneven (2k–5k, 10k–20k and 50k–100k tokens), so the work per file varies.

## 2. Environment and Execution Instructions
- Docker image `augustosalazar/slim-mpi:2` (Python 3 + mpi4py + OpenMPI) on Windows (Docker Desktop). Machine: AMD Ryzen 5 4600H, 6 cores, 16GB RAM.
- Run everything (dataset, baseline, mpi1, mpi2; 3 runs each for np = 2, 4, 8):
  ```
  docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 sh /app/run_all.sh
  ```
  Logs go to `results/logs/` and times to `results/times_summary.csv`. Tables for this README: `python analyze.py results`.

### a. Experimental plan
Same dataset for every run. Run the sequential baseline to get `T_seq`; run mpi1 and mpi2 with p = 2, 4 and 8 processes, 3 runs each. Metrics: `T_p` = mean total time, `S_p = T_seq / T_p`, `E_p = S_p / p`, and `LI = max(t_i) / mean(t_i)` where `t_i` is the local processing time of process i. The top 10 of every version must match the baseline.

### b. Sequential baseline
`baseline_secuencial.py` (unchanged) loads the query words into a set, reads every `file_*.txt` line by line, lowercases tokens, counts those in the set with a `Counter`, and prints the execution time and top 10.

### c. MPI version 1 (`mpi1.py`)
Rank 0 reads `consulta.txt` and broadcasts it (`comm.bcast`), lists and sorts the files, and assigns them statically in **round-robin** (file i → rank i mod p) with `comm.scatter`. Each rank counts its files with the same tokenization as the baseline and measures its local time. Partial counters, file counts and local times are gathered in rank 0 (`comm.gather`), which merges them and prints the per-process report, LI and top 10. Total time is measured from a start barrier to the end of the gather.

### d. Test procedure
Check that the top 10 of mpi1 and mpi2 equal the baseline's; run 3 times per version and per p; compute metrics with `analyze.py`.

## 3. Experimental Plan Execution

### a. Sequential baseline timing
`T_seq = 48.5198 s` (single run).

### b. MPI version 1 timing results
| np | runs | T_p (avg, s) | Speedup | Efficiency | LI (avg) |
|---|---|---|---|---|---|
| 2 | 3 | 16.1867 | 3.00 | 1.50 | 1.003 |
| 4 | 3 | 9.6849 | 5.01 | 1.25 | 1.025 |
| 8 | 3 | 5.8720 | 8.26 | 1.03 | 1.060 |

### c. Load imbalance evidence (MPI v1, local time per process)
| np | run | files/process | local times (s) |
|---|---|---|---|
| 2 | 1 | 1500 | 14.387, 14.336 |
| 2 | 2 | 1500 | 13.323, 13.244 |
| 2 | 3 | 1500 | 20.778, 20.662 |
| 4 | 1 | 750 | 8.726, 8.730, 9.040, 8.923 |
| 4 | 2 | 750 | 10.374, 10.120, 10.851, 10.636 |
| 4 | 3 | 750 | 8.775, 8.748, 9.032, 8.883 |
| 8 | 1 | 375 | 4.812, 4.823, 5.420, 5.040, 5.185, 5.255, 4.931, 5.273 |
| 8 | 2 | 375 | 5.319, 5.339, 5.961, 5.488, 5.710, 5.812, 5.456, 5.830 |
| 8 | 3 | 375 | 5.496, 5.508, 6.119, 5.767, 5.905, 6.025, 5.696, 6.000 |

LI (average of 3 runs) is 1.003, 1.025 and 1.060 for p = 2, 4, 8.

### d. MPI version 2 (`mpi2.py`)
**How it corrects the imbalance:** the pipeline is identical, but instead of round-robin, files are sorted by size in bytes (a proxy for work), largest first, and each file is assigned to the process with the smallest accumulated size (greedy LPT). This prevents the large 50k–100k-token files from piling up on one rank.

| np | runs | T_p (avg, s) | Speedup | Efficiency | LI (avg) |
|---|---|---|---|---|---|
| 2 | 3 | 17.6259 | 2.75 | 1.38 | 1.005 |
| 4 | 3 | 12.9700 | 3.74 | 0.94 | 1.004 |
| 8 | 3 | 10.6658 | 4.55 | 0.57 | 1.006 |

Local times (s): np=2 → 13.0–13.5; np=4 → 7.3–7.8; np=8 → 5.4–5.9, with a spread of less than 0.2 s between processes in every run (e.g. np=8 run 3: 5.435, 5.538, 5.420, 5.523, 5.534, 5.534, 5.436, 5.502). Files per process differ by at most 2.

### Comparison
| np | v1 T_p (s) | v2 T_p (s) | v1 speedup | v2 speedup | v1 efficiency | v2 efficiency | v1 LI | v2 LI |
|---|---|---|---|---|---|---|---|---|
| 2 | 16.19 | 17.63 | 3.00 | 2.75 | 1.50 | 1.38 | 1.003 | 1.005 |
| 4 | 9.68 | 12.97 | 5.01 | 3.74 | 1.25 | 0.94 | 1.025 | 1.004 |
| 8 | 5.87 | 10.67 | 8.26 | 4.55 | 1.03 | 0.57 | 1.060 | 1.006 |

## 4. Analysis

### a. Parallel performance
MPI v1 clearly improved on the sequential baseline (T_seq = 48.52 s): T_p was 16.19 s, 9.68 s and 5.87 s for p = 2, 4, 8, giving speedups of 3.00, 5.01 and 8.26. These speedups exceed the ideal S_p = p (efficiency 1.50, 1.25, 1.03), which is not physically plausible: at p = 2 the sum of local times is about 28 s, far less than T_seq. The baseline was therefore probably measured under worse conditions (a single run, cold file cache, or system load), so the true speedup is lower than reported and the improvement is real but overstated. Efficiency decreases as p grows (1.50 → 1.03), consistent with fixed costs that do not parallelize: listing files, gather/merge on rank 0, process startup, shared disk I/O, and a limited number of physical cores.

### b. Load balance
V1 was already well balanced: LI = 1.003, 1.025 and 1.060 for p = 2, 4, 8, with per-process differences below about 1 s (e.g. p = 8: 4.8–5.4 s with 375 files each). Imbalance grows with p because each rank has fewer files, so a few large files weigh more. V2 reduced LI to 1.005, 1.004 and 1.006 and its local times are very tightly grouped. However, this did **not** produce a real performance improvement: v2 was slower in total time (17.63, 12.97 and 10.67 s) with lower speedup (2.75, 3.74, 4.55). Its local times are similar to or better than v1's (e.g. p = 4: ~7.7 s vs ~9 s), but its total includes roughly 4–5 s of extra serial work on rank 0 (calling `os.path.getsize` on 3000 files and sorting them, on a Docker bind-mounted Windows volume). The balancing worked, but its cost outweighed the little imbalance that existed.

### c. Limitations
- One dataset, one machine, and only 3 runs per configuration; some runs varied a lot (e.g. v1 p = 2: 13.3 s vs 20.8 s).
- Docker on Windows adds file-system overhead and the cores may be shared with other processes.
- The baseline was run only once and looks slow, which inflates speedup and efficiency for both versions (values above 1).
- File size in bytes is only a proxy for processing work, and the size-lookup overhead in v2 is environment-specific.

These limitations weaken the absolute speedup and efficiency values considerably, but weaken the main conclusions much less, because the comparison of local times, LI and total time between v1 and v2 comes from the same environment and the same dataset.

## 5. Conclusions
For this problem and dataset I recommend the simple static round-robin distribution of v1. It was already well balanced (LI ≤ 1.06) and gave the best total time at every p (5.87 s at p = 8, speedup 8.26 against a probably inflated baseline), while v2 achieved better balance (LI ≈ 1.005) but was slower (10.67 s at p = 8, speedup 4.55, efficiency 0.57) because of the serial cost of measuring and sorting file sizes on rank 0. Size-aware balancing would only pay off if file sizes were much more skewed or if the size lookup were cheap or done in parallel.