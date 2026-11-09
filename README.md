# Lab 2 – Parallel Word Counting in a Text Corpus with MPI

**Team:** <<NAMES / IDs>> · **Course:** <<COURSE>> · Universidad del Norte

## 1. Problem Description
Given `dataset/consulta.txt` (one query word per line) and a corpus of 3000 files `file_XXXX.txt`, count how many times each query word appears in the corpus (case-insensitive, whitespace tokenization) and report the top 10. The corpus has files of very different sizes (2k–5k, 10k–20k and 50k–100k tokens), which makes the work per file uneven.

## 2. Environment and Execution Instructions
- Docker image `augustosalazar/slim-mpi:2` (Python 3 + mpi4py + OpenMPI). Machine: <<CPU model, physical cores, RAM, OS>>.
- Run everything (dataset, baseline, mpi1, mpi2; 3 runs each for np = 2, 4, 8):
  ```
  docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 sh /app/run_all.sh
  ```
  Customize with `NP_LIST="4" RUNS=5`. Logs go to `results/logs/`, times to `results/times_summary.csv`.
- Tables for this README: `python analyze.py results`.

### a. Experimental plan
Use the same dataset for all runs. Run the sequential baseline once to get `T_seq`. Run mpi1 and mpi2 with p = <<p from Detectar_procesos_mpi>>, at least 3 runs each. Metrics: `T_p` = mean total time, `S_p = T_seq/T_p`, `E_p = S_p/p`, `LI = max(t_i)/mean(t_i)` with `t_i` the local processing time of process i.

### b. Sequential baseline
`baseline_secuencial.py` (unchanged) loads the query words into a set, reads every `file_*.txt` line by line, lowercases tokens, counts those in the set with a `Counter`, and prints time and top 10.

### c. MPI version 1 (`mpi1.py`)
Rank 0 reads `consulta.txt` and broadcasts it (`comm.bcast`), lists and sorts the files, and assigns them statically **round-robin** (file i → rank i mod p) with `comm.scatter`. Each rank counts its files locally (same tokenization as the baseline) and measures its local time. Partial counters, file counts and local times are gathered in rank 0 (`comm.gather`), which merges them, prints the per-process report, LI and the top 10. The total time includes everything from the start barrier to the end of the gather.

### d. Test procedure
Check that the top 10 of mpi1 and mpi2 equals the baseline's (words and counts); run ≥3 times per version; compute metrics with `analyze.py`.

## 3. Experimental Plan Execution
### a. Sequential baseline timing
T_seq = <<value>> s. Top 10 identical to MPI output: <<yes/no>>.

### b. MPI version 1 timing results
<<paste mpi1 table from analyze.py>>

### c. Load imbalance evidence
<<paste per-process local times and LI for mpi1>>

### d. MPI version 2
**Correction:** `mpi2.py` keeps the same pipeline but replaces round-robin with size-aware balancing (LPT): files are sorted by size (bytes, a proxy for work) in descending order and each file is given to the process with the smallest accumulated size. This evens out the large 50k–100k-token files that round-robin can pile on one rank.
<<paste mpi2 table and LI>>

| Version | T_p (s) | Speedup | Efficiency | LI |
|---|---|---|---|---|
| mpi1 | <<>> | <<>> | <<>> | <<>> |
| mpi2 | <<>> | <<>> | <<>> | <<>> |

## 4. Analysis
### a. Parallel performance
<<Did mpi1 beat T_seq? How close is S_p to p? Gap explained by: process startup, file I/O (disk-bound), the sequential parts on rank 0 (listing files, gather/merge), imbalance, and cores shared or oversubscribed in Docker/VM. Cite your numbers.>>

### b. Load balance
<<Per-process times and LI of mpi1; did mpi2 lower LI and did T_p improve beyond run-to-run variation?>>

### c. Limitations
Single dataset and machine; only <<n>> runs per configuration (variance); Docker/VM overhead and OS file cache effects; a single value of p (no scalability curve); file size as a proxy for work; Python tokenization cost dominates, so results may not generalize to other workloads.

## 5. Conclusions
<<One paragraph: recommended strategy, citing measured speedup, efficiency and LI.>>
