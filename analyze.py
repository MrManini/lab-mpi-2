"""Reads results/logs/*.log (made by run_all.sh) and prints the tables for the README.
Usage: python analyze.py [results_dir]"""
import glob, os, re, sys
from statistics import mean

res = sys.argv[1] if len(sys.argv) > 1 else "results"
logs = os.path.join(res, "logs")

def read(p):
    return open(p, encoding="utf-8", errors="ignore").read()

m = re.search(r"Tiempo de ejecuci.n:\s*([\d.]+)", read(os.path.join(logs, "baseline_secuencial.log")))
tseq = float(m.group(1))
print(f"T_seq = {tseq:.4f} s\n")

for prog in ("mpi1", "mpi2"):
    print(f"### {prog}")
    print("| np | runs | T_p (avg, s) | Speedup | Efficiency | LI (avg) |")
    print("|---|---|---|---|---|---|")
    detail = []
    for np_ in sorted({int(re.search(r"np(\d+)_", f).group(1))
                       for f in glob.glob(os.path.join(logs, f"{prog}_np*_run*.log"))}):
        ts, lis = [], []
        for f in sorted(glob.glob(os.path.join(logs, f"{prog}_np{np_}_run*.log"))):
            t = read(f)
            ts.append(float(re.search(r"EXECUTION_TIME=([\d.]+)", t).group(1)))
            lis.append(float(re.search(r"LI=([\d.]+)", t).group(1)))
            detail.append((np_, os.path.basename(f), re.findall(r"rank (\d+): files=(\d+) local_time=([\d.]+)", t)))
        tp = mean(ts); sp = tseq / tp
        print(f"| {np_} | {len(ts)} | {tp:.4f} | {sp:.2f} | {sp/np_:.2f} | {mean(lis):.3f} |")
    print("\nPer-process local times:")
    for np_, name, rows in detail:
        print(f"- {name}: " + ", ".join(f"r{r}: {n} files / {float(t):.3f}s" for r, n, t in rows))
    print()
