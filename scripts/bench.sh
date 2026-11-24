#!/usr/bin/env bash
# Reproduces EVERY measurement behind the figures and tables of the report.
#
#   bash scripts/bench.sh                 # thread counts 1 2 4 8 16, comparisons at p = 8
#   bash scripts/bench.sh "1 2 4 8" 4     # other thread counts / other p for -direct and -crossover
#
# Writes (old files are moved to results/old/ first, so nothing is appended twice):
#   results/timings.csv        3 graphs x (seq + 3 backends x every p), median of 5 runs per phase
#   results/leafsweep.csv      sequential, leaf capacity m = 4..64, k49_norm_10NN and com-Amazon
#   results/direct-omp.txt     k49: Barnes-Hut vs direct O(n^2) repulsion, OpenMP, p = $PCMP
#   results/crossover-omp.txt  k49: Barnes-Hut vs direct as a function of n, OpenMP, p = $PCMP
#   results/crossover-seq.txt  the same, sequential (the crossover depends on p)
#   results/machine.txt        processor, memory, compilers, flags
# Then:  python3 scripts/plots.py results --backend openmp --cores <physical cores>
set -euo pipefail
THREADS=${1:-"1 2 4 8 16"}
PCMP=${2:-8}
R=results
GRAPHS="k49_norm_10NN com-Amazon coPapersDBLP"
cd "$(dirname "$0")/.."
make -s all
mkdir -p $R/old
for f in timings.csv leafsweep.csv direct-omp.txt crossover-omp.txt crossover-seq.txt machine.txt; do
  [ -f $R/$f ] && mv $R/$f $R/old/$f
done

{ lscpu; echo; free -g; echo; g++ --version | head -1
  ${CILKCXX:-/opt/opencilk-3.0.0/bin/clang++} --version | head -1
  echo; echo "flags (Makefile):"; grep -E '^(ARCH|OPT|CXXFLAGS) ' Makefile
  echo "  + -pthread / -fopenmp / -fopencilk for the three parallel versions"; } > $R/machine.txt 2>&1

for g in $GRAPHS; do
  echo "$g seq"
  ./bin/tsne_seq data/$g -r 5 -csv $R/timings.csv > /dev/null
  for b in pthreads omp cilk; do
    for p in $THREADS; do
      echo "$g $b p=$p"
      ./bin/tsne_$b data/$g -p $p -r 5 -csv $R/timings.csv > /dev/null
    done
  done
done

for g in k49_norm_10NN com-Amazon; do
  for m in 4 8 16 32 64; do
    echo "leaf sweep $g m=$m"
    ./bin/tsne_seq data/$g -m $m -r 5 -csv $R/leafsweep.csv > /dev/null
  done
done

echo "direct and crossover (k49)"
./bin/tsne_omp data/k49_norm_10NN -p $PCMP -r 5 -direct    | tee $R/direct-omp.txt
./bin/tsne_omp data/k49_norm_10NN -p $PCMP -r 5 -crossover | tee $R/crossover-omp.txt
./bin/tsne_seq data/k49_norm_10NN -r 5 -crossover          | tee $R/crossover-seq.txt
echo "done; old files are in $R/old/"
