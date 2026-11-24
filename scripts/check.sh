#!/usr/bin/env bash
# Correctness: the sequential version saves beta, Fattr, Frep, grad, Z of every graph in out/
# (m = 16, theta = 0.5, u = 10), every parallel version is compared with it, and every version
# with the course reference. Also checks theta = 0 against the direct O(n^2) sum.
#
#   bash scripts/check.sh [p]          (p threads for the parallel versions, default 4)
# Writes results/correctness.txt
set -euo pipefail
P=${1:-4}
cd "$(dirname "$0")/.."
make -s all
mkdir -p out results
OUT=results/correctness.txt
{
  echo "# vs reference: max|beta-ref|, max|x-ref|/max|ref| for Fattr and Frep, |Z-Zref|/Zref"
  echo "# vs out/<graph>-result-seq.mat: the same differences against the sequential result"
  for g in k49_norm_10NN com-Amazon coPapersDBLP; do
    echo; echo "== $g"
    ./bin/tsne_seq data/$g -r 1 -o out | grep -E "backend|^sum|^vs|^wrote"
    for b in pthreads omp cilk; do
      ./bin/tsne_$b data/$g -p $P -r 1 -cmp out/$g-result-seq.mat | grep -E "backend|^vs"
    done
  done
  echo; echo "== theta = 0 must give the exact sum (k49, all versions, two leaf sizes)"
  for m in 1 16; do
    ./bin/tsne_seq data/k49_norm_10NN -r 1 -m $m -theta 0 -direct | grep -E "backend|vs exact"
    for b in pthreads omp cilk; do
      ./bin/tsne_$b data/k49_norm_10NN -p $P -r 1 -m $m -theta 0 -direct | grep -E "backend|vs exact"
    done
  done
} | tee $OUT
echo "wrote $OUT"
