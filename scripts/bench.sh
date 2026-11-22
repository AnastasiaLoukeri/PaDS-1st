#!/usr/bin/env bash
# usage: bash scripts/bench.sh "1 2 4 8"     (thread counts to test)
THREADS=${1:-"1 2"}
CSV=results/timings.csv
mkdir -p results
for g in k49_norm_10NN com-Amazon coPapersDBLP; do
  ./bin/tsne_seq data/$g -r 5 -csv $CSV > /dev/null
  for b in pthreads omp cilk; do
    for p in $THREADS; do
      echo "$g $b p=$p"
      ./bin/tsne_$b data/$g -p $p -r 5 -csv $CSV > /dev/null
    done
  done
done
