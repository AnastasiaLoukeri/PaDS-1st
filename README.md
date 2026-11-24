# One t-SNE gradient with Barnes–Hut: Pthreads, OpenMP, OpenCilk

PDS 2026, Homework 1 (the assignment is in [`ASSIGNMENT.md`](ASSIGNMENT.md)).
For a sparse graph, the program computes the conditional probabilities and `P`, the attraction
`F_attr` (a sparse × dense product), the repulsion `F_rep` and `Z` with a quadtree and a
Barnes–Hut walk, and the gradient `4 (F_attr − F_rep)`, at the given state `Y0`.
Parameters: `u = 10`, `θ = 0.5`, leaf capacity `m = 16`.

## Files

| path | what |
|---|---|
| `src/tsne_seq.cpp` | sequential version |
| `src/tsne_pthreads.cpp` | Pthreads (own thread pool, atomic chunk counter) |
| `src/tsne_openmp.cpp` | OpenMP (`parallel for schedule(dynamic,1)` over chunks) |
| `src/tsne_cilk.cpp` | OpenCilk (`cilk_for`, grain size 1, work stealing) |
| `Makefile` | builds `bin/tsne_seq`, `bin/tsne_pthreads`, `bin/tsne_omp`, `bin/tsne_cilk` |
| `out/<graph>-result-seq.mat` | our `beta`, `Fattr`, `Frep`, `grad`, `Z` for each graph (m = 16, θ = 0.5, u = 10) |
| `results/timings.csv` | phase times: 3 graphs × (sequential + 3 versions × p = 1, 2, 4, 8, 16), median of 5 runs |
| `results/leafsweep.csv` | sequential phase times for leaf capacity m = 4 … 64 |
| `results/direct-omp.txt`, `results/crossover-*.txt` | Barnes–Hut vs direct O(n²) repulsion on `k49_norm_10NN` |
| `results/correctness.txt` | every version against the course reference and against the sequential result |
| `results/machine.txt` | processor, memory, compilers, flags of the measurements |
| `results/figures/`, `results/speedups.csv`, `results/unit_costs.csv` | made from the files above by `scripts/plots.py` |
| `scripts/` | `check.sh` (correctness), `bench.sh` (all measurements), `plots.py` (figures and tables) |
| `starter/`, `data/`, `env/` | from the course template: data reader, graphs, setup script |

The four source files contain the same algorithm. Only the block marked *parallel runtime*
(about 20–100 lines at the top) differs: it implements `par::chunks(nc, f)`, which runs `f(c)`
for `c = 0 … nc−1` with dynamic scheduling, and everything else is written on top of it.

## Build and run (Ubuntu 24.04 or WSL)

```bash
bash env/setup.sh && source ~/.bashrc      # once: gcc, matio, OpenCilk 3.0
make                                        # -O3 -march=native -std=c++17, + -pthread / -fopenmp / -fopencilk
./bin/tsne_omp data/k49_norm_10NN -p 8      # one gradient, 8 threads, phase times + errors vs the reference
```

Options: `-p threads` · `-m leaf capacity` (16) · `-r repetitions` (5; the median is reported) ·
`-theta`, `-u` (default: the values in `<graph>-Y0.mat`) · `-o dir` (save the result) ·
`-cmp file.mat` (compare with a saved result) · `-csv file` (append a row of phase times) ·
`-direct` (also the exact O(n²) repulsion) · `-crossover` (Barnes–Hut vs direct as a function of n).

Any graph `data/<name>.mat` with a symmetric sparse `A` works. Self-loops and non-positive
weights are dropped (with a note); a non-symmetric `A` stops the program with a message.
The state `<name>-Y0.mat` is required; if `<name>-ref.mat` is missing, the comparison with the
reference is skipped.

## Reproduce everything

```bash
bash scripts/check.sh                       # -> out/*.mat, results/correctness.txt
bash scripts/bench.sh                       # -> results/*.csv, *.txt   (old files go to results/old/)
python3 scripts/plots.py results --backend openmp --cores 8   # -> results/figures, speedups.csv, unit_costs.csv
```

`--cores` is the number of physical cores of the machine (8 for the measurements here, see
`results/machine.txt`); thread counts above it share cores (hyperthreading) and are shaded in the figures.

## Correctness criteria (from the assignment)

`β`: max |β − β_ref|. `F_attr`, `F_rep`: max |x − x_ref| / max |x_ref|. `Z`: |Z − Z_ref| / Z_ref.
The reference used Barnes–Hut at θ = 0.1; at θ = 0.5 errors around 10⁻² are expected for `F_rep` and `Z`.
All partitions of the work are independent of the number of threads and partial sums are added
in a fixed order, so the Pthreads and OpenMP results are bit-for-bit equal to the sequential one;
OpenCilk is compiled by clang and differs only by round-off.
