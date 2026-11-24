// tsne_cilk.cpp -- one gradient of the t-SNE objective for a large sparse graph.
//
//   1. P      : per-node perplexity bisection (beta_i), then symmetrization
//   2. attr   : F_attr = (D - A) Y^T, a sparse x dense (n x 2) product, A_ij = p_ij/(1+d_ij^2)
//   3. tree   : adaptive quadtree built over Morton-sorted points
//   4. rep    : Barnes-Hut walk -> unnormalized F_rep and Z;  then  grad = 4 (F_attr - F_rep/Z)
//
// THIS FILE: OpenCilk version: cilk_for with grainsize 1 over chunks (work stealing). Build with the OpenCilk clang++ and -fopencilk.
// (The four tsne_*.cpp files contain the same algorithm; only the "parallel runtime" block differs.)
//
// Usage:  bin/tsne_cilk data/<graph> [-p threads] [-m leaf] [-r reps] [-theta t] [-u u]
//                                     [-o outdir] [-csv file] [-direct] [-crossover] [-cmp result.mat]
#include <matio.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <unistd.h>

#define BACKEND_NAME "cilk"

// ---- parallel runtime: the only part of this file that differs between the four versions ----
namespace par {
//   par::init(p, argv)   use p threads
//   par::chunks(nc, f)   run f(c) for c in [0,nc), chunks handed out dynamically
//   par::for_range(n,g,f) run f(lo,hi) on consecutive blocks of at most g indices
// The OpenCilk runtime reads CILK_NWORKERS when the process starts, and there is no public
// call to change the worker count afterwards. So if the variable does not already say p,
// set it and re-execute this program once (nothing has been read or allocated yet).
inline void init(int p, char** argv) {
  char want[16];
  snprintf(want, sizeof want, "%d", p < 1 ? 1 : p);
  const char* have = getenv("CILK_NWORKERS");
  if (have && std::string(have) == want) return;
  setenv("CILK_NWORKERS", want, 1);
  execv("/proc/self/exe", argv);  // Linux
  execv(argv[0], argv);           // fallback (macOS, or no /proc)
  fprintf(stderr, "warning: cannot re-exec; run with CILK_NWORKERS=%s\n", want);
}
inline int nthreads() { return (int)__cilkrts_get_nworkers(); }

template <class F>
void chunks(int64_t nc, F&& f) {
  if (nc <= 0) return;
  if (nc == 1) { f((int64_t)0); return; }
#pragma cilk grainsize 1
  cilk_for(int64_t c = 0; c < nc; c++) f(c);
}

template <class F>
void for_range(int64_t n, int64_t grain, F&& f) {
  int64_t nc = (n + grain - 1) / grain;
  chunks(nc, [&](int64_t c) {
    int64_t lo = c * grain, hi = std::min(n, lo + grain);
    f(lo, hi);
  });
}

}  // namespace par
// ---- end of parallel runtime ----
extern "C" {
#include "data.h"
}

using i64 = int64_t;

// =============================================================== small utilities
// Uninitialized, 64-byte aligned buffer. No zero-fill: the first touch happens inside
// the parallel phases (and is not a serial bottleneck).
template <class T>
struct Buf {
  T* p = nullptr;
  size_t cap = 0;
  Buf() = default;
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  ~Buf() { free(p); }
  void ensure(size_t n) {
    if (n <= cap) return;
    free(p);
    size_t bytes = (n * sizeof(T) + 63) & ~size_t(63);
    p = (T*)aligned_alloc(64, bytes ? bytes : 64);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", bytes); exit(1); }
    cap = n;
  }
  T& operator[](size_t i) { return p[i]; }
  const T& operator[](size_t i) const { return p[i]; }
  T* get() { return p; }
  const T* get() const { return p; }
};

enum { PH_BETA, PH_SYMM, PH_ATTR, PH_KEYS, PH_SORT, PH_BUILD, PH_CM, PH_WALK, PH_ASM, NPH };
static const char* PHN[NPH] = {"beta", "symm", "attr", "keys", "sort", "build", "cm", "walk", "assemble"};

struct Lap {  // accumulates the time since the previous mark into T[phase]
  using clk = std::chrono::steady_clock;
  double* T;
  clk::time_point t;
  explicit Lap(double* T_) : T(T_), t(clk::now()) {}
  void reset() { t = clk::now(); }
  void mark(int ph) {
    auto now = clk::now();
    T[ph] += std::chrono::duration<double>(now - t).count();
    t = now;
  }
};
static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// =============================================================== graph in CSR (int32 columns)
// W is symmetric, so the CSC arrays of the reader are also the CSR arrays.
struct Csr {
  i64 n = 0, nnz = 0;
  Buf<i64> rowptr;
  Buf<int32_t> col;
  Buf<double> w;
};

// Part of reading the input (not timed). Besides the format change it makes sure the
// assumptions of the algorithm hold for ANY input graph, not only the three given ones:
//   - self-loops (w_ii) and entries with w <= 0 are not edges: they are dropped;
//   - columns are sorted inside every row (the mirror lookup in symmetrize needs it);
//   - W must be symmetric: entry (i,j) and entry (j,i) with the same weight. Otherwise P,
//     and the fast path of symmetrize, would be silently wrong, so we stop with a message.
static void to_csr(const graph_t& W, Csr& G) {
  G.n = W.n;
  if (W.n >= (1LL << 31)) { fprintf(stderr, "n must be < 2^31\n"); exit(1); }
  G.rowptr.ensure(W.n + 1);
  G.col.ensure(W.nnz > 0 ? W.nnz : 1);
  G.w.ensure(W.nnz > 0 ? W.nnz : 1);
  i64 nz = 0, dropped = 0;
  for (i64 i = 0; i < W.n; i++) {
    G.rowptr[i] = nz;
    for (i64 k = W.colptr[i]; k < W.colptr[i + 1]; k++) {
      if (W.row[k] == i || !(W.val[k] > 0)) { dropped++; continue; }
      G.col[nz] = (int32_t)W.row[k];
      G.w[nz] = W.val[k];
      nz++;
    }
  }
  G.rowptr[W.n] = nz;
  G.nnz = nz;
  if (dropped) fprintf(stderr, "note: dropped %lld self-loops / non-positive weights\n", (long long)dropped);
  // the transposed-entry lookup needs sorted columns inside every row
  i64 fixed = 0;
  std::vector<std::pair<int32_t, double>> tmp;
  for (i64 i = 0; i < G.n; i++) {
    i64 s = G.rowptr[i], e = G.rowptr[i + 1];
    if (std::is_sorted(G.col.get() + s, G.col.get() + e)) continue;
    tmp.clear();
    for (i64 k = s; k < e; k++) tmp.push_back({G.col[k], G.w[k]});
    std::sort(tmp.begin(), tmp.end());
    for (i64 k = s; k < e; k++) { G.col[k] = tmp[k - s].first; G.w[k] = tmp[k - s].second; }
    fixed++;
  }
  if (fixed) fprintf(stderr, "note: sorted %lld unsorted rows\n", (long long)fixed);
  // symmetry check: every (i,j) needs a mirror (j,i) with the same weight
  const i64 RB = 4096;
  std::vector<char> bad((G.n + RB - 1) / RB, 0);
  par::for_range(G.n, RB, [&](i64 lo, i64 hi) {
    for (i64 i = lo; i < hi; i++) {
      for (i64 k = G.rowptr[i]; k < G.rowptr[i + 1]; k++) {
        const i64 j = G.col[k];
        const int32_t* b = G.col.get() + G.rowptr[j];
        const int32_t* e = G.col.get() + G.rowptr[j + 1];
        const int32_t* it = std::lower_bound(b, e, (int32_t)i);
        if (it == e || *it != (int32_t)i || G.w[it - G.col.get()] != G.w[k]) { bad[lo / RB] = 1; return; }
      }
    }
  });
  if (std::find(bad.begin(), bad.end(), 1) != bad.end()) {
    fprintf(stderr, "error: the adjacency matrix is not symmetric; t-SNE of a graph needs W = W^T\n");
    exit(1);
  }
}

// Work partition: consecutive row blocks of about equal cost (cost of row i ~ degree + 4),
// so that a block of 1000 degree-1 rows and a block with the one degree-3299 row take
// similar time. Blocks are then handed out dynamically.
struct Partition {
  std::vector<i64> b;
  i64 count() const { return (i64)b.size() - 1; }
};
static Partition make_partition(const Csr& G) {
  const i64 ROWC = 4;
  i64 total = G.nnz + ROWC * G.n;
  i64 target = std::min<i64>(65536, std::max<i64>(2048, total / 256));
  i64 nch = std::max<i64>(1, (total + target - 1) / target);
  Partition P;
  P.b.assign(nch + 1, G.n);
  P.b[0] = 0;
  for (i64 c = 1; c < nch; c++) {
    i64 t = (total / nch) * c;
    i64 lo = P.b[c - 1], hi = G.n;  // first row i with cost(i) = rowptr[i] + ROWC*i >= t
    while (lo < hi) {
      i64 mid = (lo + hi) / 2;
      if (G.rowptr[mid] + ROWC * mid < t) lo = mid + 1; else hi = mid;
    }
    P.b[c] = lo;
  }
  return P;
}

// =============================================================== step 1: P
// One row: p_{j|i} ~ (w_ij / wmax)^beta with perplexity min(u, d). Rows with u >= d or with
// all weights equal are uniform (beta = 0).
static void row_probs(const double* w, int d, double u, double* out, double& beta_out,
                      std::vector<double>& l) {
  if (d <= 0) { beta_out = 0; return; }
  double wmax = w[0];
  for (int k = 1; k < d; k++) wmax = std::max(wmax, w[k]);
  bool equal = true;
  for (int k = 0; k < d; k++) if (w[k] != wmax) { equal = false; break; }
  if (u >= d || equal) {
    for (int k = 0; k < d; k++) out[k] = 1.0 / d;
    beta_out = 0;
    return;
  }
  const double Ht = std::log(u);
  l.resize(d);
  for (int k = 0; k < d; k++) l[k] = std::log(w[k] / wmax);  // <= 0
  double beta = 1.0, bmin = -INFINITY, bmax = INFINITY, S = 1.0;
  const int MAXIT = 200;
  for (int it = 0; it < MAXIT; it++) {
    S = 0;
    double T = 0;
    for (int k = 0; k < d; k++) {
      double e = std::exp(beta * l[k]);  // in (0,1]; the largest weight gives exactly 1
      S += e;
      T += e * l[k];
    }
    double H = std::log(S) - beta * T / S;  // entropy of the row at this beta
    double diff = H - Ht;
    if (std::fabs(diff) < 1e-5 || it == MAXIT - 1) break;
    if (diff > 0) {  // entropy too high: sharpen
      bmin = beta;
      beta = (bmax == INFINITY) ? beta * 2 : 0.5 * (beta + bmax);
    } else {
      bmax = beta;
      beta = (bmin == -INFINITY) ? beta * 0.5 : 0.5 * (beta + bmin);
    }
  }
  for (int k = 0; k < d; k++) out[k] = std::exp(beta * l[k]) / S;
  beta_out = beta;
}

// noinline: when gcc inlined this function into its caller, the beta phase of coPapersDBLP
// became about 2x slower than the same code in the OpenMP/OpenCilk builds at p = 1 (measured:
// 0.19 s vs 0.085 s). Keeping it a separate function fixed the sequential build. In the pthreads
// build the p = 1 path of par::chunks inlined the loop body again (beta 0.196 s at p = 1); there
// it now calls the same out-of-line function as the workers. (Why gcc's inlined code is slower
// was not determined.)
__attribute__((noinline)) static void compute_pcond(const Csr& G, const Partition& part, double u, double* beta, double* pc) {
  par::chunks(part.count(), [&](i64 c) {
    std::vector<double> l;
    for (i64 i = part.b[c]; i < part.b[c + 1]; i++) {
      i64 s = G.rowptr[i];
      row_probs(G.w.get() + s, (int)(G.rowptr[i + 1] - s), u, pc + s, beta[i], l);
    }
  });
}

// p_ij = (p_{j|i} + p_{i|j}) / (2n).
// General case: the mirror entry (j,i) is found by binary search in row j (a random access
// into a row that is probably not in cache: this is the expensive part).
// Fast path: if rows i and j are both uniform (beta == 0, e.g. every row of an unweighted
// graph) then p_{j|i} = 1/d_i and p_{i|j} = 1/d_j, no lookup needed. Same value, exactly.
static void symmetrize(const Csr& G, const Partition& part, const double* beta, const double* pc,
                       double* P) {
  const double inv = 1.0 / (2.0 * (double)G.n);
  const int32_t* col = G.col.get();
  par::chunks(part.count(), [&](i64 c) {
    for (i64 i = part.b[c]; i < part.b[c + 1]; i++) {
      const i64 si = G.rowptr[i], di = G.rowptr[i + 1] - si;
      for (i64 k = si; k < si + di; k++) {
        const i64 j = col[k];
        const i64 sj = G.rowptr[j], dj = G.rowptr[j + 1] - sj;
        double t;
        if (beta[i] == 0.0 && beta[j] == 0.0) {
          P[k] = (1.0 / (double)di + 1.0 / (double)dj) * inv;
          continue;
        }
        const int32_t* b = col + sj;
        const int32_t* e = b + dj;
        const int32_t* it = std::lower_bound(b, e, (int32_t)i);
        t = (it != e && *it == (int32_t)i) ? pc[it - col] : 0.0;
        P[k] = (pc[k] + t) * inv;
      }
    }
  });
}

// =============================================================== step 2: attraction (SpMM)
// F_attr[i] = sum_j p_ij / (1 + |y_i - y_j|^2) * (y_i - y_j)  =  ((D - A) Y^T)_i
// Row i costs ~deg(i) gathers of Y[j] and one divide per edge. Memory bound.
static void attraction(const Csr& G, const Partition& part, const double* P, const double* Y,
                       double* Fa) {
  const int32_t* col = G.col.get();
  par::chunks(part.count(), [&](i64 c) {
    for (i64 i = part.b[c]; i < part.b[c + 1]; i++) {
      const double xi = Y[2 * i], yi = Y[2 * i + 1];
      double ax = 0, ay = 0;
      for (i64 k = G.rowptr[i]; k < G.rowptr[i + 1]; k++) {
        const i64 j = col[k];
        const double dx = xi - Y[2 * j], dy = yi - Y[2 * j + 1];
        const double a = P[k] / (1.0 + dx * dx + dy * dy);
        ax += a * dx;
        ay += a * dy;
      }
      Fa[2 * i] = ax;
      Fa[2 * i + 1] = ay;
    }
  });
}

// =============================================================== step 3: quadtree
// Adaptive quadtree over a Morton (Z-order) sort of the points:
//   - the bounding square is cut into a 2^K x 2^K grid; a point's key interleaves its x/y cell
//     coordinates, so a cell of the quadtree = a contiguous range of the sorted points;
//   - points are sorted by key (parallel LSD radix sort, 4 passes x 12 bits);
//   - nodes are created level by level: a cell with more than m points is split into its
//     non-empty quadrants, found by binary search on the sorted keys (no point is moved);
//   - centers of mass are computed bottom-up, deepest level first.
// The sort also puts neighbouring points next to each other in memory, which makes
// consecutive walks touch the same tree cells and the same points.
struct Node {
  double cx, cy;    // center of mass (after build_cm)
  double r2;        // squared diagonal of the cell: 2 * side^2
  int32_t start;    // range of the sorted points inside the cell
  int32_t count;
  int32_t first;    // index of the first child (children are contiguous), -1 for a leaf
  int32_t nchild;   // 0 for a leaf
};
static_assert(sizeof(Node) == 40, "Node should be 40 bytes");

struct QuadTree {
  static constexpr int K = 24;  // grid resolution 2^K per axis, maximum depth
  static constexpr int RB = 12, NB = 1 << RB;
  i64 n = 0;
  int m = 8;
  double x0 = 0, y0 = 0, S = 1;
  Buf<uint64_t> kA, kB;
  Buf<uint32_t> iA, iB, hist;
  uint64_t* key = nullptr;
  uint32_t* idx = nullptr;  // idx[s] = original index of the s-th point in Morton order
  Buf<double> Ys, zpart, bb;
  std::vector<Node> nodes;
  std::vector<i64> ls;  // level l occupies nodes [ls[l], ls[l+1])
  std::vector<int32_t> cnt;
  double r2lev[K + 1];

  static inline uint64_t spread(uint32_t v) {
    uint64_t x = v;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FFULL;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x << 2)) & 0x3333333333333333ULL;
    x = (x | (x << 1)) & 0x5555555555555555ULL;
    return x;
  }

  void make_keys(const double* Y) {
    const i64 BLK = 8192;
    i64 nb = (n + BLK - 1) / BLK;
    bb.ensure(4 * nb);
    par::chunks(nb, [&](i64 c) {
      i64 lo = c * BLK, hi = std::min(n, lo + BLK);
      double mnx = Y[2 * lo], mxx = mnx, mny = Y[2 * lo + 1], mxy = mny;
      for (i64 i = lo; i < hi; i++) {
        mnx = std::min(mnx, Y[2 * i]); mxx = std::max(mxx, Y[2 * i]);
        mny = std::min(mny, Y[2 * i + 1]); mxy = std::max(mxy, Y[2 * i + 1]);
      }
      bb[4 * c] = mnx; bb[4 * c + 1] = mxx; bb[4 * c + 2] = mny; bb[4 * c + 3] = mxy;
    });
    double mnx = bb[0], mxx = bb[1], mny = bb[2], mxy = bb[3];
    for (i64 c = 1; c < nb; c++) {
      mnx = std::min(mnx, bb[4 * c]); mxx = std::max(mxx, bb[4 * c + 1]);
      mny = std::min(mny, bb[4 * c + 2]); mxy = std::max(mxy, bb[4 * c + 3]);
    }
    x0 = mnx; y0 = mny;
    double ext = std::max(mxx - mnx, mxy - mny);
    S = ext > 0 ? ext * (1.0 + 1e-12) : 1.0;
    for (int l = 0; l <= K; l++) {
      double side = std::ldexp(S, -l);
      r2lev[l] = 2.0 * side * side;
    }
    const double scale = std::ldexp(1.0, K) / S;
    const double maxq = (double)((1u << K) - 1);
    kA.ensure(n); kB.ensure(n); iA.ensure(n); iB.ensure(n);
    par::for_range(n, BLK, [&](i64 lo, i64 hi) {
      for (i64 i = lo; i < hi; i++) {
        uint32_t qx = (uint32_t)std::min((Y[2 * i] - x0) * scale, maxq);
        uint32_t qy = (uint32_t)std::min((Y[2 * i + 1] - y0) * scale, maxq);
        kA[i] = (spread(qx) << 1) | spread(qy);
        iA[i] = (uint32_t)i;
      }
    });
  }

  // Stable parallel LSD radix sort of (key, idx) by the low 2K bits of key.
  // Per pass: (1) chunk-local histograms, (2) per-digit prefix over chunks (parallel over
  // digits), (3) 4096 digit totals scanned serially, (4) chunk-local scatter.
  // Chunking is independent of the number of threads, so the order is deterministic.
  void radix_sort() {
    const i64 B = std::max<i64>(2048, (n + 255) / 256);
    const i64 nch = (n + B - 1) / B;
    hist.ensure((size_t)NB * nch);
    uint64_t *ks = kA.get(), *kd = kB.get();
    uint32_t *is = iA.get(), *id = iB.get();
    uint32_t tot[NB], base[NB];
    for (int pass = 0; pass < 2 * K / RB; pass++) {
      const int shift = pass * RB;
      par::chunks(nch, [&](i64 c) {
        uint32_t loc[NB];
        std::memset(loc, 0, sizeof loc);
        i64 lo = c * B, hi = std::min(n, lo + B);
        for (i64 i = lo; i < hi; i++) loc[(ks[i] >> shift) & (NB - 1)]++;
        for (int d = 0; d < NB; d++) hist[(size_t)d * nch + c] = loc[d];
      });
      par::chunks(NB / 256, [&](i64 bc) {
        for (int d = (int)bc * 256; d < (int)bc * 256 + 256; d++) {
          uint32_t run = 0;
          for (i64 c = 0; c < nch; c++) {
            uint32_t t = hist[(size_t)d * nch + c];
            hist[(size_t)d * nch + c] = run;
            run += t;
          }
          tot[d] = run;
        }
      });
      uint32_t s = 0;
      bool trivial = false;
      for (int d = 0; d < NB; d++) {
        base[d] = s;
        s += tot[d];
        if ((i64)tot[d] == n) trivial = true;  // every key has this digit: nothing to move
      }
      if (trivial) continue;
      par::chunks(nch, [&](i64 c) {
        uint32_t loc[NB];
        for (int d = 0; d < NB; d++) loc[d] = base[d] + hist[(size_t)d * nch + c];
        i64 lo = c * B, hi = std::min(n, lo + B);
        for (i64 i = lo; i < hi; i++) {
          uint32_t pos = loc[(ks[i] >> shift) & (NB - 1)]++;
          kd[pos] = ks[i];
          id[pos] = is[i];
        }
      });
      std::swap(ks, kd);
      std::swap(is, id);
    }
    key = ks;
    idx = is;
  }

  // number of points of node a with quadrant digit < q, for q = 1..3 -> b[1..3]; b[0]=0, b[4]=count
  inline void split(const Node& a, int shift, int b[5]) const {
    const uint64_t* kk = key + a.start;
    const int c = a.count;
    b[0] = 0;
    b[4] = c;
    for (int q = 1; q < 4; q++) {
      int lo = b[q - 1], hi = c;
      while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (((kk[mid] >> shift) & 3) < (uint64_t)q) lo = mid + 1; else hi = mid;
      }
      b[q] = lo;
    }
  }

  void build_nodes() {
    nodes.clear();
    ls.clear();
    Node root{};
    root.start = 0; root.count = (int32_t)n; root.r2 = r2lev[0]; root.first = -1;
    nodes.push_back(root);
    ls.push_back(0);
    ls.push_back(1);
    const i64 FB = 256;
    for (int level = 0; level < K; level++) {
      const i64 lo = ls[level], hi = ls[level + 1], nf = hi - lo;
      if (nf == 0) break;
      const int shift = 2 * (K - 1 - level);
      const i64 nfc = (nf + FB - 1) / FB;
      cnt.resize(nf);
      Node* nd = nodes.data();
      par::chunks(nfc, [&](i64 c) {  // pass 1: how many children does each frontier node get?
        for (i64 f = c * FB; f < std::min(nf, (c + 1) * FB); f++) {
          const Node& a = nd[lo + f];
          int k = 0, b[5];
          if (a.count > m) { split(a, shift, b); for (int q = 0; q < 4; q++) k += b[q + 1] > b[q]; }
          cnt[f] = k;
        }
      });
      i64 total = 0;  // exclusive scan: where do the children of each node go?
      for (i64 f = 0; f < nf; f++) { int k = cnt[f]; cnt[f] = (int32_t)total; total += k; }
      if (total == 0) break;
      nodes.resize(hi + total);
      nd = nodes.data();
      par::chunks(nfc, [&](i64 c) {  // pass 2: create the children
        for (i64 f = c * FB; f < std::min(nf, (c + 1) * FB); f++) {
          Node& a = nd[lo + f];
          if (a.count <= m) { a.first = -1; a.nchild = 0; continue; }
          int b[5];
          split(a, shift, b);
          i64 pos = hi + cnt[f];
          a.first = (int32_t)pos;
          int k = 0;
          for (int q = 0; q < 4; q++) {
            if (b[q + 1] > b[q]) {
              Node& ch = nd[pos + k];
              ch = Node{};
              ch.start = a.start + b[q];
              ch.count = b[q + 1] - b[q];
              ch.r2 = r2lev[level + 1];
              ch.first = -1;
              k++;
            }
          }
          a.nchild = k;
        }
      });
      ls.push_back((i64)nodes.size());
    }
  }

  void build_cm() {  // deepest level first; leaves sum their points, inner nodes their children
    const double* ys = Ys.get();
    Node* nd = nodes.data();
    for (i64 l = (i64)ls.size() - 2; l >= 0; l--) {
      const i64 a = ls[l], nn = ls[l + 1] - ls[l], NBK = 512;
      par::chunks((nn + NBK - 1) / NBK, [&](i64 c) {
        for (i64 f = a + c * NBK; f < std::min(a + nn, a + (c + 1) * NBK); f++) {
          Node& x = nd[f];
          double sx = 0, sy = 0;
          if (x.nchild == 0) {
            for (i64 t = x.start; t < (i64)x.start + x.count; t++) { sx += ys[2 * t]; sy += ys[2 * t + 1]; }
          } else {
            for (int k = 0; k < x.nchild; k++) {
              const Node& ch = nd[x.first + k];
              sx += ch.count * ch.cx;
              sy += ch.count * ch.cy;
            }
          }
          x.cx = sx / x.count;
          x.cy = sy / x.count;
        }
      });
    }
  }

  void build(const double* Y, i64 n_, int m_, Lap& lap) {
    n = n_; m = m_;
    make_keys(Y);
    lap.mark(PH_KEYS);
    radix_sort();
    Ys.ensure(2 * n);
    par::for_range(n, 8192, [&](i64 lo, i64 hi) {  // coordinates in Morton order
      for (i64 s = lo; s < hi; s++) { Ys[2 * s] = Y[2 * (i64)idx[s]]; Ys[2 * s + 1] = Y[2 * (i64)idx[s] + 1]; }
    });
    lap.mark(PH_SORT);
    build_nodes();
    lap.mark(PH_BUILD);
    build_cm();
    lap.mark(PH_CM);
  }

  // Barnes-Hut walk. Frep gets the UNNORMALIZED repulsion (original point order), *Z the sum.
  // A cell is accepted when r_cell / d < theta (r_cell = cell diagonal, d = distance from the
  // point to the cell's center of mass) and it does not contain the point itself.
  void walk(double theta, double* Frep, double* Zout) {
    const i64 WB = 256;
    const i64 nc = (n + WB - 1) / WB;
    zpart.ensure(nc);
    const double th2 = theta * theta;
    const Node* nd = nodes.data();
    const double* ys = Ys.get();
    const uint32_t* id = idx;
    par::chunks(nc, [&](i64 c) {
      const i64 lo = c * WB, hi = std::min(n, lo + WB);
      double zl = 0;
      int32_t stk[4 * K + 8];
      for (i64 s = lo; s < hi; s++) {
        const double px = ys[2 * s], py = ys[2 * s + 1];
        double rx = 0, ry = 0, zz = 0;
        int sp = 0;
        stk[sp++] = 0;
        while (sp) {
          const Node& a = nd[stk[--sp]];
          const double dx = px - a.cx, dy = py - a.cy, d2 = dx * dx + dy * dy;
          const bool inside = s >= a.start && s < (i64)a.start + a.count;
          if (!inside && a.r2 < th2 * d2) {  // far enough: the whole cell acts as one point
            const double q = 1.0 / (1.0 + d2);
            const double w = a.count * q;
            zz += w;
            const double w2 = w * q;
            rx += w2 * dx;
            ry += w2 * dy;
          } else if (a.nchild == 0) {  // leaf too close: add its points one by one
            for (i64 t = a.start; t < (i64)a.start + a.count; t++) {
              if (t == s) continue;
              const double ex = px - ys[2 * t], ey = py - ys[2 * t + 1];
              const double q = 1.0 / (1.0 + ex * ex + ey * ey);
              zz += q;
              const double q2 = q * q;
              rx += q2 * ex;
              ry += q2 * ey;
            }
          } else {
            for (int k = a.nchild - 1; k >= 0; k--) stk[sp++] = a.first + k;
          }
        }
        const i64 o = id[s];
        Frep[2 * o] = rx;
        Frep[2 * o + 1] = ry;
        zl += zz;
      }
      zpart[c] = zl;
    });
    double z = 0;
    for (i64 c = 0; c < nc; c++) z += zpart[c];  // fixed order: same Z for every thread count
    *Zout = z;
  }
};

// =============================================================== exact O(n^2) repulsion
// Same unnormalized convention as walk(); normalized by the caller.
static void repulsion_direct_raw(const double* Y, i64 n, double* Frep, double* Zout) {
  const i64 B = 16, nc = (n + B - 1) / B;
  std::vector<double> zp(nc);
  par::chunks(nc, [&](i64 c) {
    double zl = 0;
    for (i64 i = c * B; i < std::min(n, (c + 1) * B); i++) {
      const double xi = Y[2 * i], yi = Y[2 * i + 1];
      double rx = 0, ry = 0, zz = 0;
#pragma omp simd reduction(+ : rx, ry, zz)
      for (i64 j = 0; j < n; j++) {
        const double dx = xi - Y[2 * j], dy = yi - Y[2 * j + 1];
        const double q = 1.0 / (1.0 + dx * dx + dy * dy);
        zz += q;
        rx += q * q * dx;
        ry += q * q * dy;
      }
      Frep[2 * i] = rx;  // j == i adds 0 to the force
      Frep[2 * i + 1] = ry;
      zl += zz - 1.0;    // ... and exactly 1 to the sum: remove it
    }
    zp[c] = zl;
  });
  double z = 0;
  for (i64 c = 0; c < nc; c++) z += zp[c];
  *Zout = z;
}

static void scale_inplace(double* x, i64 len, double s) {
  par::for_range(len, 16384, [&](i64 lo, i64 hi) { for (i64 k = lo; k < hi; k++) x[k] *= s; });
}

// =============================================================== the whole computation
struct Work {
  Buf<double> beta, pc, P, Fa, Fr, grad;
  QuadTree qt;
  double Z = 0;
};

static void run_once(const Csr& G, const Partition& part, const double* Y, double u, double theta,
                     int m, Work& W, double* T) {
  const i64 n = G.n;
  Lap lap(T);
  compute_pcond(G, part, u, W.beta.get(), W.pc.get());
  lap.mark(PH_BETA);
  symmetrize(G, part, W.beta.get(), W.pc.get(), W.P.get());
  lap.mark(PH_SYMM);
  attraction(G, part, W.P.get(), Y, W.Fa.get());
  lap.mark(PH_ATTR);
  W.qt.build(Y, n, m, lap);  // marks keys, sort, build, cm
  W.qt.walk(theta, W.Fr.get(), &W.Z);
  lap.mark(PH_WALK);
  scale_inplace(W.Fr.get(), 2 * n, 1.0 / W.Z);  // F_rep = (unnormalized sum) / Z
  const double *fa = W.Fa.get(), *fr = W.Fr.get();
  double* g = W.grad.get();
  par::for_range(2 * n, 16384, [&](i64 lo, i64 hi) { for (i64 k = lo; k < hi; k++) g[k] = 4.0 * (fa[k] - fr[k]); });
  lap.mark(PH_ASM);
}

// =============================================================== I/O and checks
static double maxabs(const double* a, i64 len) {
  double m = 0;
  for (i64 k = 0; k < len; k++) m = std::max(m, std::fabs(a[k]));
  return m;
}
static double rel_err(const double* x, const double* ref, i64 len) {  // max|x-ref| / max|ref|
  double e = 0;
  for (i64 k = 0; k < len; k++) e = std::max(e, std::fabs(x[k] - ref[k]));
  double d = maxabs(ref, len);
  return d > 0 ? e / d : e;
}
static double abs_err(const double* x, const double* ref, i64 len) {
  double e = 0;
  for (i64 k = 0; k < len; k++) e = std::max(e, std::fabs(x[k] - ref[k]));
  return e;
}
static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static std::string basename_of(const std::string& p) {
  size_t s = p.find_last_of('/');
  return s == std::string::npos ? p : p.substr(s + 1);
}

static void put_var(mat_t* f, const char* name, double* data, size_t r, size_t c) {
  size_t dims[2] = {r, c};
  matvar_t* v = Mat_VarCreate(name, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, data, MAT_F_DONT_COPY_DATA);
  Mat_VarWrite(f, v, MAT_COMPRESSION_ZLIB);
  Mat_VarFree(v);
}
static void write_result(const std::string& path, i64 n, Work& W) {
  mat_t* f = Mat_CreateVer(path.c_str(), nullptr, MAT_FT_MAT5);
  if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); return; }
  double Z = W.Z;
  put_var(f, "beta", W.beta.get(), n, 1);
  put_var(f, "Fattr", W.Fa.get(), 2, n);
  put_var(f, "Frep", W.Fr.get(), 2, n);
  put_var(f, "grad", W.grad.get(), 2, n);
  put_var(f, "Z", &Z, 1, 1);
  Mat_Close(f);
}
static bool read_var(mat_t* f, const char* name, std::vector<double>& out) {
  matvar_t* v = Mat_VarRead(f, name);
  if (!v || v->data_type != MAT_T_DOUBLE) { if (v) Mat_VarFree(v); return false; }
  size_t len = 1;
  for (int d = 0; d < v->rank; d++) len *= v->dims[d];
  out.assign((double*)v->data, (double*)v->data + len);
  Mat_VarFree(v);
  return true;
}

// =============================================================== main
int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr,
            "usage: %s data/<graph> [-p threads] [-m leaf] [-r reps] [-theta t] [-u u]\n"
            "          [-o outdir] [-csv file] [-direct] [-crossover] [-cmp result.mat]\n", argv[0]);
    return 1;
  }
  std::string base = argv[1], outdir, csv, cmpfile;
  int p = (int)std::max(1u, std::thread::hardware_concurrency()), m = 16, reps = 5;
  double theta = -1, u = -1;
  bool direct = false, crossover = false;
  for (int a = 2; a < argc; a++) {
    std::string s = argv[a];
    auto need = [&](void) -> const char* {
      if (a + 1 >= argc) { fprintf(stderr, "missing value after %s\n", s.c_str()); exit(1); }
      return argv[++a];
    };
    if (s == "-p") p = atoi(need());
    else if (s == "-m") m = atoi(need());
    else if (s == "-r") reps = std::max(1, atoi(need()));
    else if (s == "-theta") theta = atof(need());
    else if (s == "-u") u = atof(need());
    else if (s == "-o") outdir = need();
    else if (s == "-csv") csv = need();
    else if (s == "-cmp") cmpfile = need();
    else if (s == "-direct") direct = true;
    else if (s == "-crossover") crossover = true;
    else { fprintf(stderr, "unknown option %s\n", s.c_str()); return 1; }
  }
  par::init(p, argv);

  auto exists = [](const std::string& path) {
    FILE* t = fopen(path.c_str(), "r");
    if (t) fclose(t);
    return t != nullptr;
  };
  graph_t W0;
  state_t S;
  if (read_graph(base.c_str(), &W0) || read_state(base.c_str(), &S)) return 1;
  ref_t R;
  bool have_ref = exists(base + "-ref.mat") && read_ref(base.c_str(), &R) == 0;
  if (S.n != W0.n) { fprintf(stderr, "sizes do not match\n"); return 1; }
  if (u < 0) u = S.u;
  if (theta < 0) theta = S.theta;
  Csr G;
  to_csr(W0, G);
  free_graph(&W0);
  const i64 n = G.n;
  const double* Y = S.Y0;
  const std::string name = basename_of(base);

  i64 dmax = 0;
  for (i64 i = 0; i < n; i++) dmax = std::max(dmax, G.rowptr[i + 1] - G.rowptr[i]);
  printf("%s  backend=%s threads=%d | n=%lld nnz=%lld maxdeg=%lld | u=%g theta=%g leaf m=%d\n", name.c_str(),
         BACKEND_NAME, par::nthreads(), (long long)n, (long long)G.nnz, (long long)dmax, u, theta, m);

  // ------------------------------------------------ crossover: Barnes-Hut vs direct sum, as a function of n
  if (crossover) {
    printf("\nrepulsion only (tree build + walk vs direct), n points sampled with a stride from Y0, median of %d runs\n",
           reps);
    printf("%9s %12s %12s %9s %10s %10s\n", "n", "BH [s]", "direct [s]", "direct/BH", "err F", "err Z");
    std::vector<i64> sizes;
    for (i64 s = 256; s < n; s *= 2) sizes.push_back(s);
    sizes.push_back(n);
    QuadTree qt;
    for (i64 ns : sizes) {
      std::vector<double> Ysub(2 * ns), F1(2 * ns), F2(2 * ns);
      for (i64 i = 0; i < ns; i++) {
        i64 j = i * n / ns;
        Ysub[2 * i] = Y[2 * j]; Ysub[2 * i + 1] = Y[2 * j + 1];
      }
      int rr = reps;  // median of reps runs for every n (the largest direct sum takes < 1 s)
      std::vector<double> tb, td;
      double Z1 = 0, Z2 = 0;
      for (int r = 0; r < rr; r++) {
        double T[NPH] = {0};
        Lap lap(T);
        double t0 = now_s();
        qt.build(Ysub.data(), ns, m, lap);
        qt.walk(theta, F1.data(), &Z1);
        scale_inplace(F1.data(), 2 * ns, 1.0 / Z1);
        tb.push_back(now_s() - t0);
        t0 = now_s();
        repulsion_direct_raw(Ysub.data(), ns, F2.data(), &Z2);
        scale_inplace(F2.data(), 2 * ns, 1.0 / Z2);
        td.push_back(now_s() - t0);
      }
      double tbm = median(tb), tdm = median(td);
      printf("%9lld %12.5f %12.5f %9.2f %10.2e %10.2e\n", (long long)ns, tbm, tdm, tdm / tbm,
             rel_err(F1.data(), F2.data(), 2 * ns), std::fabs(Z1 - Z2) / Z2);
    }
    return 0;
  }

  // ------------------------------------------------ the gradient
  Partition part = make_partition(G);
  Work Wk;
  Wk.beta.ensure(n); Wk.pc.ensure(G.nnz); Wk.P.ensure(G.nnz);
  Wk.Fa.ensure(2 * n); Wk.Fr.ensure(2 * n); Wk.grad.ensure(2 * n);

  std::vector<std::vector<double>> times(NPH);
  std::vector<double> totals;
  for (int r = 0; r < reps; r++) {
    double T[NPH] = {0};
    run_once(G, part, Y, u, theta, m, Wk, T);
    double tot = 0;
    for (int k = 0; k < NPH; k++) { times[k].push_back(T[k]); tot += T[k]; }
    totals.push_back(tot);
  }
  double med[NPH];
  for (int k = 0; k < NPH; k++) med[k] = median(times[k]);
  double medtot = median(totals);

  double sumP = 0;
  for (i64 k = 0; k < G.nnz; k++) sumP += Wk.P[k];
  printf("\nphase times, median of %d runs [s]  (%lld row blocks)\n", reps, (long long)part.count());
  printf("  P:   beta %.5f  symmetrize %.5f\n", med[PH_BETA], med[PH_SYMM]);
  printf("  attr:      %.5f\n", med[PH_ATTR]);
  printf("  quadtree:  keys %.5f  sort %.5f  build %.5f  (%zu nodes, %zu levels)\n", med[PH_KEYS],
         med[PH_SORT], med[PH_BUILD], Wk.qt.nodes.size(), Wk.qt.ls.size() - 1);
  printf("  centers of mass: %.5f\n", med[PH_CM]);
  printf("  walk:      %.5f\n", med[PH_WALK]);
  printf("  assemble:  %.5f\n", med[PH_ASM]);
  printf("  total:     %.5f   (gradient without P: %.5f)\n", medtot, medtot - med[PH_BETA] - med[PH_SYMM]);
  printf("\nsum(P) = %.12f (should be 1),  Z = %.10g\n", sumP, Wk.Z);

  double e_beta = -1, e_fa = -1, e_fr = -1, e_z = -1;
  if (have_ref) {
    e_beta = abs_err(Wk.beta.get(), R.beta, n);
    e_fa = rel_err(Wk.Fa.get(), R.Fattr, 2 * n);
    e_fr = rel_err(Wk.Fr.get(), R.Frep, 2 * n);
    e_z = std::fabs(Wk.Z - R.Z) / R.Z;
    printf("vs reference:  max|beta-ref| = %.3e   Fattr err = %.3e   Frep err = %.3e   Z rel err = %.3e\n",
           e_beta, e_fa, e_fr, e_z);
  }

  if (direct) {
    Buf<double> Fd;
    Fd.ensure(2 * n);
    double Zd = 0;
    std::vector<double> tds;
    for (int r = 0; r < reps; r++) {  // median of reps runs, like the Barnes-Hut times
      double t0 = now_s();
      repulsion_direct_raw(Y, n, Fd.get(), &Zd);
      scale_inplace(Fd.get(), 2 * n, 1.0 / Zd);
      tds.push_back(now_s() - t0);
    }
    double td = median(tds);
    double bh = med[PH_KEYS] + med[PH_SORT] + med[PH_BUILD] + med[PH_CM] + med[PH_WALK];
    printf("direct O(n^2) repulsion (median of %d): %.5f s   Barnes-Hut (build + cm + walk): %.5f s   speedup %.1fx\n",
           reps, td, bh, td / bh);
    printf("Barnes-Hut vs exact:  Frep err = %.3e   Z rel err = %.3e\n", rel_err(Wk.Fr.get(), Fd.get(), 2 * n),
           std::fabs(Wk.Z - Zd) / Zd);
  }

  if (!cmpfile.empty()) {
    mat_t* f = Mat_Open(cmpfile.c_str(), MAT_ACC_RDONLY);
    std::vector<double> b, fa, fr, z;
    if (f && read_var(f, "beta", b) && read_var(f, "Fattr", fa) && read_var(f, "Frep", fr) && read_var(f, "Z", z))
      printf("vs %s:  beta %.2e  Fattr %.2e  Frep %.2e  Z %.2e\n", cmpfile.c_str(), abs_err(Wk.beta.get(), b.data(), n),
             rel_err(Wk.Fa.get(), fa.data(), 2 * n), rel_err(Wk.Fr.get(), fr.data(), 2 * n),
             std::fabs(Wk.Z - z[0]) / z[0]);
    else
      fprintf(stderr, "cannot read %s\n", cmpfile.c_str());
    if (f) Mat_Close(f);
  }

  if (!outdir.empty()) {
    std::string path = outdir + "/" + name + "-result-" + BACKEND_NAME + ".mat";
    write_result(path, n, Wk);
    printf("wrote %s\n", path.c_str());
  }

  if (!csv.empty()) {
    bool fresh = false;
    if (FILE* t = fopen(csv.c_str(), "r")) fclose(t); else fresh = true;
    if (FILE* f = fopen(csv.c_str(), "a")) {
      if (fresh) {
        fprintf(f, "graph,backend,threads,n,nnz,leaf_m,theta,u,reps");
        for (int k = 0; k < NPH; k++) fprintf(f, ",%s", PHN[k]);
        fprintf(f, ",total,err_beta,err_Fattr,err_Frep,err_Z\n");
      }
      fprintf(f, "%s,%s,%d,%lld,%lld,%d,%g,%g,%d", name.c_str(), BACKEND_NAME, par::nthreads(), (long long)n,
              (long long)G.nnz, m, theta, u, reps);
      for (int k = 0; k < NPH; k++) fprintf(f, ",%.6f", med[k]);
      fprintf(f, ",%.6f,%.3e,%.3e,%.3e,%.3e\n", medtot, e_beta, e_fa, e_fr, e_z);
      fclose(f);
    }
  }
  return 0;
}
