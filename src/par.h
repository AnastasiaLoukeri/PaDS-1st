// par.h -- one parallel primitive, four implementations.
//
//   par::init(p, argv)      use p threads (workers); argv is only needed by the OpenCilk build
//   par::chunks(nc, f)      run f(c) for every chunk c in [0, nc), chunks handed out
//                           dynamically (first come, first served)
//   par::for_range(n,g,f)   run f(lo, hi) on consecutive blocks of at most g indices
//
// Everything in tsne.cpp is written against these calls only, so the four
// versions run exactly the same algorithm and differ only in the runtime that
// schedules the chunks:
//
//   -DBACKEND_SEQ       plain loop
//   -DBACKEND_PTHREADS  hand-written thread pool (pthread_create + condvar) with an
//                       atomic chunk counter
//   -DBACKEND_OPENMP    #pragma omp parallel for schedule(dynamic,1)
//   -DBACKEND_CILK      cilk_for with grainsize 1 (work stealing)
#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>

#if defined(BACKEND_OPENMP)
#include <omp.h>
#define BACKEND_NAME "openmp"
#elif defined(BACKEND_CILK)
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <unistd.h>
#define BACKEND_NAME "cilk"
#elif defined(BACKEND_PTHREADS)
#include <pthread.h>
#define BACKEND_NAME "pthreads"
#else
#ifndef BACKEND_SEQ
#define BACKEND_SEQ
#endif
#define BACKEND_NAME "seq"
#endif

namespace par {

// ------------------------------------------------------------------ Pthreads
#if defined(BACKEND_PTHREADS)
namespace detail {
struct Pool {
  int P = 1;
  pthread_t* th = nullptr;
  pthread_mutex_t m;
  pthread_cond_t cv_start, cv_done;
  uint64_t gen = 0;   // job counter: a worker wakes up when it changes
  int active = 0;     // workers still busy with the current job
  bool quit = false;
  void (*fn)(void*, int64_t) = nullptr;
  void* ctx = nullptr;
  int64_t nc = 0;
  std::atomic<int64_t> next{0};
};
inline Pool pool;

inline void run_chunks() {
  for (;;) {
    int64_t c = pool.next.fetch_add(1, std::memory_order_relaxed);
    if (c >= pool.nc) break;
    pool.fn(pool.ctx, c);
  }
}

inline void* worker(void*) {
  uint64_t seen = 0;
  for (;;) {
    pthread_mutex_lock(&pool.m);
    while (pool.gen == seen && !pool.quit) pthread_cond_wait(&pool.cv_start, &pool.m);
    if (pool.quit) { pthread_mutex_unlock(&pool.m); break; }
    seen = pool.gen;
    pthread_mutex_unlock(&pool.m);
    run_chunks();
    pthread_mutex_lock(&pool.m);
    if (--pool.active == 0) pthread_cond_signal(&pool.cv_done);
    pthread_mutex_unlock(&pool.m);
  }
  return nullptr;
}

inline void shutdown() {
  if (pool.P <= 1) return;
  pthread_mutex_lock(&pool.m);
  pool.quit = true;
  pthread_cond_broadcast(&pool.cv_start);
  pthread_mutex_unlock(&pool.m);
  for (int t = 0; t < pool.P - 1; t++) pthread_join(pool.th[t], nullptr);
  pool.P = 1;
}
}  // namespace detail

inline void init(int p, char**) {
  using namespace detail;
  if (p < 1) p = 1;
  pool.P = p;
  pthread_mutex_init(&pool.m, nullptr);
  pthread_cond_init(&pool.cv_start, nullptr);
  pthread_cond_init(&pool.cv_done, nullptr);
  pool.th = new pthread_t[p];
  for (int t = 0; t < p - 1; t++) pthread_create(&pool.th[t], nullptr, worker, nullptr);
  std::atexit(shutdown);
}
inline int nthreads() { return detail::pool.P; }

template <class F>
void chunks(int64_t nc, F&& f) {
  using namespace detail;
  if (nc <= 0) return;
  if (nc == 1 || pool.P == 1) { for (int64_t c = 0; c < nc; c++) f(c); return; }
  using Fn = typename std::remove_reference<F>::type;
  auto tramp = [](void* ctx, int64_t c) { (*static_cast<Fn*>(ctx))(c); };
  pthread_mutex_lock(&pool.m);
  pool.fn = tramp;
  pool.ctx = const_cast<void*>(static_cast<const void*>(std::addressof(f)));
  pool.nc = nc;
  pool.next.store(0, std::memory_order_relaxed);
  pool.active = pool.P - 1;
  pool.gen++;
  pthread_cond_broadcast(&pool.cv_start);
  pthread_mutex_unlock(&pool.m);
  run_chunks();  // the calling thread works too
  pthread_mutex_lock(&pool.m);
  while (pool.active > 0) pthread_cond_wait(&pool.cv_done, &pool.m);
  pthread_mutex_unlock(&pool.m);
}

// ------------------------------------------------------------------ OpenMP
#elif defined(BACKEND_OPENMP)
inline void init(int p, char**) { omp_set_num_threads(p < 1 ? 1 : p); }
inline int nthreads() { return omp_get_max_threads(); }

template <class F>
void chunks(int64_t nc, F&& f) {
  if (nc <= 0) return;
  if (nc == 1) { f((int64_t)0); return; }
#pragma omp parallel for schedule(dynamic, 1)
  for (int64_t c = 0; c < nc; c++) f(c);
}

// ------------------------------------------------------------------ OpenCilk
#elif defined(BACKEND_CILK)
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

// ------------------------------------------------------------------ sequential
#else
inline void init(int, char**) {}
inline int nthreads() { return 1; }

template <class F>
void chunks(int64_t nc, F&& f) {
  for (int64_t c = 0; c < nc; c++) f(c);
}
#endif

template <class F>
void for_range(int64_t n, int64_t grain, F&& f) {
  int64_t nc = (n + grain - 1) / grain;
  chunks(nc, [&](int64_t c) {
    int64_t lo = c * grain, hi = std::min(n, lo + grain);
    f(lo, hi);
  });
}

}  // namespace par
