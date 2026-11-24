# One source file per version:
#   src/tsne_seq.cpp  src/tsne_pthreads.cpp  src/tsne_openmp.cpp  src/tsne_cilk.cpp
#   make            builds bin/tsne_seq  bin/tsne_pthreads  bin/tsne_omp  bin/tsne_cilk
#   make omp        builds one of them  (seq | pthreads | omp | cilk)
#   CILKCXX=/path/to/opencilk/bin/clang++ make cilk
CXX      ?= g++
CC       ?= gcc
CILKCXX  ?= $(firstword $(wildcard /opt/opencilk-3.0.0/bin/clang++) clang++)
ARCH     ?= -march=native
OPT      ?= -O3
CXXFLAGS := $(OPT) $(ARCH) -std=c++17 -Wall -Wextra -fopenmp-simd -Istarter
MATIO_CFLAGS := $(shell pkg-config --cflags matio 2>/dev/null)
MATIO_LIBS   := $(shell pkg-config --libs matio 2>/dev/null || echo -lmatio)

all: bin/tsne_seq bin/tsne_pthreads bin/tsne_omp bin/tsne_cilk
seq: bin/tsne_seq
pthreads: bin/tsne_pthreads
omp: bin/tsne_omp
cilk: bin/tsne_cilk

bin:
	mkdir -p bin
bin/data.o: starter/data.c starter/data.h | bin
	$(CC) -O2 $(MATIO_CFLAGS) -c -o $@ $<

bin/tsne_seq: src/tsne_seq.cpp bin/data.o
	$(CXX) $(CXXFLAGS) $(MATIO_CFLAGS) -o $@ $< bin/data.o $(MATIO_LIBS)
bin/tsne_pthreads: src/tsne_pthreads.cpp bin/data.o
	$(CXX) $(CXXFLAGS) $(MATIO_CFLAGS) -pthread -o $@ $< bin/data.o $(MATIO_LIBS)
bin/tsne_omp: src/tsne_openmp.cpp bin/data.o
	$(CXX) $(CXXFLAGS) $(MATIO_CFLAGS) -fopenmp -o $@ $< bin/data.o $(MATIO_LIBS)
bin/tsne_cilk: src/tsne_cilk.cpp bin/data.o
	$(CILKCXX) $(CXXFLAGS) $(MATIO_CFLAGS) -fopencilk -o $@ $< bin/data.o $(MATIO_LIBS)

clean:
	rm -rf bin
.PHONY: all seq pthreads omp cilk clean
