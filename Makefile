# Four builds of the same program:  make  ->  bin/tsne_seq  bin/tsne_pthreads  bin/tsne_omp  bin/tsne_cilk
#   make seq | pthreads | omp | cilk     build one
#   CILKCXX=/path/to/opencilk/bin/clang++ make cilk
CXX      ?= g++
CC       ?= gcc
CILKCXX  ?= $(firstword $(wildcard /opt/opencilk-3.0.0/bin/clang++) clang++)
ARCH     ?= -march=native
OPT      ?= -O3
CXXFLAGS := $(OPT) $(ARCH) -std=c++17 -Wall -Wextra -fopenmp-simd -Istarter -Isrc
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

SRC = src/tsne.cpp src/par.h
bin/tsne_seq: $(SRC) bin/data.o
	$(CXX) $(CXXFLAGS) $(MATIO_CFLAGS) -DBACKEND_SEQ -o $@ src/tsne.cpp bin/data.o $(MATIO_LIBS)
bin/tsne_pthreads: $(SRC) bin/data.o
	$(CXX) $(CXXFLAGS) $(MATIO_CFLAGS) -DBACKEND_PTHREADS -pthread -o $@ src/tsne.cpp bin/data.o $(MATIO_LIBS)
bin/tsne_omp: $(SRC) bin/data.o
	$(CXX) $(CXXFLAGS) $(MATIO_CFLAGS) -DBACKEND_OPENMP -fopenmp -o $@ src/tsne.cpp bin/data.o $(MATIO_LIBS)
bin/tsne_cilk: $(SRC) bin/data.o
	$(CILKCXX) $(CXXFLAGS) $(MATIO_CFLAGS) -DBACKEND_CILK -fopencilk -o $@ src/tsne.cpp bin/data.o $(MATIO_LIBS)

clean:
	rm -rf bin
.PHONY: all seq pthreads omp cilk clean
