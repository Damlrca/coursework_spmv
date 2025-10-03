# Copyright (C) 2025 Sadikov Damir
# github.com/Damlrca/coursework_spmv

# need to load module before compilation
# old module:
# module load Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.8.1
# new module:
# module load gcc-riscv64-14.2.0

CC = riscv64-unknown-linux-gnu-gcc
CXX = riscv64-unknown-linux-gnu-g++
CFLAGS = -march=rv64gcv -fopenmp -O2

# outdated main.cpp, use main2.cpp instead
# all: main_exe main2_exe
# main_exe: main.o mtx_input.o mmio.o spmv_functions.o storage_formats.o
#	$(CXX) $(CFLAGS) $^ -o main_exe
# main.o: main/main.cpp
#	$(CXX) -c $(CFLAGS) main/main.cpp -o $@

all: main2_exe

# main2_exe executable for testing spmv (everything required!)
main2_exe: main2.o mtx_input.o mmio.o spmv_functions.o storage_formats.o
	$(CXX) $(CFLAGS) $^ -o main2_exe

# main2_exe executable for testing spmv (spmv_functions.o, mtx_input.o, storage_formats.o required)
main2.o: main/main2.cpp spmv_functions.o mtx_input.o storage_formats.o
	$(CXX) -c $(CFLAGS) $< -o $@

# spmv functions (storage_formats.o required)
spmv_functions.o: spmv_functions/spmv_functions.cpp spmv_functions/spmv_functions.hpp storage_formats.o
	$(CXX) -c $(CFLAGS) $< -o $@

# I/O functions for matrices (storage_formats.o, mmio.o required)
mtx_input.o: mtx_input/mtx_input.cpp mtx_input/mtx_input.hpp storage_formats.o mmio.o
	$(CXX) -c $(CFLAGS) $< -o $@

# storage formats for sparse matrices
storage_formats.o: storage_formats/storage_formats.cpp storage_formats/storage_formats.hpp
	$(CXX) -c $(CFLAGS) $< -o $@

# Matrix Market I/O library
mmio.o: mtx_input/mmio.c mtx_input/mmio.h
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f *.o

