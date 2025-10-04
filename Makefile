# Copyright (C) 2025 Sadikov Damir
# github.com/Damlrca/coursework_spmv

# should i make cmake file instead?

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
# main_exe: main.o mtx_io.o mmio.o spmv_functions.o storage_formats.o
#	$(CXX) $(CFLAGS) $^ -o main_exe
# main.o: main/main.cpp
#	$(CXX) -c $(CFLAGS) main/main.cpp -o $@

all: main2_exe


# main2_exe executable for testing spmv (everything required!)
main2_exe: main2.o mtx_io.o mmio.o spmv_functions.o storage_formats.o
	$(CXX) $(CFLAGS) $^ -o $@

# main2_exe executable for testing spmv (spmv_functions.o, mtx_io.o, storage_formats.o required)
main2.o: main/main2.cpp spmv_functions.o mtx_io.o storage_formats.o
	$(CXX) -c $(CFLAGS) $< -o $@


# mtx2bin_exe executable for convertion matrix from mtx to bin format
mtx2bin_exe: mtx2bin.o mtx_io.o mmio.o storage_formats.o
	$(CXX) $(CFLAGS) $^ -o $@

# mtx2bin_exe executable for convertion matrix from mtx to bin format
mtx2bin.o: mtx_io/mtx2bin.cpp mtx_io.o storage_formats.o
	$(CXX) -c $(CFLAGS) $< -o $@


# spmv functions (storage_formats.o required)
spmv_functions.o: spmv_functions/spmv_functions.cpp spmv_functions/spmv_functions.hpp storage_formats.o
	$(CXX) -c $(CFLAGS) $< -o $@

# I/O functions for matrices (storage_formats.o, mmio.o required)
mtx_io.o: mtx_io/mtx_io.cpp mtx_io/mtx_io.hpp storage_formats.o mmio.o
	$(CXX) -c $(CFLAGS) $< -o $@

# storage formats for sparse matrices
storage_formats.o: storage_formats/storage_formats.cpp storage_formats/storage_formats.hpp
	$(CXX) -c $(CFLAGS) $< -o $@

# Matrix Market I/O library
mmio.o: mtx_io/mmio.c mtx_io/mmio.h
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f *.o

