# Copyright (C) 2024 Sadikov Damir
# github.com/Damlrca/coursework_spmv

all: main_exe main2_exe

# module load Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.8.1
# module load gcc-riscv64-14.2.0
CC = riscv64-unknown-linux-gnu-gcc
CXX = riscv64-unknown-linux-gnu-g++
CFLAGS = -march=rv64gcv -fopenmp -O2

main2_exe: main2.o mtx_input.o mmio.o spmv_functions.o storage_formats.o
	$(CXX) $(CFLAGS) $^ -o main2_exe

main2.o: main/main2.cpp
	$(CXX) -c $(CFLAGS) main/main2.cpp -o $@

main_exe: main.o mtx_input.o mmio.o spmv_functions.o storage_formats.o
	$(CXX) $(CFLAGS) $^ -o main_exe

main.o: main/main.cpp
	$(CXX) -c $(CFLAGS) main/main.cpp -o $@

mtx_input.o: mtx_input/mtx_input.hpp mtx_input/mtx_input.cpp
	$(CXX) -c $(CFLAGS) mtx_input/mtx_input.cpp -o $@

mmio.o: mtx_input/mmio.h mtx_input/mmio.c
	$(CC) -c $(CFLAGS) mtx_input/mmio.c -o $@

spmv_functions.o: spmv_functions/spmv_functions.hpp spmv_functions/spmv_functions.cpp
	$(CXX) -c $(CFLAGS) spmv_functions/spmv_functions.cpp -o $@

storage_formats.o: storage_formats/storage_formats.hpp storage_formats/storage_formats.cpp
	$(CXX) -c $(CFLAGS) storage_formats/storage_formats.cpp -o $@

clean:
	rm -f *.o

# error: no used header *.o in prerequisites -> to compile correctly need to 'make clean'
