## coursework_spmv

repo for my coursework project

optimization of spmv on risc-v

previos repos:  
https://github.com/Damlrca/coursework  
https://github.com/Damlrca/ALBUS/tree/RISC-V-2

implemented ALBUS method for CSR matrices  
(Haodong Bian and Jianqiang Huang and Lingbin Liu and Dongqiang Huang and Xiaoying
Wang. ALBUS: A method for efficiently processing SpMV using SIMD and Load balancing
// Future Generation Computer Systems. 2021. Vol. 116. P. 371-392, DOI:
https://doi.org/10.1016/j.future.2020.10.036)  
and SELL-C-Sigma matrix format  
(Kreutzer, Moritz and Hager, Georg and Wellein, Gerhard and Fehske, Holger and Bishop,
Alan R., A Unified Sparse Matrix Data Format for Efficient General Sparse Matrix-Vector
Multiplication on Modern Processors with Wide SIMD Units // SSIAM Journal on Scientific
Computing. 2014. Vol. 36:5. P. C401-C423. DOI: https://doi.org/10.1137/130930352)

## Structure:

### main
- main.cpp // outdated main file
- main2.cpp // main file for testing spmv
	- test_naive <double/float>
	- test_albus <double/float>
	- test_albus_v <double/float, 1/2/4/8>
	- test_sell_c_sigma <4/8/16/32, Sigma, double>
	- test_sell_c_sigma <8/16/32/64, Sigma, float>
	- test_sell_c_sigma_novec <C, Sigma, double>

### graphs
- graphs.py // build plots for main2 ouput!
- run_graphs_py.sh //

### mtx_io
- mmio.c, mmio.h // Matrix Market I/O library
- mtx_io.cpp, mtx_io.hpp // functions for matrices input/output
	- read_MTX_to_COO
	- read_MTX_to_CSR
	- read_BIN_to_CSR
	- write_BIN_from_CSR
- mtx2bin.cpp // converter from MTX to binary format for CSR matrices

### spmv_functions
- spmv_functions.cpp, spmv_functions.hpp // spmv functions!
	- alloc_vector_res
	- spmv_naive_noalloc<T>
	- preproc_albus_balance<T>
	- spmv_albus_omp_v_noalloc<double/float, 1/2/4/8>
	- spmv_albus_omp_noalloc<T>
	- spmv_sell_c_sigma_noalloc<4/8/16/32, Sigma, double>
	- spmv_sell_c_sigma_noalloc<8/16/32/64, Sigma, float>
	- spmv_sell_c_sigma_noalloc_novec<C, Sigma, T>

### storage_formats
- storage_formats.cpp, storage_formats.hpp // formats for matrices and vectors
	- matrix_COO<T>
	- matrix_CSR<T>
	- convert_COO_to_CSR(), transpose_CSR()
	- vector_format<T>
	- matrix_SELL_C_sigma<C, Sigma, T>
	- convert_CSR_to_SELL_C_sigma

### unused code
- unused_code.cpp // some unused code like unrolled stuff...

### Makefile // use it to build everything

### run_\<something>.sh // sbatch run_\<something>.sh
