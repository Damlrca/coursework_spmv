// Copyright (C) 2024 Sadikov Damir
// github.com/Damlrca/coursework_spmv

// used functions from albus library and my adoptation of it

#ifndef SPMV_FUNCTIONS_HPP
#define SPMV_FUNCTIONS_HPP

#include "../storage_formats/storage_formats.hpp"

vector_format alloc_vector_res(const matrix_CSR& mtx_CSR);
vector_format alloc_vector_res(const matrix_SELL_C_sigma<8, 1>& mtx);
vector_format alloc_vector_res(const matrix_SELL_C_sigma<4, 1>& mtx);

// NAIVE

vector_format spmv_naive(const matrix_CSR& mtx_CSR, const vector_format& vec, int threads_num);
void spmv_naive_noalloc(const matrix_CSR& mtx_CSR, const vector_format& vec, int threads_num, vector_format& res);

// ALBUS

void preproc_albus_balance(const matrix_CSR& mtx_CSR, int* start, int* block_start, int threads_num);

// ALBUS_OMP

vector_format spmv_albus_omp(const matrix_CSR& mtx_CSR, const vector_format& vec, int* start, int* block_start, int threads_num);
void spmv_albus_omp_noalloc(const matrix_CSR& mtx_CSR, const vector_format& vec, int* start, int* block_start, int threads_num, vector_format& res);

// ALBUS_OMP_V

vector_format spmv_albus_omp_v(const matrix_CSR& mtx_CSR, const vector_format& vec, int* start, int* block_start, int threads_num);
void spmv_albus_omp_v_noalloc(const matrix_CSR& mtx_CSR, const vector_format& vec, int* start, int* block_start, int threads_num, vector_format& res);

// SELL_C_SIGMA

vector_format spmv_sell_c_sigma(const matrix_SELL_C_sigma<8, 1>& mtx, const vector_format& vec, int threads_num);
void spmv_sell_c_sigma_noalloc(const matrix_SELL_C_sigma<8, 1>& mtx, const vector_format& vec, int threads_num, vector_format& res);
vector_format spmv_sell_c_sigma(const matrix_SELL_C_sigma<4, 1>& mtx, const vector_format& vec, int threads_num);
void spmv_sell_c_sigma_noalloc(const matrix_SELL_C_sigma<4, 1>& mtx, const vector_format& vec, int threads_num, vector_format& res);

#endif // !SPMV_FUNCTIONS_HPP
