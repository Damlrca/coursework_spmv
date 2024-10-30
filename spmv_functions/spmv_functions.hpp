// Copyright (C) 2024 Sadikov Damir
// github.com/Damlrca/coursework_spmv

// used functions from albus library and my adoptation of it

#ifndef SPMV_FUNCTIONS_HPP
#define SPMV_FUNCTIONS_HPP

#include "../storage_formats/storage_formats.hpp"

// NAIVE

vector_format spmv_naive(const matrix_CSR& mtx_CSR, const vector_format& vec, int threads_num);

// ALBUS

void preproc_albus_balance(const matrix_CSR& mtx_CSR, int* start, int* block_start, int threads_num);

// ALBUS_OMP

vector_format spmv_albus_omp(const matrix_CSR& mtx_CSR, const vector_format& vec, int* start, int* block_start, int threads_num);

// ALBUS_OMP_V

vector_format spmv_albus_omp_v(const matrix_CSR& mtx_CSR, const vector_format& vec, int* start, int* block_start, int threads_num);

// TODO
//vector_format spmv_sellcsigma(const sellcsigma& mtx_CSR, const vector_format& vec);

#endif // !SPMV_FUNCTIONS_HPP
