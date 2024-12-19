// Copyright (C) 2024 Sadikov Damir
// github.com/Damlrca/coursework_spmv

// used functions from albus library and my adoptation of it

#ifndef SPMV_FUNCTIONS_HPP
#define SPMV_FUNCTIONS_HPP

#include "../storage_formats/storage_formats.hpp"

vector_format alloc_vector_res(const matrix_CSR& mtx_CSR);

template<int C, int sigma>
vector_format alloc_vector_res(const matrix_SELL_C_sigma<C, sigma>& mtx) {
	vector_format res;
	res.N = mtx.N;
	res.value = new double[res.N];
	std::memset(res.value, 0, sizeof(double) * res.N);
	return res;
}

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

vector_format spmv_sell_c_sigma(const matrix_SELL_C_sigma<4, 1>& mtx, const vector_format& vec, int threads_num);
vector_format spmv_sell_c_sigma(const matrix_SELL_C_sigma<8, 1>& mtx, const vector_format& vec, int threads_num);

void spmv_sell_c_sigma_noalloc(const matrix_SELL_C_sigma<2, 1>& mtx, const vector_format& vec, int threads_num, vector_format& res);
void spmv_sell_c_sigma_noalloc(const matrix_SELL_C_sigma<4, 1>& mtx, const vector_format& vec, int threads_num, vector_format& res);
void spmv_sell_c_sigma_noalloc(const matrix_SELL_C_sigma<8, 1>& mtx, const vector_format& vec, int threads_num, vector_format& res);
void spmv_sell_c_sigma_noalloc(const matrix_SELL_C_sigma<16, 1>& mtx, const vector_format& vec, int threads_num, vector_format& res);

// SELL_C_SIGMA_no_vec

template<int C, int sigma>
void spmv_sell_c_sigma_noalloc_novec(const matrix_SELL_C_sigma<C, sigma>& mtx, const vector_format& vec, int threads_num, vector_format& res) {
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / C; i++) {
		double v_summ[C]{};
		for (int j = mtx.cs[i]; j < mtx.cs[i + 1]; j += C) {
			for (int k = 0; k < C; k++) {
				v_summ[k] += vec.value[*(mtx.col + j + k) >> 3] * mtx.value[j + k];
			}
		}
		for (int k = 0; k < C; k++) {
			res.value[i * C + k] = v_summ[k];
		}
	}
}

#endif // !SPMV_FUNCTIONS_HPP
