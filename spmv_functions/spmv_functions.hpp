// Copyright (C) 2024 Sadikov Damir
// github.com/Damlrca/coursework_spmv

// used functions from albus library and my adoptation of it

#ifndef SPMV_FUNCTIONS_HPP
#define SPMV_FUNCTIONS_HPP

#include <riscv_vector.h>

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

template<int sigma>
void spmv_sell_c_sigma_noalloc(const matrix_SELL_C_sigma<4, sigma>& mtx, const vector_format& vec, int threads_num, vector_format& res) {
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / 4; i++) {
		vfloat64m1_t v_summ = __riscv_vfmv_v_f_f64m1(0.0, 4);
		for (int j = mtx.cs[i]; j < mtx.cs[i + 1]; j += 4) {
			// index of column in bits
			vuint32mf2_t index_shftd = __riscv_vle32_v_u32mf2(reinterpret_cast<uint32_t *>(mtx.col + j), 4);
			vfloat64m1_t v_1 = __riscv_vluxei32_v_f64m1(vec.value, index_shftd, 4);
			vfloat64m1_t v_2 = __riscv_vle64_v_f64m1(mtx.value + j, 4);
			v_summ = __riscv_vfmacc_vv_f64m1(v_summ, v_1, v_2, 4);
		}
		__riscv_vse64_v_f64m1(res.value + i * 4, v_summ, 4);
	}
}

template<int sigma>
void spmv_sell_c_sigma_noalloc(const matrix_SELL_C_sigma<8, sigma>& mtx, const vector_format& vec, int threads_num, vector_format& res) {
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / 8; i++) {
		vfloat64m2_t v_summ = __riscv_vfmv_v_f_f64m2(0.0, 8);
		for (int j = mtx.cs[i]; j < mtx.cs[i + 1]; j += 8) {
			// index of column in bits
			vuint32m1_t index_shftd = __riscv_vle32_v_u32m1(reinterpret_cast<uint32_t *>(mtx.col + j), 8);
			vfloat64m2_t v_1 = __riscv_vluxei32_v_f64m2(vec.value, index_shftd, 8);
			vfloat64m2_t v_2 = __riscv_vle64_v_f64m2(mtx.value + j, 8);
			v_summ = __riscv_vfmacc_vv_f64m2(v_summ, v_1, v_2, 8);
		}
		__riscv_vse64_v_f64m2(res.value + i * 8, v_summ, 8);
	}
}

template<int sigma>
void spmv_sell_c_sigma_noalloc(const matrix_SELL_C_sigma<16, sigma>& mtx, const vector_format& vec, int threads_num, vector_format& res) {
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / 16; i++) {
		vfloat64m4_t v_summ = __riscv_vfmv_v_f_f64m4(0.0, 16);
		for (int j = mtx.cs[i]; j < mtx.cs[i + 1]; j += 16) {
			// index of column in bits
			vuint32m2_t index_shftd = __riscv_vle32_v_u32m2(reinterpret_cast<uint32_t *>(mtx.col + j), 16);
			vfloat64m4_t v_1 = __riscv_vluxei32_v_f64m4(vec.value, index_shftd, 16);
			vfloat64m4_t v_2 = __riscv_vle64_v_f64m4(mtx.value + j, 16);
			v_summ = __riscv_vfmacc_vv_f64m4(v_summ, v_1, v_2, 16);
		}
		__riscv_vse64_v_f64m4(res.value + i * 16, v_summ, 16);
	}
}

template<int sigma>
void spmv_sell_c_sigma_noalloc(const matrix_SELL_C_sigma<32, sigma>& mtx, const vector_format& vec, int threads_num, vector_format& res) {
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / 32; i++) {
		vfloat64m8_t v_summ = __riscv_vfmv_v_f_f64m8(0.0, 32);
		for (int j = mtx.cs[i]; j < mtx.cs[i + 1]; j += 32) {
			// index of column in bits
			vuint32m4_t index_shftd = __riscv_vle32_v_u32m4(reinterpret_cast<uint32_t *>(mtx.col + j), 32);
			vfloat64m8_t v_1 = __riscv_vluxei32_v_f64m8(vec.value, index_shftd, 32);
			vfloat64m8_t v_2 = __riscv_vle64_v_f64m8(mtx.value + j, 32);
			v_summ = __riscv_vfmacc_vv_f64m8(v_summ, v_1, v_2, 32);
		}
		__riscv_vse64_v_f64m8(res.value + i * 32, v_summ, 32);
	}
}

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
