// Copyright (C) 2025 Sadikov Damir
// github.com/Damlrca/coursework_spmv

// used functions from albus library and my risc-v adoptation of it
// https://github.com/nulidangxueshen/ALBUS
// my fork: https://github.com/Damlrca/ALBUS

#ifndef SPMV_FUNCTIONS_HPP
#define SPMV_FUNCTIONS_HPP

#include <riscv_vector.h>
#include <cassert>

#include "../storage_formats/storage_formats.hpp"

// functions for vector allocation for result of spmv

template <typename T>
vector_format<T> alloc_vector_res(const matrix_CSR<T>& mtx_CSR) {
	vector_format<T> res;
	res.alloc(mtx_CSR.N, 32);
	std::memset(res.value, 0, sizeof(T) * res.N);
	return res;
}

template<int C, int sigma, typename T>
vector_format<T> alloc_vector_res(const matrix_SELL_C_sigma<C, sigma, T>& mtx) {
	vector_format<T> res;
	res.alloc(mtx.N, C);
	std::memset(res.value, 0, sizeof(T) * res.N);
	return res;
}

// NAIVE function

//vector_format spmv_naive(const matrix_CSR& mtx_CSR, const vector_format& vec, int threads_num);

template <typename T>
void spmv_naive_noalloc(const matrix_CSR<T>& mtx_CSR, const vector_format<T>& vec, int threads_num, vector_format<T>& res) {
#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < mtx_CSR.N; i++) {
		T temp = 0;
		for (int j = mtx_CSR.row_id[i]; j < mtx_CSR.row_id[i + 1]; j++) {
			temp += mtx_CSR.value[j] * vec.value[mtx_CSR.col[j]];
		}
		res.value[i] = temp;
	}
}

// ALBUS PREPROC

inline int preproc_albus_binary_search(int *row_id, int block_start_indx, int N) {
	// return max c : row_id[c] <= block_start_indx
	int l = 0, r = N, res = 0;
	while (l <= r) {
		int c = (l + r) / 2;
        if (row_id[c] <= block_start_indx) {
			res = c;
			l = c + 1;
		}
		else {
			r = c - 1;
		}
	}
	return res;
}

template <typename T>
void preproc_albus_balance(const matrix_CSR<T>& mtx_CSR, int* start, int* block_start, int threads_num) {
	const int n = mtx_CSR.N;
	const int nz = mtx_CSR.row_id[n];
	const int TT = nz / threads_num;
	start[threads_num] = n;
	block_start[threads_num] = nz;
	for (int i = 0; i < threads_num; i++) {
		block_start[i] = i * TT;
		start[i] = preproc_albus_binary_search(mtx_CSR.row_id, block_start[i], n);
	}
}

// ALBUS_INTERNAL

template <typename T>
T RV_fast2(int start1, int num, int* col_idx, T* mtx_val, T* vec_val) {
	T answer = 0;
	int end1 = start1 + num;
	while (start1 < end1) {
		answer += mtx_val[start1] * vec_val[col_idx[start1]];
		start1++;
	}
	return answer;
}

template <typename T, int M>
T RV_fast1(int start1, int num, int* col_idx, T* mtx_val, T* vec_val);

template <typename T, int M>
T calculation(int start1, int num, int* col_idx, T* mtx_val, T* vec_val);

// DOUBLE

template<>
double RV_fast1<double, 1>(int start1, int num, int* col_idx, double* mtx_val, double* vec_val);

template<>
double RV_fast1<double, 2>(int start1, int num, int* col_idx, double* mtx_val, double* vec_val);

template<>
double RV_fast1<double, 4>(int start1, int num, int* col_idx, double* mtx_val, double* vec_val);

template<>
double RV_fast1<double, 8>(int start1, int num, int* col_idx, double* mtx_val, double* vec_val);

template<>
double calculation<double, 1>(int start1, int num, int* col_idx, double* mtx_val, double* vec_val);

template<>
double calculation<double, 2>(int start1, int num, int* col_idx, double* mtx_val, double* vec_val);

template<>
double calculation<double, 4>(int start1, int num, int* col_idx, double* mtx_val, double* vec_val);

template<>
double calculation<double, 8>(int start1, int num, int* col_idx, double* mtx_val, double* vec_val);

// FLOAT

template<>
float RV_fast1<float, 1>(int start1, int num, int* col_idx, float* mtx_val, float* vec_val);

template<>
float RV_fast1<float, 2>(int start1, int num, int* col_idx, float* mtx_val, float* vec_val);

template<>
float RV_fast1<float, 4>(int start1, int num, int* col_idx, float* mtx_val, float* vec_val);

template<>
float RV_fast1<float, 8>(int start1, int num, int* col_idx, float* mtx_val, float* vec_val);

template<>
float calculation<float, 1>(int start1, int num, int* col_idx, float* mtx_val, float* vec_val);

template<>
float calculation<float, 2>(int start1, int num, int* col_idx, float* mtx_val, float* vec_val);

template<>
float calculation<float, 4>(int start1, int num, int* col_idx, float* mtx_val, float* vec_val);

template<>
float calculation<float, 8>(int start1, int num, int* col_idx, float* mtx_val, float* vec_val);

// ALBUS_OMP_V

//vector_format spmv_albus_omp_v(const matrix_CSR& mtx_CSR, const vector_format& vec, int* start, int* block_start, int threads_num);

template <typename T, int M>
void albus_thread_block_v(T* mtx_val, int* mtx_col, int* row_id,
                          T* vec_val, T* vec_res,
                          int* start, int* block_start, int thread_id, T* mid_ans) {
	if (start[thread_id] < start[thread_id + 1]) {
		{
			int strt = block_start[thread_id];
			int num = row_id[start[thread_id] + 1] - strt;
			mid_ans[thread_id * 2] = calculation<T, M>(strt, num, mtx_col, mtx_val, vec_val);
		}
		for (int i = start[thread_id] + 1; i < start[thread_id + 1]; i++) {
			int strt = row_id[i];
			int num = row_id[i + 1] - strt;
			vec_res[i] = calculation<T, M>(strt, num, mtx_col, mtx_val, vec_val);
		}
		{
			int strt = row_id[start[thread_id + 1]];
			int num = block_start[thread_id + 1] - strt;
			mid_ans[thread_id * 2 + 1] = calculation<T, M>(strt, num, mtx_col, mtx_val, vec_val);
		}
	}
	else {
		{
			int strt = block_start[thread_id];
			int num = block_start[thread_id + 1] - strt;
			mid_ans[thread_id * 2] = calculation<T, M>(strt, num, mtx_col, mtx_val, vec_val);
			mid_ans[thread_id * 2 + 1] = 0; 
		}
	}
}

template <typename T, int M>
void spmv_albus_omp_v_noalloc(const matrix_CSR<T>& mtx_CSR, const vector_format<T>& vec, int* start, int* block_start, int threads_num, vector_format<T>& res) {
	std::memset(res.value, 0, sizeof(T) * res.N);
	
	T *mid_ans = new T[threads_num * 2];
	
#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < threads_num; i++) {
        albus_thread_block_v<T, M>(mtx_CSR.value, mtx_CSR.col, mtx_CSR.row_id,
                           vec.value, res.value, start, block_start, i, mid_ans);
	}
	
	res.value[start[0]] = mid_ans[0];
	for (int i = 1; i < threads_num; i++) {
		res.value[start[i]] += mid_ans[i * 2 - 1] + mid_ans[i * 2];
	}
	
	delete[] mid_ans;
}

// ALBUS_OMP

//vector_format spmv_albus_omp(const matrix_CSR& mtx_CSR, const vector_format& vec, int* start, int* block_start, int threads_num);

template <typename T>
void albus_thread_block(T* mtx_val, int* mtx_col, int* row_id,
                        T* vec_val, T* vec_res,
                        int* start, int* block_start, int thread_id, T* mid_ans) {
	if (start[thread_id] < start[thread_id + 1]) {
		T temp = 0;
		for (int j = block_start[thread_id]; j < row_id[start[thread_id] + 1]; j++) {
			temp += mtx_val[j] * vec_val[mtx_col[j]];
		}
		mid_ans[thread_id * 2] = temp;
		for (int i = start[thread_id] + 1; i < start[thread_id + 1]; i++) {
			temp = 0;
			for (int j = row_id[i]; j < row_id[i + 1]; j++) {
				temp += mtx_val[j] * vec_val[mtx_col[j]];
			}
			vec_res[i] = temp;
		}
		temp = 0;
		for (int j = row_id[start[thread_id + 1]]; j < block_start[thread_id + 1]; j++) {
			temp += mtx_val[j] * vec_val[mtx_col[j]];
		}
		mid_ans[thread_id * 2 + 1] = temp;
	}
	else {
		T temp = 0;
		for (int j = block_start[thread_id]; j < block_start[thread_id + 1]; j++) {
			temp += mtx_val[j] * vec_val[mtx_col[j]];
		}
		mid_ans[thread_id * 2] = temp;
		mid_ans[thread_id * 2 + 1] = 0; 
	}
}

template <typename T>
void spmv_albus_omp_noalloc(const matrix_CSR<T>& mtx_CSR, const vector_format<T>& vec, int* start, int* block_start, int threads_num, vector_format<T>& res) {
	std::memset(res.value, 0, sizeof(T) * res.N);
	
	T *mid_ans = new T[threads_num * 2];
	
#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < threads_num; i++) {
        albus_thread_block(mtx_CSR.value, mtx_CSR.col, mtx_CSR.row_id,
                           vec.value, res.value, start, block_start, i, mid_ans);
	}
	
	res.value[start[0]] = mid_ans[0];
	for (int i = 1; i < threads_num; i++) {
		res.value[start[i]] += mid_ans[i * 2 - 1] + mid_ans[i * 2];
	}
	
	delete[] mid_ans;
}

// SELL_C_SIGMA

//vector_format spmv_sell_c_sigma(const matrix_SELL_C_sigma<4, 1>& mtx, const vector_format& vec, int threads_num);
//vector_format spmv_sell_c_sigma(const matrix_SELL_C_sigma<8, 1>& mtx, const vector_format& vec, int threads_num);

// spmv_sell_c_sigma_noalloc <double>

template<int sigma>
void spmv_sell_c_sigma_noalloc(const matrix_SELL_C_sigma<4, sigma, double>& mtx, const vector_format<double>& vec, int threads_num, vector_format<double>& res) {
	size_t vl = __riscv_vsetvlmax_e64m1();
	assert(vl == 4);
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / vl; i++) {
		vfloat64m1_t v_summ = __riscv_vfmv_v_f_f64m1(0.0, vl);
		for (int j = mtx.cs[i]; j < mtx.cs[i + 1]; j += vl) {
			vuint32mf2_t index_shftd = __riscv_vle32_v_u32mf2(reinterpret_cast<uint32_t *>(mtx.col + j), vl);
			vfloat64m1_t v_1 = __riscv_vluxei32_v_f64m1(vec.value, index_shftd, vl);
			vfloat64m1_t v_2 = __riscv_vle64_v_f64m1(mtx.value + j, vl);
			v_summ = __riscv_vfmacc_vv_f64m1(v_summ, v_1, v_2, vl);
		}
		__riscv_vse64_v_f64m1(res.value + i * vl, v_summ, vl);
	}
}

template<int sigma>
void spmv_sell_c_sigma_noalloc(const matrix_SELL_C_sigma<8, sigma, double>& mtx, const vector_format<double>& vec, int threads_num, vector_format<double>& res) {
	size_t vl = __riscv_vsetvlmax_e64m2();
	assert(vl == 8);
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / vl; i++) {
		vfloat64m2_t v_summ = __riscv_vfmv_v_f_f64m2(0.0, vl);
		for (int j = mtx.cs[i]; j < mtx.cs[i + 1]; j += vl) {
			vuint32m1_t index_shftd = __riscv_vle32_v_u32m1(reinterpret_cast<uint32_t *>(mtx.col + j), vl);
			vfloat64m2_t v_1 = __riscv_vluxei32_v_f64m2(vec.value, index_shftd, vl);
			vfloat64m2_t v_2 = __riscv_vle64_v_f64m2(mtx.value + j, vl);
			v_summ = __riscv_vfmacc_vv_f64m2(v_summ, v_1, v_2, vl);
		}
		__riscv_vse64_v_f64m2(res.value + i * vl, v_summ, vl);
	}
}

template<int sigma>
void spmv_sell_c_sigma_noalloc(const matrix_SELL_C_sigma<16, sigma, double>& mtx, const vector_format<double>& vec, int threads_num, vector_format<double>& res) {
	size_t vl = __riscv_vsetvlmax_e64m4();
	assert(vl == 16);
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / vl; i++) {
		vfloat64m4_t v_summ = __riscv_vfmv_v_f_f64m4(0.0, vl);
		for (int j = mtx.cs[i]; j < mtx.cs[i + 1]; j += vl) {
			vuint32m2_t index_shftd = __riscv_vle32_v_u32m2(reinterpret_cast<uint32_t *>(mtx.col + j), vl);
			vfloat64m4_t v_1 = __riscv_vluxei32_v_f64m4(vec.value, index_shftd, vl);
			vfloat64m4_t v_2 = __riscv_vle64_v_f64m4(mtx.value + j, vl);
			v_summ = __riscv_vfmacc_vv_f64m4(v_summ, v_1, v_2, vl);
		}
		__riscv_vse64_v_f64m4(res.value + i * vl, v_summ, vl);
	}
}

template<int sigma>
void spmv_sell_c_sigma_noalloc(const matrix_SELL_C_sigma<32, sigma, double>& mtx, const vector_format<double>& vec, int threads_num, vector_format<double>& res) {
	size_t vl = __riscv_vsetvlmax_e64m8();
	assert(vl == 32);
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / vl; i++) {
		vfloat64m8_t v_summ = __riscv_vfmv_v_f_f64m8(0.0, vl);
		for (int j = mtx.cs[i]; j < mtx.cs[i + 1]; j += vl) {
			vuint32m4_t index_shftd = __riscv_vle32_v_u32m4(reinterpret_cast<uint32_t *>(mtx.col + j), vl);
			vfloat64m8_t v_1 = __riscv_vluxei32_v_f64m8(vec.value, index_shftd, vl);
			vfloat64m8_t v_2 = __riscv_vle64_v_f64m8(mtx.value + j, vl);
			v_summ = __riscv_vfmacc_vv_f64m8(v_summ, v_1, v_2, vl);
		}
		__riscv_vse64_v_f64m8(res.value + i * vl, v_summ, vl);
	}
}

// spmv_sell_c_sigma_noalloc <float>

template<int sigma>
void spmv_sell_c_sigma_noalloc(const matrix_SELL_C_sigma<8, sigma, float>& mtx, const vector_format<float>& vec, int threads_num, vector_format<float>& res) {
	size_t vl = __riscv_vsetvlmax_e32m1();
	assert(vl == 8);
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / vl; i++) {
		vfloat32m1_t v_summ = __riscv_vfmv_v_f_f32m1(0.0, vl);
		for (int j = mtx.cs[i]; j < mtx.cs[i + 1]; j += vl) {
			vuint32m1_t index_shftd = __riscv_vle32_v_u32m1(reinterpret_cast<uint32_t *>(mtx.col + j), vl);
			vfloat32m1_t v_1 = __riscv_vluxei32_v_f32m1(vec.value, index_shftd, vl);
			vfloat32m1_t v_2 = __riscv_vle32_v_f32m1(mtx.value + j, vl);
			v_summ = __riscv_vfmacc_vv_f32m1(v_summ, v_1, v_2, vl);
		}
		__riscv_vse32_v_f32m1(res.value + i * vl, v_summ, vl);
	}
}

template<int sigma>
void spmv_sell_c_sigma_noalloc(const matrix_SELL_C_sigma<16, sigma, float>& mtx, const vector_format<float>& vec, int threads_num, vector_format<float>& res) {
	size_t vl = __riscv_vsetvlmax_e32m2();
	assert(vl == 16);
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / vl; i++) {
		vfloat32m2_t v_summ = __riscv_vfmv_v_f_f32m2(0.0, vl);
		for (int j = mtx.cs[i]; j < mtx.cs[i + 1]; j += vl) {
			vuint32m2_t index_shftd = __riscv_vle32_v_u32m2(reinterpret_cast<uint32_t *>(mtx.col + j), vl);
			vfloat32m2_t v_1 = __riscv_vluxei32_v_f32m2(vec.value, index_shftd, vl);
			vfloat32m2_t v_2 = __riscv_vle32_v_f32m2(mtx.value + j, vl);
			v_summ = __riscv_vfmacc_vv_f32m2(v_summ, v_1, v_2, vl);
		}
		__riscv_vse32_v_f32m2(res.value + i * vl, v_summ, vl);
	}
}

template<int sigma>
void spmv_sell_c_sigma_noalloc(const matrix_SELL_C_sigma<32, sigma, float>& mtx, const vector_format<float>& vec, int threads_num, vector_format<float>& res) {
	size_t vl = __riscv_vsetvlmax_e32m4();
	assert(vl == 32);
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / vl; i++) {
		vfloat32m4_t v_summ = __riscv_vfmv_v_f_f32m4(0.0, vl);
		for (int j = mtx.cs[i]; j < mtx.cs[i + 1]; j += vl) {
			vuint32m4_t index_shftd = __riscv_vle32_v_u32m4(reinterpret_cast<uint32_t *>(mtx.col + j), vl);
			vfloat32m4_t v_1 = __riscv_vluxei32_v_f32m4(vec.value, index_shftd, vl);
			vfloat32m4_t v_2 = __riscv_vle32_v_f32m4(mtx.value + j, vl);
			v_summ = __riscv_vfmacc_vv_f32m4(v_summ, v_1, v_2, vl);
		}
		__riscv_vse32_v_f32m4(res.value + i * vl, v_summ, vl);
	}
}

template<int sigma>
void spmv_sell_c_sigma_noalloc(const matrix_SELL_C_sigma<64, sigma, float>& mtx, const vector_format<float>& vec, int threads_num, vector_format<float>& res) {
	size_t vl = __riscv_vsetvlmax_e32m8();
	assert(vl == 64);
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / vl; i++) {
		vfloat32m8_t v_summ = __riscv_vfmv_v_f_f32m8(0.0, vl);
		for (int j = mtx.cs[i]; j < mtx.cs[i + 1]; j += vl) {
			vuint32m8_t index_shftd = __riscv_vle32_v_u32m8(reinterpret_cast<uint32_t *>(mtx.col + j), vl);
			vfloat32m8_t v_1 = __riscv_vluxei32_v_f32m8(vec.value, index_shftd, vl);
			vfloat32m8_t v_2 = __riscv_vle32_v_f32m8(mtx.value + j, vl);
			v_summ = __riscv_vfmacc_vv_f32m8(v_summ, v_1, v_2, vl);
		}
		__riscv_vse32_v_f32m8(res.value + i * vl, v_summ, vl);
	}
}

// spmv_sell_c_sigma_noalloc_unroll4 <double>

/*
template<int sigma>
void spmv_sell_c_sigma_noalloc_unroll4(const matrix_SELL_C_sigma<4, sigma, double>& mtx, const vector_format<double>& vec, int threads_num, vector_format<double>& res) {
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / 4; i++) {
		vfloat64m1_t v_summ_t1 = __riscv_vfmv_v_f_f64m1(0.0, 4);
		vfloat64m1_t v_summ_t2 = __riscv_vfmv_v_f_f64m1(0.0, 4);
		vfloat64m1_t v_summ_t3 = __riscv_vfmv_v_f_f64m1(0.0, 4);
		vfloat64m1_t v_summ_t4 = __riscv_vfmv_v_f_f64m1(0.0, 4);
		int j = mtx.cs[i];
		for (; j + 4 * 3 < mtx.cs[i + 1]; j += 4 * 4) {
			vuint32mf2_t index_shftd_1 = __riscv_vle32_v_u32mf2(reinterpret_cast<uint32_t *>(mtx.col + j + 4 * 0), 4);
			vuint32mf2_t index_shftd_2 = __riscv_vle32_v_u32mf2(reinterpret_cast<uint32_t *>(mtx.col + j + 4 * 1), 4);
			vuint32mf2_t index_shftd_3 = __riscv_vle32_v_u32mf2(reinterpret_cast<uint32_t *>(mtx.col + j + 4 * 2), 4);
			vuint32mf2_t index_shftd_4 = __riscv_vle32_v_u32mf2(reinterpret_cast<uint32_t *>(mtx.col + j + 4 * 3), 4);
			vfloat64m1_t v_1_1 = __riscv_vluxei32_v_f64m1(vec.value, index_shftd_1, 4);
			vfloat64m1_t v_1_2 = __riscv_vluxei32_v_f64m1(vec.value, index_shftd_2, 4);
			vfloat64m1_t v_1_3 = __riscv_vluxei32_v_f64m1(vec.value, index_shftd_3, 4);
			vfloat64m1_t v_1_4 = __riscv_vluxei32_v_f64m1(vec.value, index_shftd_4, 4);
			vfloat64m1_t v_2_1 = __riscv_vle64_v_f64m1(mtx.value + j + 4 * 0, 4);
			vfloat64m1_t v_2_2 = __riscv_vle64_v_f64m1(mtx.value + j + 4 * 1, 4);
			vfloat64m1_t v_2_3 = __riscv_vle64_v_f64m1(mtx.value + j + 4 * 2, 4);
			vfloat64m1_t v_2_4 = __riscv_vle64_v_f64m1(mtx.value + j + 4 * 3, 4);
			v_summ_t1 = __riscv_vfmacc_vv_f64m1(v_summ_t1, v_1_1, v_2_1, 4);
			v_summ_t2 = __riscv_vfmacc_vv_f64m1(v_summ_t2, v_1_2, v_2_2, 4);
			v_summ_t3 = __riscv_vfmacc_vv_f64m1(v_summ_t3, v_1_3, v_2_3, 4);
			v_summ_t4 = __riscv_vfmacc_vv_f64m1(v_summ_t4, v_1_4, v_2_4, 4);
		}
		vfloat64m1_t temp_summ_1 = __riscv_vfadd_vv_f64m1(v_summ_t1, v_summ_t2, 4);
		vfloat64m1_t temp_summ_2 = __riscv_vfadd_vv_f64m1(v_summ_t3, v_summ_t4, 4);
		vfloat64m1_t v_summ = __riscv_vfadd_vv_f64m1(temp_summ_1, temp_summ_2, 4);
		for (; j < mtx.cs[i + 1]; j += 4) {
			vuint32mf2_t index_shftd = __riscv_vle32_v_u32mf2(reinterpret_cast<uint32_t *>(mtx.col + j), 4);
			vfloat64m1_t v_1 = __riscv_vluxei32_v_f64m1(vec.value, index_shftd, 4);
			vfloat64m1_t v_2 = __riscv_vle64_v_f64m1(mtx.value + j, 4);
			v_summ = __riscv_vfmacc_vv_f64m1(v_summ, v_1, v_2, 4);
		}
		__riscv_vse64_v_f64m1(res.value + i * 4, v_summ, 4);
	}
}

template<int sigma>
void spmv_sell_c_sigma_noalloc_unroll4(const matrix_SELL_C_sigma<8, sigma, double>& mtx, const vector_format<double>& vec, int threads_num, vector_format<double>& res) {
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / 8; i++) {
		vfloat64m2_t v_summ_t1 = __riscv_vfmv_v_f_f64m2(0.0, 8);
		vfloat64m2_t v_summ_t2 = __riscv_vfmv_v_f_f64m2(0.0, 8);
		vfloat64m2_t v_summ_t3 = __riscv_vfmv_v_f_f64m2(0.0, 8);
		vfloat64m2_t v_summ_t4 = __riscv_vfmv_v_f_f64m2(0.0, 8);
		int j = mtx.cs[i];
		for (; j + 8 * 3 < mtx.cs[i + 1]; j += 8 * 4) {
			vuint32m1_t index_shftd_1 = __riscv_vle32_v_u32m1(reinterpret_cast<uint32_t *>(mtx.col + j + 8 * 0), 8);
			vuint32m1_t index_shftd_2 = __riscv_vle32_v_u32m1(reinterpret_cast<uint32_t *>(mtx.col + j + 8 * 1), 8);
			vuint32m1_t index_shftd_3 = __riscv_vle32_v_u32m1(reinterpret_cast<uint32_t *>(mtx.col + j + 8 * 2), 8);
			vuint32m1_t index_shftd_4 = __riscv_vle32_v_u32m1(reinterpret_cast<uint32_t *>(mtx.col + j + 8 * 3), 8);
			vfloat64m2_t v_1_1 = __riscv_vluxei32_v_f64m2(vec.value, index_shftd_1, 8);
			vfloat64m2_t v_1_2 = __riscv_vluxei32_v_f64m2(vec.value, index_shftd_2, 8);
			vfloat64m2_t v_1_3 = __riscv_vluxei32_v_f64m2(vec.value, index_shftd_3, 8);
			vfloat64m2_t v_1_4 = __riscv_vluxei32_v_f64m2(vec.value, index_shftd_4, 8);
			vfloat64m2_t v_2_1 = __riscv_vle64_v_f64m2(mtx.value + j + 8 * 0, 8);
			vfloat64m2_t v_2_2 = __riscv_vle64_v_f64m2(mtx.value + j + 8 * 1, 8);
			vfloat64m2_t v_2_3 = __riscv_vle64_v_f64m2(mtx.value + j + 8 * 2, 8);
			vfloat64m2_t v_2_4 = __riscv_vle64_v_f64m2(mtx.value + j + 8 * 3, 8);
			v_summ_t1 = __riscv_vfmacc_vv_f64m2(v_summ_t1, v_1_1, v_2_1, 8);
			v_summ_t2 = __riscv_vfmacc_vv_f64m2(v_summ_t2, v_1_2, v_2_2, 8);
			v_summ_t3 = __riscv_vfmacc_vv_f64m2(v_summ_t3, v_1_3, v_2_3, 8);
			v_summ_t4 = __riscv_vfmacc_vv_f64m2(v_summ_t4, v_1_4, v_2_4, 8);
		}
		vfloat64m2_t temp_summ_1 = __riscv_vfadd_vv_f64m2(v_summ_t1, v_summ_t2, 8);
		vfloat64m2_t temp_summ_2 = __riscv_vfadd_vv_f64m2(v_summ_t3, v_summ_t4, 8);
		vfloat64m2_t v_summ = __riscv_vfadd_vv_f64m2(temp_summ_1, temp_summ_2, 8);
		for (; j < mtx.cs[i + 1]; j += 8) {
			vuint32m1_t index_shftd = __riscv_vle32_v_u32m1(reinterpret_cast<uint32_t *>(mtx.col + j), 8);
			vfloat64m2_t v_1 = __riscv_vluxei32_v_f64m2(vec.value, index_shftd, 8);
			vfloat64m2_t v_2 = __riscv_vle64_v_f64m2(mtx.value + j, 8);
			v_summ = __riscv_vfmacc_vv_f64m2(v_summ, v_1, v_2, 8);
		}
		__riscv_vse64_v_f64m2(res.value + i * 8, v_summ, 8);
	}
}

template<int sigma>
void spmv_sell_c_sigma_noalloc_unroll4(const matrix_SELL_C_sigma<16, sigma, double>& mtx, const vector_format<double>& vec, int threads_num, vector_format<double>& res) {
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / 16; i++) {
		vfloat64m4_t v_summ_t1 = __riscv_vfmv_v_f_f64m4(0.0, 16);
		vfloat64m4_t v_summ_t2 = __riscv_vfmv_v_f_f64m4(0.0, 16);
		vfloat64m4_t v_summ_t3 = __riscv_vfmv_v_f_f64m4(0.0, 16);
		vfloat64m4_t v_summ_t4 = __riscv_vfmv_v_f_f64m4(0.0, 16);
		int j = mtx.cs[i];
		for (; j + 16 * 3 < mtx.cs[i + 1]; j += 16 * 4) {
			vuint32m2_t index_shftd_1 = __riscv_vle32_v_u32m2(reinterpret_cast<uint32_t *>(mtx.col + j + 16 * 0), 16);
			vuint32m2_t index_shftd_2 = __riscv_vle32_v_u32m2(reinterpret_cast<uint32_t *>(mtx.col + j + 16 * 1), 16);
			vuint32m2_t index_shftd_3 = __riscv_vle32_v_u32m2(reinterpret_cast<uint32_t *>(mtx.col + j + 16 * 2), 16);
			vuint32m2_t index_shftd_4 = __riscv_vle32_v_u32m2(reinterpret_cast<uint32_t *>(mtx.col + j + 16 * 3), 16);
			vfloat64m4_t v_1_1 = __riscv_vluxei32_v_f64m4(vec.value, index_shftd_1, 16);
			vfloat64m4_t v_1_2 = __riscv_vluxei32_v_f64m4(vec.value, index_shftd_2, 16);
			vfloat64m4_t v_1_3 = __riscv_vluxei32_v_f64m4(vec.value, index_shftd_3, 16);
			vfloat64m4_t v_1_4 = __riscv_vluxei32_v_f64m4(vec.value, index_shftd_4, 16);
			vfloat64m4_t v_2_1 = __riscv_vle64_v_f64m4(mtx.value + j + 16 * 0, 16);
			vfloat64m4_t v_2_2 = __riscv_vle64_v_f64m4(mtx.value + j + 16 * 1, 16);
			vfloat64m4_t v_2_3 = __riscv_vle64_v_f64m4(mtx.value + j + 16 * 2, 16);
			vfloat64m4_t v_2_4 = __riscv_vle64_v_f64m4(mtx.value + j + 16 * 3, 16);
			v_summ_t1 = __riscv_vfmacc_vv_f64m4(v_summ_t1, v_1_1, v_2_1, 16);
			v_summ_t2 = __riscv_vfmacc_vv_f64m4(v_summ_t2, v_1_2, v_2_2, 16);
			v_summ_t3 = __riscv_vfmacc_vv_f64m4(v_summ_t3, v_1_3, v_2_3, 16);
			v_summ_t4 = __riscv_vfmacc_vv_f64m4(v_summ_t4, v_1_4, v_2_4, 16);
		}
		vfloat64m4_t temp_summ_1 = __riscv_vfadd_vv_f64m4(v_summ_t1, v_summ_t2, 16);
		vfloat64m4_t temp_summ_2 = __riscv_vfadd_vv_f64m4(v_summ_t3, v_summ_t4, 16);
		vfloat64m4_t v_summ = __riscv_vfadd_vv_f64m4(temp_summ_1, temp_summ_2, 16);
		for (; j < mtx.cs[i + 1]; j += 16) {
			vuint32m2_t index_shftd = __riscv_vle32_v_u32m2(reinterpret_cast<uint32_t *>(mtx.col + j), 16);
			vfloat64m4_t v_1 = __riscv_vluxei32_v_f64m4(vec.value, index_shftd, 16);
			vfloat64m4_t v_2 = __riscv_vle64_v_f64m4(mtx.value + j, 16);
			v_summ = __riscv_vfmacc_vv_f64m4(v_summ, v_1, v_2, 16);
		}
		__riscv_vse64_v_f64m4(res.value + i * 16, v_summ, 16);
	}
}

template<int sigma>
void spmv_sell_c_sigma_noalloc_unroll4(const matrix_SELL_C_sigma<32, sigma, double>& mtx, const vector_format<double>& vec, int threads_num, vector_format<double>& res) {
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / 32; i++) {
		vfloat64m8_t v_summ_t1 = __riscv_vfmv_v_f_f64m8(0.0, 32);
		vfloat64m8_t v_summ_t2 = __riscv_vfmv_v_f_f64m8(0.0, 32);
		vfloat64m8_t v_summ_t3 = __riscv_vfmv_v_f_f64m8(0.0, 32);
		vfloat64m8_t v_summ_t4 = __riscv_vfmv_v_f_f64m8(0.0, 32);
		int j = mtx.cs[i];
		for (; j + 32 * 3 < mtx.cs[i + 1]; j += 32 * 4) {
			vuint32m4_t index_shftd_1 = __riscv_vle32_v_u32m4(reinterpret_cast<uint32_t *>(mtx.col + j + 32 * 0), 32);
			vuint32m4_t index_shftd_2 = __riscv_vle32_v_u32m4(reinterpret_cast<uint32_t *>(mtx.col + j + 32 * 1), 32);
			vuint32m4_t index_shftd_3 = __riscv_vle32_v_u32m4(reinterpret_cast<uint32_t *>(mtx.col + j + 32 * 2), 32);
			vuint32m4_t index_shftd_4 = __riscv_vle32_v_u32m4(reinterpret_cast<uint32_t *>(mtx.col + j + 32 * 3), 32);
			vfloat64m8_t v_1_1 = __riscv_vluxei32_v_f64m8(vec.value, index_shftd_1, 32);
			vfloat64m8_t v_1_2 = __riscv_vluxei32_v_f64m8(vec.value, index_shftd_2, 32);
			vfloat64m8_t v_1_3 = __riscv_vluxei32_v_f64m8(vec.value, index_shftd_3, 32);
			vfloat64m8_t v_1_4 = __riscv_vluxei32_v_f64m8(vec.value, index_shftd_4, 32);
			vfloat64m8_t v_2_1 = __riscv_vle64_v_f64m8(mtx.value + j + 32 * 0, 32);
			vfloat64m8_t v_2_2 = __riscv_vle64_v_f64m8(mtx.value + j + 32 * 1, 32);
			vfloat64m8_t v_2_3 = __riscv_vle64_v_f64m8(mtx.value + j + 32 * 2, 32);
			vfloat64m8_t v_2_4 = __riscv_vle64_v_f64m8(mtx.value + j + 32 * 3, 32);
			v_summ_t1 = __riscv_vfmacc_vv_f64m8(v_summ_t1, v_1_1, v_2_1, 32);
			v_summ_t2 = __riscv_vfmacc_vv_f64m8(v_summ_t2, v_1_2, v_2_2, 32);
			v_summ_t3 = __riscv_vfmacc_vv_f64m8(v_summ_t3, v_1_3, v_2_3, 32);
			v_summ_t4 = __riscv_vfmacc_vv_f64m8(v_summ_t4, v_1_4, v_2_4, 32);
		}
		vfloat64m8_t temp_summ_1 = __riscv_vfadd_vv_f64m8(v_summ_t1, v_summ_t2, 32);
		vfloat64m8_t temp_summ_2 = __riscv_vfadd_vv_f64m8(v_summ_t3, v_summ_t4, 32);
		vfloat64m8_t v_summ = __riscv_vfadd_vv_f64m8(temp_summ_1, temp_summ_2, 32);
		for (; j < mtx.cs[i + 1]; j += 32) {
			vuint32m4_t index_shftd = __riscv_vle32_v_u32m4(reinterpret_cast<uint32_t *>(mtx.col + j), 32);
			vfloat64m8_t v_1 = __riscv_vluxei32_v_f64m8(vec.value, index_shftd, 32);
			vfloat64m8_t v_2 = __riscv_vle64_v_f64m8(mtx.value + j, 32);
			v_summ = __riscv_vfmacc_vv_f64m8(v_summ, v_1, v_2, 32);
		}
		__riscv_vse64_v_f64m8(res.value + i * 32, v_summ, 32);
	}
}
*/

// spmv_sell_c_sigma_noalloc_unroll4 <float>

/*
*/

// SELL_C_SIGMA_no_vec

template<int C, int sigma, typename T>
void spmv_sell_c_sigma_noalloc_novec(const matrix_SELL_C_sigma<C, sigma, T>& mtx, const vector_format<T>& vec, int threads_num, vector_format<T>& res) {
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / C; i++) {
		T v_summ[C]{};
		for (int j = mtx.cs[i]; j < mtx.cs[i + 1]; j += C) {
			for (int k = 0; k < C; k++) {
				v_summ[k] += vec.value[*(mtx.col + j + k) / sizeof(T)] * mtx.value[j + k];
			}
		}
		for (int k = 0; k < C; k++) {
			res.value[i * C + k] = v_summ[k];
		}
	}
}

#endif // !SPMV_FUNCTIONS_HPP
