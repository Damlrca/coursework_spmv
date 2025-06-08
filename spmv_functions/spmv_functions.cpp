// Copyright (C) 2025 Sadikov Damir
// github.com/Damlrca/coursework_spmv

// used functions from albus library and my risc-v adoptation of it
// https://github.com/nulidangxueshen/ALBUS
// my fork: https://github.com/Damlrca/ALBUS

#include <riscv_vector.h>

#include <cstring>
#include <omp.h>

#include "spmv_functions.hpp"

// NAIVE

/*
vector_format spmv_naive(const matrix_CSR& mtx_CSR, const vector_format& vec, int threads_num) {
	vector_format res = alloc_vector_res(mtx_CSR);
	spmv_naive_noalloc(mtx_CSR, vec, threads_num, res);
	return res;
}
*/

/*
vector_format spmv_albus_omp(const matrix_CSR& mtx_CSR, const vector_format& vec, int* start, int* block_start, int threads_num) {
	vector_format res = alloc_vector_res(mtx_CSR);
	spmv_albus_omp_noalloc(mtx_CSR, vec, start, block_start, threads_num, res);
	return res;
}
*/

// ALBUS_OMP_V

template<>
double RV_fast1<double, 1>(int start1, int num, int* col_idx, double* mtx_val, double* vec_val) {
	const size_t vl = __riscv_vsetvlmax_e64m1();
	int end1 = start1 + num;
	double answer = 0;
	
	vfloat64m1_t v_summ = __riscv_vfmv_v_f_f64m1(0.0, vl);
	while (num > vl) {
		vuint32mf2_t index = __riscv_vle32_v_u32mf2(reinterpret_cast<uint32_t *>(col_idx + start1), vl);
		vuint32mf2_t index_shftd = __riscv_vsll_vx_u32mf2(index, 3, vl);
		vfloat64m1_t v_1 = __riscv_vle64_v_f64m1(mtx_val + start1, vl);
		vfloat64m1_t v_2 = __riscv_vluxei32_v_f64m1(vec_val, index_shftd, vl);
		v_summ = __riscv_vfmacc_vv_f64m1(v_summ, v_1, v_2, vl);
		start1 += vl;
		num -= vl;
	}
	vfloat64m1_t v_res = __riscv_vfmv_v_f_f64m1(0.0, __riscv_vsetvlmax_e64m1());
	v_res = __riscv_vfredosum_vs_f64m1_f64m1(v_summ, v_res, vl);
	__riscv_vse64_v_f64m1(&answer, v_res, 1);
	while (start1 < end1) {
		answer += mtx_val[start1] * vec_val[col_idx[start1]];
		start1++;
	}
	return answer;
}

template<>
double RV_fast1<double, 2>(int start1, int num, int* col_idx, double* mtx_val, double* vec_val) {
	const size_t vl = __riscv_vsetvlmax_e64m2();
	int end1 = start1 + num;
	double answer = 0;
	
	vfloat64m2_t v_summ = __riscv_vfmv_v_f_f64m2(0.0, vl);
	while (num > vl) {
		vuint32m1_t index = __riscv_vle32_v_u32m1(reinterpret_cast<uint32_t *>(col_idx + start1), vl);
		vuint32m1_t index_shftd = __riscv_vsll_vx_u32m1(index, 3, vl);
		vfloat64m2_t v_1 = __riscv_vle64_v_f64m2(mtx_val + start1, vl);
		vfloat64m2_t v_2 = __riscv_vluxei32_v_f64m2(vec_val, index_shftd, vl);
		v_summ = __riscv_vfmacc_vv_f64m2(v_summ, v_1, v_2, vl);
		start1 += vl;
		num -= vl;
	}
	vfloat64m1_t v_res = __riscv_vfmv_v_f_f64m1(0.0, __riscv_vsetvlmax_e64m1());
	v_res = __riscv_vfredosum_vs_f64m2_f64m1(v_summ, v_res, vl);
	__riscv_vse64_v_f64m1(&answer, v_res, 1);
	while (start1 < end1) {
		answer += mtx_val[start1] * vec_val[col_idx[start1]];
		start1++;
	}
	return answer;
}

template<>
double RV_fast1<double, 4>(int start1, int num, int* col_idx, double* mtx_val, double* vec_val) {
	const size_t vl = __riscv_vsetvlmax_e64m4();
	int end1 = start1 + num;
	double answer = 0;
	
	vfloat64m4_t v_summ = __riscv_vfmv_v_f_f64m4(0.0, vl);
	while (num > vl) {
		vuint32m2_t index = __riscv_vle32_v_u32m2(reinterpret_cast<uint32_t *>(col_idx + start1), vl);
		vuint32m2_t index_shftd = __riscv_vsll_vx_u32m2(index, 3, vl);
		vfloat64m4_t v_1 = __riscv_vle64_v_f64m4(mtx_val + start1, vl);
		vfloat64m4_t v_2 = __riscv_vluxei32_v_f64m4(vec_val, index_shftd, vl);
		v_summ = __riscv_vfmacc_vv_f64m4(v_summ, v_1, v_2, vl);
		start1 += vl;
		num -= vl;
	}
	vfloat64m1_t v_res = __riscv_vfmv_v_f_f64m1(0.0, __riscv_vsetvlmax_e64m1());
	v_res = __riscv_vfredosum_vs_f64m4_f64m1(v_summ, v_res, vl);
	__riscv_vse64_v_f64m1(&answer, v_res, 1);
	while (start1 < end1) {
		answer += mtx_val[start1] * vec_val[col_idx[start1]];
		start1++;
	}
	return answer;
}

template<>
double RV_fast1<double, 8>(int start1, int num, int* col_idx, double* mtx_val, double* vec_val) {
	const size_t vl = __riscv_vsetvlmax_e64m8();
	int end1 = start1 + num;
	double answer = 0;
	
	vfloat64m8_t v_summ = __riscv_vfmv_v_f_f64m8(0.0, vl);
	while (num > vl) {
		vuint32m4_t index = __riscv_vle32_v_u32m4(reinterpret_cast<uint32_t *>(col_idx + start1), vl);
		vuint32m4_t index_shftd = __riscv_vsll_vx_u32m4(index, 3, vl);
		vfloat64m8_t v_1 = __riscv_vle64_v_f64m8(mtx_val + start1, vl);
		vfloat64m8_t v_2 = __riscv_vluxei32_v_f64m8(vec_val, index_shftd, vl);
		v_summ = __riscv_vfmacc_vv_f64m8(v_summ, v_1, v_2, vl);
		start1 += vl;
		num -= vl;
	}
	vfloat64m1_t v_res = __riscv_vfmv_v_f_f64m1(0.0, __riscv_vsetvlmax_e64m1());
	v_res = __riscv_vfredosum_vs_f64m8_f64m1(v_summ, v_res, vl);
	__riscv_vse64_v_f64m1(&answer, v_res, 1);
	while (start1 < end1) {
		answer += mtx_val[start1] * vec_val[col_idx[start1]];
		start1++;
	}
	return answer;
}

template<>
double calculation<double, 1>(int start1, int num, int* col_idx, double* mtx_val, double* vec_val) {
	if (num >= __riscv_vsetvlmax_e64m1())
		return RV_fast1<double, 1>(start1, num, col_idx, mtx_val, vec_val);
	else
		return RV_fast2(start1, num, col_idx, mtx_val, vec_val);
}


template<>
double calculation<double, 2>(int start1, int num, int* col_idx, double* mtx_val, double* vec_val) {
	if (num >= __riscv_vsetvlmax_e64m2())
		return RV_fast1<double, 2>(start1, num, col_idx, mtx_val, vec_val);
	else
		return RV_fast2(start1, num, col_idx, mtx_val, vec_val);
}

template<>
double calculation<double, 4>(int start1, int num, int* col_idx, double* mtx_val, double* vec_val) {
	if (num >= __riscv_vsetvlmax_e64m4())
		return RV_fast1<double, 4>(start1, num, col_idx, mtx_val, vec_val);
	else
		return RV_fast2(start1, num, col_idx, mtx_val, vec_val);
}

template<>
double calculation<double, 8>(int start1, int num, int* col_idx, double* mtx_val, double* vec_val) {
	if (num >= __riscv_vsetvlmax_e64m8())
		return RV_fast1<double, 8>(start1, num, col_idx, mtx_val, vec_val);
	else
		return RV_fast2(start1, num, col_idx, mtx_val, vec_val);
}

// SELL_C_SIGMA

/*
vector_format spmv_albus_omp_v(const matrix_CSR& mtx_CSR, const vector_format& vec, int* start, int* block_start, int threads_num) {
	vector_format res = alloc_vector_res(mtx_CSR);
	spmv_albus_omp_v_noalloc(mtx_CSR, vec, start, block_start, threads_num, res);
	return res;
}
*/

/*
vector_format spmv_sell_c_sigma(const matrix_SELL_C_sigma<4, 1>& mtx, const vector_format& vec, int threads_num) {
	vector_format res = alloc_vector_res(mtx);
	spmv_sell_c_sigma_noalloc(mtx, vec, threads_num, res);
	return res;
}
*/

/*
vector_format spmv_sell_c_sigma(const matrix_SELL_C_sigma<8, 1>& mtx, const vector_format& vec, int threads_num) {
	vector_format res = alloc_vector_res(mtx);
	spmv_sell_c_sigma_noalloc(mtx, vec, threads_num, res);
	return res;
}
*/
