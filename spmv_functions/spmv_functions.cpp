// Copyright (C) 2024 Sadikov Damir
// github.com/Damlrca/coursework_spmv

#include <riscv_vector.h>

#include <cstring>
#include <omp.h>

#include "spmv_functions.hpp"

/*
vector_format spmv_naive(const matrix_CSR& mtx_CSR, const vector_format& vec, int threads_num) {
	vector_format res = alloc_vector_res(mtx_CSR);
	spmv_naive_noalloc(mtx_CSR, vec, threads_num, res);
	return res;
}
*/

static inline void albus_thread_block(double* mtx_val, int* mtx_col, int* row_id,
                                      double* vec_val, double* vec_res,
                                      int* start, int* block_start, int thread_id, double* mid_ans) {
	if (start[thread_id] < start[thread_id + 1]) {
		double temp = 0;
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
		double temp = 0;
		for (int j = block_start[thread_id]; j < block_start[thread_id + 1]; j++) {
			temp += mtx_val[j] * vec_val[mtx_col[j]];
		}
		mid_ans[thread_id * 2] = temp;
		mid_ans[thread_id * 2 + 1] = 0; 
	}
}

void spmv_albus_omp_noalloc(const matrix_CSR<double>& mtx_CSR, const vector_format<double>& vec, int* start, int* block_start, int threads_num, vector_format<double>& res) {
	std::memset(res.value, 0, sizeof(double) * res.N);
	
	double *mid_ans = new double[threads_num * 2];
	
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

/*
vector_format spmv_albus_omp(const matrix_CSR& mtx_CSR, const vector_format& vec, int* start, int* block_start, int threads_num) {
	vector_format res = alloc_vector_res(mtx_CSR);
	spmv_albus_omp_noalloc(mtx_CSR, vec, start, block_start, threads_num, res);
	return res;
}
*/

static inline double RV_fast2(int start1, int num, int* col_idx, double * mtx_val, double* vec_val) {
	double answer = 0;
	int end1 = start1 + num;
	while (start1 < end1) {
		answer += mtx_val[start1] * vec_val[col_idx[start1]];
		start1++;
	}
	return answer;
}

static inline double RV_fast1(int start1, int num, int* col_idx, double * mtx_val, double* vec_val) {
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

static inline double calculation(int start1, int num, int* col_idx, double * mtx_val, double* vec_val) {
	if (num >= __riscv_vsetvlmax_e64m2())
		return RV_fast1(start1, num, col_idx, mtx_val, vec_val);
	else
		return RV_fast2(start1, num, col_idx, mtx_val, vec_val);
}

static inline void albus_thread_block_v(double* mtx_val, int* mtx_col, int* row_id,
                                        double* vec_val, double* vec_res,
                                        int* start, int* block_start, int thread_id, double* mid_ans) {
	if (start[thread_id] < start[thread_id + 1]) {
		{
			int strt = block_start[thread_id];
			int num = row_id[start[thread_id] + 1] - strt;
			mid_ans[thread_id * 2] = calculation(strt, num, mtx_col, mtx_val, vec_val);
		}
		for (int i = start[thread_id] + 1; i < start[thread_id + 1]; i++) {
			int strt = row_id[i];
			int num = row_id[i + 1] - strt;
			vec_res[i] = calculation(strt, num, mtx_col, mtx_val, vec_val);
		}
		{
			int strt = row_id[start[thread_id + 1]];
			int num = block_start[thread_id + 1] - strt;
			mid_ans[thread_id * 2 + 1] = calculation(strt, num, mtx_col, mtx_val, vec_val);
		}
	}
	else {
		{
			int strt = block_start[thread_id];
			int num = block_start[thread_id + 1] - strt;
			mid_ans[thread_id * 2] = calculation(strt, num, mtx_col, mtx_val, vec_val);
			mid_ans[thread_id * 2 + 1] = 0; 
		}
	}
}

void spmv_albus_omp_v_noalloc(const matrix_CSR<double>& mtx_CSR, const vector_format<double>& vec, int* start, int* block_start, int threads_num, vector_format<double>& res) {
	std::memset(res.value, 0, sizeof(double) * res.N);
	
	double *mid_ans = new double[threads_num * 2];
	
#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < threads_num; i++) {
        albus_thread_block_v(mtx_CSR.value, mtx_CSR.col, mtx_CSR.row_id,
                           vec.value, res.value, start, block_start, i, mid_ans);
	}
	
	res.value[start[0]] = mid_ans[0];
	for (int i = 1; i < threads_num; i++) {
		res.value[start[i]] += mid_ans[i * 2 - 1] + mid_ans[i * 2];
	}
	
	delete[] mid_ans;
}

static inline double RV_fast1_m1(int start1, int num, int* col_idx, double * mtx_val, double* vec_val) {
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

static inline double RV_fast1_m2(int start1, int num, int* col_idx, double * mtx_val, double* vec_val) {
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

static inline double RV_fast1_m4(int start1, int num, int* col_idx, double * mtx_val, double* vec_val) {
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

static inline double RV_fast1_m8(int start1, int num, int* col_idx, double * mtx_val, double* vec_val) {
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

static inline double calculation_m1(int start1, int num, int* col_idx, double * mtx_val, double* vec_val) {
	if (num >= __riscv_vsetvlmax_e64m1())
		return RV_fast1_m1(start1, num, col_idx, mtx_val, vec_val);
	else
		return RV_fast2(start1, num, col_idx, mtx_val, vec_val);
}

static inline double calculation_m2(int start1, int num, int* col_idx, double * mtx_val, double* vec_val) {
	if (num >= __riscv_vsetvlmax_e64m2())
		return RV_fast1_m2(start1, num, col_idx, mtx_val, vec_val);
	else
		return RV_fast2(start1, num, col_idx, mtx_val, vec_val);
}

static inline double calculation_m4(int start1, int num, int* col_idx, double * mtx_val, double* vec_val) {
	if (num >= __riscv_vsetvlmax_e64m4())
		return RV_fast1_m4(start1, num, col_idx, mtx_val, vec_val);
	else
		return RV_fast2(start1, num, col_idx, mtx_val, vec_val);
}

static inline double calculation_m8(int start1, int num, int* col_idx, double * mtx_val, double* vec_val) {
	if (num >= __riscv_vsetvlmax_e64m8())
		return RV_fast1_m8(start1, num, col_idx, mtx_val, vec_val);
	else
		return RV_fast2(start1, num, col_idx, mtx_val, vec_val);
}

static inline void albus_thread_block_v_m1(double* mtx_val, int* mtx_col, int* row_id,
                                        double* vec_val, double* vec_res,
                                        int* start, int* block_start, int thread_id, double* mid_ans) {
	if (start[thread_id] < start[thread_id + 1]) {
		{
			int strt = block_start[thread_id];
			int num = row_id[start[thread_id] + 1] - strt;
			mid_ans[thread_id * 2] = calculation_m1(strt, num, mtx_col, mtx_val, vec_val);
		}
		for (int i = start[thread_id] + 1; i < start[thread_id + 1]; i++) {
			int strt = row_id[i];
			int num = row_id[i + 1] - strt;
			vec_res[i] = calculation_m1(strt, num, mtx_col, mtx_val, vec_val);
		}
		{
			int strt = row_id[start[thread_id + 1]];
			int num = block_start[thread_id + 1] - strt;
			mid_ans[thread_id * 2 + 1] = calculation_m1(strt, num, mtx_col, mtx_val, vec_val);
		}
	}
	else {
		{
			int strt = block_start[thread_id];
			int num = block_start[thread_id + 1] - strt;
			mid_ans[thread_id * 2] = calculation_m1(strt, num, mtx_col, mtx_val, vec_val);
			mid_ans[thread_id * 2 + 1] = 0; 
		}
	}
}

static inline void albus_thread_block_v_m2(double* mtx_val, int* mtx_col, int* row_id,
                                        double* vec_val, double* vec_res,
                                        int* start, int* block_start, int thread_id, double* mid_ans) {
	if (start[thread_id] < start[thread_id + 1]) {
		{
			int strt = block_start[thread_id];
			int num = row_id[start[thread_id] + 1] - strt;
			mid_ans[thread_id * 2] = calculation_m2(strt, num, mtx_col, mtx_val, vec_val);
		}
		for (int i = start[thread_id] + 1; i < start[thread_id + 1]; i++) {
			int strt = row_id[i];
			int num = row_id[i + 1] - strt;
			vec_res[i] = calculation_m2(strt, num, mtx_col, mtx_val, vec_val);
		}
		{
			int strt = row_id[start[thread_id + 1]];
			int num = block_start[thread_id + 1] - strt;
			mid_ans[thread_id * 2 + 1] = calculation_m2(strt, num, mtx_col, mtx_val, vec_val);
		}
	}
	else {
		{
			int strt = block_start[thread_id];
			int num = block_start[thread_id + 1] - strt;
			mid_ans[thread_id * 2] = calculation_m2(strt, num, mtx_col, mtx_val, vec_val);
			mid_ans[thread_id * 2 + 1] = 0; 
		}
	}
}

static inline void albus_thread_block_v_m4(double* mtx_val, int* mtx_col, int* row_id,
                                        double* vec_val, double* vec_res,
                                        int* start, int* block_start, int thread_id, double* mid_ans) {
	if (start[thread_id] < start[thread_id + 1]) {
		{
			int strt = block_start[thread_id];
			int num = row_id[start[thread_id] + 1] - strt;
			mid_ans[thread_id * 2] = calculation_m4(strt, num, mtx_col, mtx_val, vec_val);
		}
		for (int i = start[thread_id] + 1; i < start[thread_id + 1]; i++) {
			int strt = row_id[i];
			int num = row_id[i + 1] - strt;
			vec_res[i] = calculation_m4(strt, num, mtx_col, mtx_val, vec_val);
		}
		{
			int strt = row_id[start[thread_id + 1]];
			int num = block_start[thread_id + 1] - strt;
			mid_ans[thread_id * 2 + 1] = calculation_m4(strt, num, mtx_col, mtx_val, vec_val);
		}
	}
	else {
		{
			int strt = block_start[thread_id];
			int num = block_start[thread_id + 1] - strt;
			mid_ans[thread_id * 2] = calculation_m4(strt, num, mtx_col, mtx_val, vec_val);
			mid_ans[thread_id * 2 + 1] = 0; 
		}
	}
}

static inline void albus_thread_block_v_m8(double* mtx_val, int* mtx_col, int* row_id,
                                        double* vec_val, double* vec_res,
                                        int* start, int* block_start, int thread_id, double* mid_ans) {
	if (start[thread_id] < start[thread_id + 1]) {
		{
			int strt = block_start[thread_id];
			int num = row_id[start[thread_id] + 1] - strt;
			mid_ans[thread_id * 2] = calculation_m8(strt, num, mtx_col, mtx_val, vec_val);
		}
		for (int i = start[thread_id] + 1; i < start[thread_id + 1]; i++) {
			int strt = row_id[i];
			int num = row_id[i + 1] - strt;
			vec_res[i] = calculation_m8(strt, num, mtx_col, mtx_val, vec_val);
		}
		{
			int strt = row_id[start[thread_id + 1]];
			int num = block_start[thread_id + 1] - strt;
			mid_ans[thread_id * 2 + 1] = calculation_m8(strt, num, mtx_col, mtx_val, vec_val);
		}
	}
	else {
		{
			int strt = block_start[thread_id];
			int num = block_start[thread_id + 1] - strt;
			mid_ans[thread_id * 2] = calculation_m8(strt, num, mtx_col, mtx_val, vec_val);
			mid_ans[thread_id * 2 + 1] = 0; 
		}
	}
}

void spmv_albus_omp_v_noalloc_m1(const matrix_CSR<double>& mtx_CSR, const vector_format<double>& vec, int* start, int* block_start, int threads_num, vector_format<double>& res) {
	std::memset(res.value, 0, sizeof(double) * res.N);
	
	double *mid_ans = new double[threads_num * 2];
	
#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < threads_num; i++) {
        albus_thread_block_v_m1(mtx_CSR.value, mtx_CSR.col, mtx_CSR.row_id,
                           vec.value, res.value, start, block_start, i, mid_ans);
	}
	
	res.value[start[0]] = mid_ans[0];
	for (int i = 1; i < threads_num; i++) {
		res.value[start[i]] += mid_ans[i * 2 - 1] + mid_ans[i * 2];
	}
	
	delete[] mid_ans;
}

void spmv_albus_omp_v_noalloc_m2(const matrix_CSR<double>& mtx_CSR, const vector_format<double>& vec, int* start, int* block_start, int threads_num, vector_format<double>& res) {
	std::memset(res.value, 0, sizeof(double) * res.N);
	
	double *mid_ans = new double[threads_num * 2];
	
#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < threads_num; i++) {
        albus_thread_block_v_m2(mtx_CSR.value, mtx_CSR.col, mtx_CSR.row_id,
                           vec.value, res.value, start, block_start, i, mid_ans);
	}
	
	res.value[start[0]] = mid_ans[0];
	for (int i = 1; i < threads_num; i++) {
		res.value[start[i]] += mid_ans[i * 2 - 1] + mid_ans[i * 2];
	}
	
	delete[] mid_ans;
}

void spmv_albus_omp_v_noalloc_m4(const matrix_CSR<double>& mtx_CSR, const vector_format<double>& vec, int* start, int* block_start, int threads_num, vector_format<double>& res) {
	std::memset(res.value, 0, sizeof(double) * res.N);
	
	double *mid_ans = new double[threads_num * 2];
	
#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < threads_num; i++) {
        albus_thread_block_v_m4(mtx_CSR.value, mtx_CSR.col, mtx_CSR.row_id,
                           vec.value, res.value, start, block_start, i, mid_ans);
	}
	
	res.value[start[0]] = mid_ans[0];
	for (int i = 1; i < threads_num; i++) {
		res.value[start[i]] += mid_ans[i * 2 - 1] + mid_ans[i * 2];
	}
	
	delete[] mid_ans;
}

void spmv_albus_omp_v_noalloc_m8(const matrix_CSR<double>& mtx_CSR, const vector_format<double>& vec, int* start, int* block_start, int threads_num, vector_format<double>& res) {
	std::memset(res.value, 0, sizeof(double) * res.N);
	
	double *mid_ans = new double[threads_num * 2];
	
#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < threads_num; i++) {
        albus_thread_block_v_m8(mtx_CSR.value, mtx_CSR.col, mtx_CSR.row_id,
                           vec.value, res.value, start, block_start, i, mid_ans);
	}
	
	res.value[start[0]] = mid_ans[0];
	for (int i = 1; i < threads_num; i++) {
		res.value[start[i]] += mid_ans[i * 2 - 1] + mid_ans[i * 2];
	}
	
	delete[] mid_ans;
}

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
