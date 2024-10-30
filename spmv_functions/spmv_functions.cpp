// Copyright (C) 2024 Sadikov Damir
// github.com/Damlrca/coursework_spmv

#include <riscv_vector.h>

#include <cstring>
#include <omp.h>

#include "spmv_functions.hpp"
vector_format spmv_naive(const matrix_CSR& mtx_CSR, const vector_format& vec, int threads_num) {
	vector_format res;
	res.N = mtx_CSR.N;
	res.value = new double[res.N];
	std::memset(res.value, 0, sizeof(double) * res.N);
	
#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < mtx_CSR.N; i++) {
		double temp = 0;
		for (int j = mtx_CSR.row_id[i]; j < mtx_CSR.row_id[i + 1]; j++) {
			temp += mtx_CSR.value[j] * vec.value[mtx_CSR.col[j]];
		}
		res.value[i] = temp;
	}
	return res;
}

static inline int preproc_albus_binary_search(int *row_id, int block_start_indx, int N) {
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

void preproc_albus_balance(const matrix_CSR& mtx_CSR, int* start, int* block_start, int threads_num) {
	const int n = mtx_CSR.N;
	const int nz = mtx_CSR.row_id[n];
	const int T = nz / threads_num;
	start[threads_num] = n;
	block_start[threads_num] = nz;
	for (int i = 0; i < threads_num; i++) {
		block_start[i] = i * T;
		start[i] = preproc_albus_binary_search(mtx_CSR.row_id, block_start[i], n);
	}
}

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

vector_format spmv_albus_omp(const matrix_CSR& mtx_CSR, const vector_format& vec, int* start, int* block_start, int threads_num) {
	vector_format res;
	res.N = mtx_CSR.N;
	res.value = new double[res.N];
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
		
	return res;
}

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
	constexpr size_t vl = 8;
	int end1 = start1 + num;
	double answer = 0;
	
	vfloat64m4_t v_summ = vfmv_v_f_f64m4(0.0, vl);
	while (num > vl) {
		vuint32m2_t index = vle32_v_u32m2(reinterpret_cast<uint32_t *>(col_idx + start1), vl);
		vuint32m2_t index_shftd = vsll_vx_u32m2(index, 3, vl);
		vfloat64m4_t v_1 = vle64_v_f64m4(mtx_val + start1, vl);
		//vfloat64m4_t v_2 = vluxei32_v_f64m4(vec_val, index_shftd, vl);
		vfloat64m4_t v_2 = vloxei32_v_f64m4(vec_val, index_shftd, vl); // test vlox
		v_summ = vfmacc_vv_f64m4(v_summ, v_1, v_2, vl);
		start1 += vl;
		num -= vl;
	}
	vfloat64m1_t v_res = vfmv_v_f_f64m1(0.0, vsetvlmax_e64m1());
	v_res = vfredosum_vs_f64m4_f64m1(v_res, v_summ, v_res, vl);
	vse64_v_f64m1(&answer, v_res, 1);
	while (start1 < end1) {
		answer += mtx_val[start1] * vec_val[col_idx[start1]];
		start1++;
	}
	return answer;
}

static inline double calculation(int start1, int num, int* col_idx, double * mtx_val, double* vec_val) {
	// vsetvlmax_e64m4() == 8
	if (num >= 8)
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

vector_format spmv_albus_omp_v(const matrix_CSR& mtx_CSR, const vector_format& vec, int* start, int* block_start, int threads_num) {
	vector_format res;
	res.N = mtx_CSR.N;
	res.value = new double[res.N];
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
		
	return res;
}
