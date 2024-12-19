// Copyright (C) 2024 Sadikov Damir
// github.com/Damlrca/coursework_spmv

#include <riscv_vector.h>

#include <cstring>
#include <omp.h>

#include "spmv_functions.hpp"

vector_format alloc_vector_res(const matrix_CSR& mtx_CSR) {
	vector_format res;
	res.N = mtx_CSR.N;
	res.value = new double[res.N];
	std::memset(res.value, 0, sizeof(double) * res.N);
	return res;
}

vector_format alloc_vector_res(const matrix_SELL_C_sigma<8, 1>& mtx) {
	vector_format res;
	res.N = mtx.N;
	res.value = new double[res.N];
	std::memset(res.value, 0, sizeof(double) * res.N);
	return res;
}

vector_format alloc_vector_res(const matrix_SELL_C_sigma<4, 1>& mtx) {
	vector_format res;
	res.N = mtx.N;
	res.value = new double[res.N];
	std::memset(res.value, 0, sizeof(double) * res.N);
	return res;
}

void spmv_naive_noalloc(const matrix_CSR& mtx_CSR, const vector_format& vec, int threads_num, vector_format& res) {
#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < mtx_CSR.N; i++) {
		double temp = 0;
		for (int j = mtx_CSR.row_id[i]; j < mtx_CSR.row_id[i + 1]; j++) {
			temp += mtx_CSR.value[j] * vec.value[mtx_CSR.col[j]];
		}
		res.value[i] = temp;
	}
}

vector_format spmv_naive(const matrix_CSR& mtx_CSR, const vector_format& vec, int threads_num) {
	vector_format res = alloc_vector_res(mtx_CSR);
	spmv_naive_noalloc(mtx_CSR, vec, threads_num, res);
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

void spmv_albus_omp_noalloc(const matrix_CSR& mtx_CSR, const vector_format& vec, int* start, int* block_start, int threads_num, vector_format& res) {
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

vector_format spmv_albus_omp(const matrix_CSR& mtx_CSR, const vector_format& vec, int* start, int* block_start, int threads_num) {
	vector_format res = alloc_vector_res(mtx_CSR);
	spmv_albus_omp_noalloc(mtx_CSR, vec, start, block_start, threads_num, res);
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
	
	vfloat64m4_t v_summ = __riscv_vfmv_v_f_f64m4(0.0, vl);
	while (num > vl) {
		vuint32m2_t index = __riscv_vle32_v_u32m2(reinterpret_cast<uint32_t *>(col_idx + start1), vl);
		vuint32m2_t index_shftd = __riscv_vsll_vx_u32m2(index, 3, vl);
		vfloat64m4_t v_1 = __riscv_vle64_v_f64m4(mtx_val + start1, vl);
		//vfloat64m4_t v_2 = vluxei32_v_f64m4(vec_val, index_shftd, vl);
		vfloat64m4_t v_2 = __riscv_vloxei32_v_f64m4(vec_val, index_shftd, vl); // test vlox
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

void spmv_albus_omp_v_noalloc(const matrix_CSR& mtx_CSR, const vector_format& vec, int* start, int* block_start, int threads_num, vector_format& res) {
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

vector_format spmv_albus_omp_v(const matrix_CSR& mtx_CSR, const vector_format& vec, int* start, int* block_start, int threads_num) {
	vector_format res = alloc_vector_res(mtx_CSR);
	spmv_albus_omp_v_noalloc(mtx_CSR, vec, start, block_start, threads_num, res);
	return res;
}

void spmv_sell_c_sigma_noalloc(const matrix_SELL_C_sigma<8, 1>& mtx, const vector_format& vec, int threads_num, vector_format& res) {
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / 8; i++) {
		vfloat64m4_t v_summ = __riscv_vfmv_v_f_f64m4(0.0, 8);
		for (int j = mtx.cs[i]; j < mtx.cs[i + 1]; j += 8) {
			
			// index of column in bits
			// vuint32m2_t index = __riscv_vle32_v_u32m2(reinterpret_cast<uint32_t *>(mtx.col + j), 8);
			// vuint32m2_t index_shftd = __riscv_vsll_vx_u32m2(index, 3, 8);
			vuint32m2_t index_shftd = __riscv_vle32_v_u32m2(reinterpret_cast<uint32_t *>(mtx.col + j), 8);
			
			vfloat64m4_t v_1 = __riscv_vluxei32_v_f64m4(vec.value, index_shftd, 8);
			vfloat64m4_t v_2 = __riscv_vle64_v_f64m4(mtx.value + j, 8);
			v_summ = __riscv_vfmacc_vv_f64m4(v_summ, v_1, v_2, 8);
		}
		__riscv_vse64_v_f64m4(res.value + i * 8, v_summ, 8);
	}
}

vector_format spmv_sell_c_sigma(const matrix_SELL_C_sigma<8, 1>& mtx, const vector_format& vec, int threads_num) {
	vector_format res = alloc_vector_res(mtx);
	spmv_sell_c_sigma_noalloc(mtx, vec, threads_num, res);
	return res;
}

void spmv_sell_c_sigma_noalloc(const matrix_SELL_C_sigma<4, 1>& mtx, const vector_format& vec, int threads_num, vector_format& res) {
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / 4; i++) {
		vfloat64m2_t v_summ = __riscv_vfmv_v_f_f64m2(0.0, 4);
		for (int j = mtx.cs[i]; j < mtx.cs[i + 1]; j += 4) {
			
			// index of column in bits
			// vuint32m1_t index = __riscv_vle32_v_u32m1(reinterpret_cast<uint32_t *>(mtx.col + j), 4);
			// vuint32m1_t index_shftd = __riscv_vsll_vx_u32m1(index, 3, 4);
			vuint32m1_t index_shftd = __riscv_vle32_v_u32m1(reinterpret_cast<uint32_t *>(mtx.col + j), 4);
			
			vfloat64m2_t v_1 = __riscv_vluxei32_v_f64m2(vec.value, index_shftd, 4);
			vfloat64m2_t v_2 = __riscv_vle64_v_f64m2(mtx.value + j, 4);
			v_summ = __riscv_vfmacc_vv_f64m2(v_summ, v_1, v_2, 4);
		}
		__riscv_vse64_v_f64m2(res.value + i * 4, v_summ, 4);
	}
}

vector_format spmv_sell_c_sigma(const matrix_SELL_C_sigma<4, 1>& mtx, const vector_format& vec, int threads_num) {
	vector_format res = alloc_vector_res(mtx);
	spmv_sell_c_sigma_noalloc(mtx, vec, threads_num, res);
	return res;
}


void spmv_sell_c_sigma_noalloc_novec(const matrix_SELL_C_sigma<8, 1>& mtx, const vector_format& vec, int threads_num, vector_format& res) {
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / 8; i++) {
		//vfloat64m4_t v_summ = __riscv_vfmv_v_f_f64m4(0.0, 8);
		double v_summ[8]{};
		for (int j = mtx.cs[i]; j < mtx.cs[i + 1]; j += 8) {
			//vuint32m2_t index_shftd = __riscv_vle32_v_u32m2(reinterpret_cast<uint32_t *>(mtx.col + j), 8);
			//vfloat64m4_t v_1 = __riscv_vluxei32_v_f64m4(vec.value, index_shftd, 8);
			//vfloat64m4_t v_2 = __riscv_vle64_v_f64m4(mtx.value + j, 8);
			//v_summ = __riscv_vfmacc_vv_f64m4(v_summ, v_1, v_2, 8);
			for (int k = 0; k < 8; k++) {
				v_summ[k] += vec.value[*(mtx.col + j + k) >> 3] * mtx.value[j + k];
			}
		}
		//__riscv_vse64_v_f64m4(res.value + i * 8, v_summ, 8);
		for (int k = 0; k < 8; k++) {
			res.value[i * 8 + k] = v_summ[k];
		}
	}
}

void spmv_sell_c_sigma_noalloc_novec(const matrix_SELL_C_sigma<4, 1>& mtx, const vector_format& vec, int threads_num, vector_format& res) {
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / 4; i++) {
		//vfloat64m2_t v_summ = __riscv_vfmv_v_f_f64m2(0.0, 4);
		double v_summ[4]{};
		for (int j = mtx.cs[i]; j < mtx.cs[i + 1]; j += 4) {
			//vuint32m1_t index_shftd = __riscv_vle32_v_u32m1(reinterpret_cast<uint32_t *>(mtx.col + j), 4);
			//vfloat64m2_t v_1 = __riscv_vluxei32_v_f64m2(vec.value, index_shftd, 4);
			//vfloat64m2_t v_2 = __riscv_vle64_v_f64m2(mtx.value + j, 4);
			//v_summ = __riscv_vfmacc_vv_f64m2(v_summ, v_1, v_2, 4);
			for (int k = 0; k < 4; k++) {
				v_summ[k] += vec.value[*(mtx.col + j + k) >> 3] * mtx.value[j + k];
			}
		}
		//__riscv_vse64_v_f64m2(res.value + i * 4, v_summ, 4);
		for (int k = 0; k < 4; k++) {
			res.value[i * 4 + k] = v_summ[k];
		}
	}
}
