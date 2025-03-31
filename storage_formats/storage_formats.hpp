// Copyright (C) 2025 Sadikov Damir
// github.com/Damlrca/coursework_spmv

#ifndef STORAGE_FORMATS_HPP
#define STORAGE_FORMATS_HPP

#include <utility>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <memory>
#include <iostream>

// COO - Coordinate list sparse matrix format
struct matrix_COO {
	int N = 0;
	int M = 0;
	int nz = 0;
	double* val = nullptr;
	int* I = nullptr;
	int* J = nullptr;
	matrix_COO() {}
	matrix_COO(matrix_COO&& mtx) {
		*this = std::move(mtx);
	}
	matrix_COO& operator=(matrix_COO&& mtx) {
		std::swap(N, mtx.N);
		std::swap(M, mtx.M);
		std::swap(nz, mtx.nz);
		std::swap(val, mtx.val);
		std::swap(I, mtx.I);
		std::swap(J, mtx.J);
		return *this;
	}
	~matrix_COO() {
		delete[] val;
		delete[] I;
		delete[] J;
	}
};

// CSR - Compressed sparse row sparse matrix format (CRS - Compressed row storage)
struct matrix_CSR {
	int N = 0;
	int M = 0;
	int* row_id = nullptr;
	int* col = nullptr;
	double* value = nullptr;
	matrix_CSR() {}
	matrix_CSR(matrix_CSR&& mtx) {
		*this = std::move(mtx);
	}
	matrix_CSR& operator=(matrix_CSR&& mtx) {
		std::swap(N, mtx.N);
		std::swap(M, mtx.M);
		std::swap(row_id, mtx.row_id);
		std::swap(col, mtx.col);
		std::swap(value, mtx.value);
		return *this;
	}
	~matrix_CSR() {
		delete[] row_id;
		delete[] col;
		delete[] value;
	}
};

matrix_CSR convert_COO_to_CSR(const matrix_COO& mtx_COO);

void transpose_CSR(matrix_CSR& mtx_CSR);

// struct for storing a vector
struct vector_format {
	int N = 0;
	double* value = nullptr;
	double* value_buf = nullptr; // buffer for value (used for align memory)
	vector_format() {}
	vector_format(vector_format&& vec) {
		*this = std::move(vec);
	}
	vector_format& operator=(vector_format&& vec) {
		std::swap(N, vec.N);
		std::swap(value, vec.value);
		std::swap(value_buf, vec.value_buf);
		return *this;
	}
	~vector_format() {
		delete[] value_buf;
	}
	void alloc(int N, int C) {
		if (value_buf != nullptr) {
			delete[] value_buf;
			value = nullptr;
			value_buf = nullptr;
			this->N = 0;
		}
		value_buf = new double[N + C];
		std::size_t value_buf_size = (N + C) * sizeof(double);
		void* temp_value_buf = (void*)value_buf;
		value = (double*)std::align(C * sizeof(double), N * sizeof(double), temp_value_buf, value_buf_size);
		this->N = N;
	}
};

// SELL_C_sigma sparse matrix format
// C - number of rows in block
// sigma - number of consecutive blocks in which rows are sorted 
//         in descending order of the number of non-zero elements
// if sigma is 1 then sorting is not applied
template<int C, int sigma>
struct matrix_SELL_C_sigma {
	int N = 0;
	int M = 0;
	double* value = nullptr;
	int* col = nullptr;
	double* value_buf = nullptr; // buffer for value (used for align memory)
	int* col_buf = nullptr; // buffer for col (used for align memory)
	int* cs = nullptr;
	int* cl = nullptr;
	int* rows_perm = nullptr; // permutation of rows
	matrix_SELL_C_sigma() {}
	matrix_SELL_C_sigma(matrix_SELL_C_sigma&& mtx) {
		*this = std::move(mtx);
	}
	matrix_SELL_C_sigma& operator=(matrix_SELL_C_sigma&& mtx) {
		std::swap(N, mtx.N);
		std::swap(M, mtx.M);
		std::swap(value, mtx.value);
		std::swap(col, mtx.col);
		std::swap(value_buf, mtx.value_buf);
		std::swap(col_buf, mtx.col_buf);
		std::swap(cs, mtx.cs);
		std::swap(cl, mtx.cl);
		std::swap(rows_perm, mtx.rows_perm);
		return *this;
	}
	~matrix_SELL_C_sigma() {
		delete[] value_buf;
		delete[] col_buf;
		delete[] cs;
		delete[] cl;
		delete[] rows_perm;
	}
};

template<int C, int sigma>
matrix_SELL_C_sigma<C, sigma> convert_CSR_to_SELL_C_sigma(const matrix_CSR& mtx_CSR, bool fill_null_elements = true) {
	constexpr int vertical_block_size = 1024;
	matrix_SELL_C_sigma<C, sigma> res;
	res.N = (mtx_CSR.N + C - 1) / C * C;
	res.M = mtx_CSR.M;
	res.rows_perm = new int[res.N];
	std::iota(res.rows_perm, res.rows_perm + res.N, 0);
	if (sigma != 1) {
		for (int i = 0; i < res.N; i += C * sigma) {
			std::sort(res.rows_perm + i, res.rows_perm + std::min(res.N, i + C * sigma), [&](int a, int b)->bool{
				int sz_a = 0;
				int sz_b = 0;
				if (a < mtx_CSR.N) sz_a = mtx_CSR.row_id[a + 1] - mtx_CSR.row_id[a];
				if (b < mtx_CSR.N) sz_b = mtx_CSR.row_id[b + 1] - mtx_CSR.row_id[b];
				if (sz_a != sz_b)
					return sz_a > sz_b;
				return a < b;
			});
		}
	}
	res.cs = new int[res.N / C + 1];
	res.cl = new int[res.N / C];
	memset(res.cs, 0, (res.N / C + 1) * sizeof(int));
	memset(res.cl, 0, (res.N / C) * sizeof(int));
	std::vector<std::vector<std::pair<int, double>>> TEMP(res.N);
	for (int i = 0; i < res.N; i += C) {
		std::vector<int> i_ids(C), i_sz(C);
		for (int k = 0; k < C; k++) {
			int ik_id = res.rows_perm[i + k];
			if (ik_id < mtx_CSR.N) {
				i_sz[k] = mtx_CSR.row_id[ik_id + 1] - mtx_CSR.row_id[ik_id];
			}
		}
		while (true) {
			int block = -1;
			for (int k = 0; k < C; k++) {
				if (i_ids[k] < i_sz[k]) {
					int ik_id = res.rows_perm[i + k];
					int j = mtx_CSR.row_id[ik_id] + i_ids[k];
					if (block == -1 || mtx_CSR.col[j] / vertical_block_size < block) {
						block = mtx_CSR.col[j] / vertical_block_size;
					}
				}
			}
			if (block == -1) {
				break;
			}
			for (int k = 0; k < C; k++) {
				if (i_ids[k] < i_sz[k]) {
					int ik_id = res.rows_perm[i + k];
					int j = mtx_CSR.row_id[ik_id] + i_ids[k];
					if (mtx_CSR.col[j] / vertical_block_size == block) {
						TEMP[i + k].push_back({mtx_CSR.col[j], mtx_CSR.value[j]});
						i_ids[k]++;
					}
					else {
						TEMP[i + k].push_back({-1,0});
					}
				}
				else {
					TEMP[i + k].push_back({-1,0});
				}
			}
		}
	}
	int S = 0;
	for (int i = 0; i < res.N / C; i++) {
		res.cs[i] = S;
		res.cl[i] = TEMP[i * C].size();
		S += res.cl[i] * C;
	}
	res.cs[res.N / C] = S;
	res.value_buf = new double[S + C];
	std::size_t value_buf_size = (S + C) * sizeof(double);
	void* temp_value_buf = (void*)res.value_buf;
	res.value = (double*)std::align(C * sizeof(double), S * sizeof(double), temp_value_buf, value_buf_size);
	res.col_buf = new int[S + C];
	std::size_t col_buf_size = (S + C) * sizeof(int);
	void* temp_col_buf = (void*)res.col_buf;
	res.col = (int*)std::align(C * sizeof(int), S * sizeof(int), temp_col_buf, col_buf_size);
	memset(res.value, 0, S * sizeof(double));
	if (fill_null_elements) {
		memset(res.col, -1, S * sizeof(int));
	}
	else {
		memset(res.col, 0, S * sizeof(int));
	}
	for (int i = 0; i < res.N; i++) {
		for (int j = 0; j < TEMP[i].size(); j++) {
			int indx = res.cs[i / C] + (i % C) + (j * C);
			res.value[indx] = TEMP[i][j].second;
			// index of column in bits (for riscv vlux intrinsic)
			if (TEMP[i][j].first == -1)
				res.col[indx] = -1;
			else
				res.col[indx] = TEMP[i][j].first * 8;
		}
	}
	if (fill_null_elements) {
		for (int i = 0; i < res.N / C; i++) {
			for (int j = res.cs[i]; j < res.cs[i + 1]; j += C) {
				int id = -1;
				for (int k = 0; k < C; k++) {
					if (res.col[j + k] != -1) {
						id = res.col[j + k];
						break;
					}
				}
				for (int k = 0; k < C; k++) {
					if (res.col[j + k] == -1) {
						res.col[j + k] = id;
					}
				}
			}
		}
	}
	return res;
}

#endif // !STORAGE_FORMATS_HPP
