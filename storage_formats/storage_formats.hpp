// Copyright (C) 2025 Sadikov Damir
// github.com/Damlrca/coursework_spmv

#ifndef STORAGE_FORMATS_HPP
#define STORAGE_FORMATS_HPP

#include <utility>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <memory>
#include <vector>

// COO - Coordinate list sparse matrix format
template <typename T = double>
struct matrix_COO {
	int N = 0;
	int M = 0;
	int nz = 0;
	T* val = nullptr;
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
template <typename T = double>
struct matrix_CSR {
	int N = 0;
	int M = 0;
	int* row_id = nullptr;
	int* col = nullptr;
	T* value = nullptr;
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

matrix_CSR<double> convert_COO_to_CSR(const matrix_COO<double>& mtx_COO);

void transpose_CSR(matrix_CSR<double>& mtx_CSR);

// struct for storing a vector
template <typename T = double>
struct vector_format {
	int N = 0;
	T* value = nullptr;
	T* value_buf = nullptr; // buffer for value (used for align memory)
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
		value_buf = new T[N + C];
		std::size_t value_buf_size = (N + C) * sizeof(T);
		void* temp_value_buf = (void*)value_buf;
		value = (T*)std::align(C * sizeof(T), N * sizeof(T), temp_value_buf, value_buf_size);
		this->N = N;
	}
};

// SELL_C_sigma sparse matrix format
// C - number of rows in block
// sigma - number of consecutive blocks in which rows are sorted 
//         in descending order of the number of non-zero elements
// if sigma is 1 then sorting is not applied
template<int C, int sigma, typename T = double>
struct matrix_SELL_C_sigma {
	int N = 0;
	int M = 0;
	T* value = nullptr;
	int* col = nullptr;
	T* value_buf = nullptr; // buffer for value (used for align memory)
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

template<int C, int sigma, typename T>
matrix_SELL_C_sigma<C, sigma, T> convert_CSR_to_SELL_C_sigma(const matrix_CSR<T>& mtx_CSR, int vertical_block_size = 9'999'999) {
	matrix_SELL_C_sigma<C, sigma, T> res;
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
	
	std::vector<std::vector<std::pair<int, T>>> TEMP(res.N);
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
					int k_block = mtx_CSR.col[j] / vertical_block_size;
					if (block == -1 || k_block < block) {
						block = k_block;
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
					int k_block = mtx_CSR.col[j] / vertical_block_size;
					if (k_block == block) {
						TEMP[i + k].push_back({mtx_CSR.col[j], mtx_CSR.value[j]});
						i_ids[k]++;
					}
					else {
						TEMP[i + k].push_back({-1, 0});
					}
				}
				else {
					TEMP[i + k].push_back({-1, 0});
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
	res.value_buf = new T[S + C];
	std::size_t value_buf_size = (S + C) * sizeof(T);
	void* temp_value_buf = (void*)res.value_buf;
	res.value = (T*)std::align(C * sizeof(T), S * sizeof(T), temp_value_buf, value_buf_size);
	res.col_buf = new int[S + C];
	std::size_t col_buf_size = (S + C) * sizeof(int);
	void* temp_col_buf = (void*)res.col_buf;
	res.col = (int*)std::align(C * sizeof(int), S * sizeof(int), temp_col_buf, col_buf_size);
	memset(res.value, 0, S * sizeof(T));
	for (int i = 0; i < res.N; i++) {
		for (int j = 0; j < TEMP[i].size(); j++) {
			int indx = res.cs[i / C] + (i % C) + (j * C);
			res.value[indx] = TEMP[i][j].second;
			// index of column in bits (for riscv vlux intrinsic) // OR IN BYTES???
			if (TEMP[i][j].first == -1)
				res.col[indx] = -1;
			else
				res.col[indx] = TEMP[i][j].first * sizeof(T);
		}
	}
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
	return res;
}

#endif // !STORAGE_FORMATS_HPP
