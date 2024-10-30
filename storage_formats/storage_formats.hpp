// Copyright (C) 2024 Sadikov Damir
// github.com/Damlrca/coursework_spmv

#ifndef STORAGE_FORMATS_HPP
#define STORAGE_FORMATS_HPP

#include <utility>

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

struct vector_format {
	int N = 0;
	double* value = nullptr;
	vector_format() {}
	vector_format(vector_format&& vec) {
		*this = std::move(vec);
	}
	vector_format& operator=(vector_format&& vec) {
		std::swap(N, vec.N);
		std::swap(value, vec.value);
		return *this;
	}
	~vector_format() {
		delete[] value;
	}
};

matrix_CSR convert_COO_to_CSR(const matrix_COO& mtx_COO);

void transpose_CSR(matrix_CSR& mtx_CSR);

#endif // !STORAGE_FORMATS_HPP
