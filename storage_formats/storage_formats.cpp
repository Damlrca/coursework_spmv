// Copyright (C) 2025 Sadikov Damir
// github.com/Damlrca/coursework_spmv

#include "storage_formats.hpp"
#include <algorithm>

extern "C" {
#include <string.h>
}

matrix_CSR convert_COO_to_CSR(const matrix_COO& mtx_COO) {
	int N = mtx_COO.N;
	int M = mtx_COO.M;
	int nz = mtx_COO.nz;
	double* val = mtx_COO.val;
	int* I = mtx_COO.I;
	int* J = mtx_COO.J;
	
	int* row_id = new int[N + 1];
	int* col = new int[nz];
	double* value = new double[nz];
	int S = 0, pr = 0;
	memset(row_id, 0, (N + 1) * sizeof(int));
	for (int i = 0; i < nz; ++i) {
		++row_id[I[i] + 1];
	}
	for (int i = 1; i < N + 1; ++i) {
		S += pr;
		pr = row_id[i];
		row_id[i] = S;
	}
	for (int i = 0; i < nz; ++i) {
		int RIndex = row_id[I[i] + 1];
		col[RIndex] = J[i];
		value[RIndex] = val[i];
		++row_id[I[i] + 1];
	}

	matrix_CSR mtx_CSR;
	mtx_CSR.N = mtx_COO.N;
	mtx_CSR.M = mtx_COO.M;
	mtx_CSR.row_id = row_id;
	mtx_CSR.col = col;
	mtx_CSR.value = value;
	
	return mtx_CSR;
}

static matrix_CSR create_transposed_CSR(matrix_CSR& mtx_CSR) {
	matrix_CSR res;
	int nz = mtx_CSR.row_id[mtx_CSR.N];
	res.N = mtx_CSR.M;
	res.M = mtx_CSR.N;
	res.row_id = new int[res.N + 1];
	res.col = new int[nz];
	res.value = new double[nz];
	int S = 0, pr = 0;
	memset(res.row_id, 0, (res.N + 1) * sizeof(int));
	for (int i = 0; i < nz; i++) {
		++res.row_id[mtx_CSR.col[i] + 1];
	}
	for (int i = 1; i < res.N + 1; i++) {
		S += pr;
		pr = res.row_id[i];
		res.row_id[i] = S;
	}
	for (int i = 0; i < mtx_CSR.N; i++) {
		int a = mtx_CSR.row_id[i];
		int b = mtx_CSR.row_id[i + 1];
		while (a != b) {
			int RIndex = res.row_id[mtx_CSR.col[a] + 1];
			res.col[RIndex] = i;
			res.value[RIndex] = mtx_CSR.value[a];
			++res.row_id[mtx_CSR.col[a] + 1];
			++a;
		}
	}
	return res;
}

void transpose_CSR(matrix_CSR& mtx_CSR) {
	mtx_CSR = std::move(create_transposed_CSR(mtx_CSR));
}
