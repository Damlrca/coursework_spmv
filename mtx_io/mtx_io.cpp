// Copyright (C) 2025 Sadikov Damir
// github.com/Damlrca/coursework_spmv

#include "mtx_io.hpp"

extern "C" {
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "mmio.h"
}

#include <iostream>
#include <fstream>

static void read_MTX_to_COO_internal(const char* fname, int* M_, int* N_, int* nz_, double** val_, int** I_, int** J_) {
    // based on mm_read_unsymmetric_sparse

    FILE* f;
    MM_typecode matcode;
    int M, N, nz;
    int i;
    double* val;
    int* I;
    int* J;

    if ((f = fopen(fname, "r")) == NULL) {
		std::cout << __func__ << ": Failed to read file " << fname << std::endl;
        throw -1;
    }

    if (mm_read_banner(f, &matcode) != 0) {
		std::cout << __func__ << ": Could not process Matrix Market banner in file " << fname << std::endl;
        throw -1;
    }

    if (!(/*mm_is_real(matcode) &&*/ mm_is_matrix(matcode) && mm_is_sparse(matcode)) || mm_is_complex(matcode)) {
		std::cout << __func__ << ": This application does not support Matrix Market type: " <<
			mm_typecode_to_str(matcode) << ", file: " << fname << std::endl;
        throw -1;
    }

    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
		std::cout << __func__ << ": Could not parse matrix size, file: " << fname << std::endl;
        throw -1;
    }

    if (mm_is_general(matcode)) { // matrix is unsymmetric
        I = new int[nz];
        J = new int[nz];
        val = new double[nz];

        for (i = 0; i < nz; ++i) {
			if (mm_is_real(matcode)) {
				if (fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]) != 3) {
					std::cout << __func__ << ": Failed to read matrix data, file: " << fname << std::endl;
					throw -1;
				}
			}
			else if (mm_is_pattern(matcode)) {
				if (fscanf(f, "%d %d\n", &I[i], &J[i]) != 2) {
					std::cout << __func__ << ": Failed to read matrix data, file: " << fname << std::endl;
					throw -1;
				}
				val[i] = 1;
			}
			else { // mm_is_integer(matcode)
				int temp = 0;
				if (fscanf(f, "%d %d %d\n", &I[i], &J[i], &temp) != 3) {
					std::cout << __func__ << ": Failed to read matrix data, file: " << fname << std::endl;
					throw -1;
				}
				val[i] = temp;
			}
            --I[i];
            --J[i];
        }

        *M_ = M;
        *N_ = N;
        *nz_ = nz;

        *val_ = val;
        *I_ = I;
        *J_ = J;
    }
    else { // matrix is symmetric
        int nnz = 0;
        int max_nz = nz * 2;
		
		I = new int[max_nz];
        J = new int[max_nz];
        val = new double[max_nz];
		
        for (i = 0; i < nz; ++i) {
            if (mm_is_real(matcode)) {
				if (fscanf(f, "%d %d %lg\n", &I[nnz], &J[nnz], &val[nnz]) != 3) {
					std::cout << __func__ << ": Failed to read matrix data, file: " << fname << std::endl;
					throw -1;
				}
			}
			else if (mm_is_pattern(matcode)) {
				if (fscanf(f, "%d %d\n", &I[nnz], &J[nnz]) != 2) {
					std::cout << __func__ << ": Failed to read matrix data, file: " << fname << std::endl;
					throw -1;
				}
				val[nnz] = 1;
			}
			else { // mm_is_integer(matcode)
				int temp = 0;
				if (fscanf(f, "%d %d %d\n", &I[nnz], &J[nnz], &temp) != 3) {
					std::cout << __func__ << ": Failed to read matrix data, file: " << fname << std::endl;
					throw -1;
				}
				val[nnz] = temp;
			}
            --I[nnz];
            --J[nnz];
            ++nnz;
            if (I[nnz - 1] != J[nnz - 1]) {
                I[nnz] = J[nnz - 1];
                J[nnz] = I[nnz - 1];
                val[nnz] = val[nnz - 1];
                ++nnz;
            }
        }

        *M_ = M;
        *N_ = N;
        *nz_ = nnz;

        *val_ = val;
        *I_ = I;
        *J_ = J;
    }

    fclose(f);
}

matrix_COO<double> read_MTX_to_COO(const char* fname) {
	matrix_COO<double> mtx_COO;
	read_MTX_to_COO_internal(fname, &mtx_COO.M, &mtx_COO.N, &mtx_COO.nz,
		&mtx_COO.val, &mtx_COO.I, &mtx_COO.J);
	return mtx_COO;
}

matrix_CSR<double> read_MTX_to_CSR(const char* fname) {
	matrix_COO<double> mtx_COO = read_MTX_to_COO(fname);
	matrix_CSR<double> mtx_CSR = convert_COO_to_CSR(mtx_COO);
	return mtx_CSR;
}

matrix_CSR<double> read_BIN_to_CSR(const char* fname) {
	matrix_CSR<double> mtx_CSR;
	std::ifstream fin(fname, std::ios::binary);
	if (!fin) {
		std::cout << "cannot open " << fname << " to read" << std::endl;
		exit(-1);
	}
	fin.read(reinterpret_cast<char*>(&mtx_CSR.N), sizeof(int));
	fin.read(reinterpret_cast<char*>(&mtx_CSR.M), sizeof(int));
	int nz = 0;
	fin.read(reinterpret_cast<char*>(&nz), sizeof(int));
	mtx_CSR.row_id = new int[mtx_CSR.N + 1];
	mtx_CSR.col = new int[nz];
    mtx_CSR.value = new double[nz];
	fin.read(reinterpret_cast<char*>(mtx_CSR.row_id), (mtx_CSR.N + 1) * sizeof(int));
	fin.read(reinterpret_cast<char*>(mtx_CSR.col), nz * sizeof(int));
	fin.read(reinterpret_cast<char*>(mtx_CSR.value), nz * sizeof(double));
	fin.close();
	return mtx_CSR;
}

void write_BIN_from_CSR(const matrix_CSR<double>& mtx, const char* fname) {
	std::ofstream fout(fname, std::ios::binary);
	if (!fout) {
		std::cout << "cannot open " << fname << " to write" << std::endl;
		exit(-1);
	}
	fout.write(reinterpret_cast<const char*>(&mtx.N), sizeof(int));
	fout.write(reinterpret_cast<const char*>(&mtx.M), sizeof(int));
	int nz = mtx.row_id[mtx.N];
	fout.write(reinterpret_cast<const char*>(&nz), sizeof(int));
	fout.write(reinterpret_cast<const char*>(mtx.row_id), (mtx.N + 1) * sizeof(int));
	fout.write(reinterpret_cast<const char*>(mtx.col), nz * sizeof(int));
	fout.write(reinterpret_cast<const char*>(mtx.value), nz * sizeof(double));
	fout.close();
}
