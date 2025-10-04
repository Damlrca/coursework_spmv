// Copyright (C) 2025 Sadikov Damir
// github.com/Damlrca/coursework_spmv

#include <iostream>

#include "mtx_io.hpp"
#include "../storage_formats/storage_formats.hpp"

void mtx2bin(const char* mtx, const char* bin) {
	matrix_CSR<double> mtx_CSR = read_MTX_to_CSR(mtx);
	std::cout << mtx_CSR.N << " " << mtx_CSR.M << " " << mtx_CSR.row_id[mtx_CSR.N] << std::endl;
	write_BIN_from_CSR(mtx_CSR, bin);
}

void check(const char* mtx, const char* bin) {
	matrix_CSR<double> mtx_1 = read_MTX_to_CSR(mtx);
	matrix_CSR<double> mtx_2 = read_BIN_to_CSR(bin);
	if (mtx_1.N != mtx_2.N) {
		std::cout << "mtx_1.N != mtx_2.N :" << std::endl;
		std::cout << "mtx_1.N = " << mtx_1.N << std::endl;
		std::cout << "mtx_2.N = " << mtx_2.N << std::endl;
		exit(-1);
	}
	if (mtx_1.M != mtx_2.M) {
		std::cout << "mtx_1.M != mtx_2.M :" << std::endl;
		std::cout << "mtx_1.M = " << mtx_1.M << std::endl;
		std::cout << "mtx_2.M = " << mtx_2.M << std::endl;
		exit(-1);
	}
	for (int i = 0; i <= mtx_1.N; i++) {
		if (mtx_1.row_id[i] != mtx_2.row_id[i]) {
			std::cout << "mtx_1.row_id[i] != mtx_2.row_id[i] :" << std::endl;
			std::cout << "i = " << i << std::endl;
			std::cout << "mtx_1.row_id[i] = " << mtx_1.row_id[i] << std::endl;
			std::cout << "mtx_2.row_id[i] = " << mtx_2.row_id[i] << std::endl;
			exit(-1);
		}
	}
	for (int j = mtx_1.row_id[0]; j < mtx_1.row_id[mtx_1.N]; j++) {
		if (mtx_1.col[j] != mtx_2.col[j]) {
			std::cout << "mtx_1.col[j] != mtx_2.col[j] :" << std::endl;
			std::cout << "j = " << j << std::endl;
			std::cout << "mtx_1.col[j] = " << mtx_1.col[j] << std::endl;
			std::cout << "mtx_2.col[j] = " << mtx_2.col[j] << std::endl;
			exit(-1);
		}
		if (mtx_1.value[j] != mtx_2.value[j]) {
			std::cout << "mtx_1.value[j] != mtx_2.value[j] :" << std::endl;
			std::cout << "j = " << j << std::endl;
			std::cout << "mtx_1.value[j] = " << mtx_1.value[j] << std::endl;
			std::cout << "mtx_2.value[j] = " << mtx_2.value[j] << std::endl;
			exit(-1);
		}
	}
}

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cout << "error : no names input/output matrices" << std::endl;
		return -1;
	}
	// argv[1] - input matrix in MTX format
	auto input_mtx = argv[1];
	// argv[2] - output matrix in BIN format
	auto output_bin = argv[2];
	std::cout << input_mtx << " -> " << output_bin << std::endl;
	mtx2bin(input_mtx, output_bin);
	check(input_mtx, output_bin);
	std::cout << "ok!" << std::endl;
	return 0;
}
