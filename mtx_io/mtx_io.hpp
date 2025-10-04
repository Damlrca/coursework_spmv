// Copyright (C) 2025 Sadikov Damir
// github.com/Damlrca/coursework_spmv

#ifndef MTX_IO_HPP
#define MTX_IO_HPP

#include "../storage_formats/storage_formats.hpp"

// MTX

matrix_COO<double> read_MTX_to_COO(const char* fname);

matrix_CSR<double> read_MTX_to_CSR(const char* fname);

// BIN

matrix_CSR<double> read_BIN_to_CSR(const char* fname);

void write_BIN_from_CSR(const matrix_CSR<double>& mtx, const char* fname);

#endif // !MTX_IO_HPP
