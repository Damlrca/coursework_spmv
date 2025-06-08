// Copyright (C) 2025 Sadikov Damir
// github.com/Damlrca/coursework_spmv

#ifndef MTX_INPUT_HPP
#define MTX_INPUT_HPP

#include "../storage_formats/storage_formats.hpp"

matrix_COO<double> read_MTX_as_COO(const char* fname);

matrix_CSR<double> read_MTX_as_CSR(const char* fname);

#endif // !MTX_INPUT_HPP
