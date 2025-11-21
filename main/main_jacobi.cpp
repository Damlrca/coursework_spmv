// Copyright (C) 2025 Sadikov Damir
// github.com/Damlrca/coursework_spmv

#include <iostream>
#include <chrono>
#include <set>
#include <random>
#include <omp.h>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <cassert>
#include <vector>

#include "../storage_formats/storage_formats.hpp"
#include "../mtx_io/mtx_io.hpp"
#include "../spmv_functions/spmv_functions.hpp"

using namespace std;

class MyTimer{
	using myclock = std::chrono::high_resolution_clock;
	myclock::time_point start_time;
	myclock::time_point end_time;
public:
	void SetStartTime() {
		start_time = myclock::now();
	}
	void SetEndTime() {
		end_time = myclock::now();
	}
	// difference in MICROSECONDS (us)
	long long GetDifferenceUs() {
		return std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
	}
};

// time in microseconds to string of time in ms
string us_to_ms_string(long long time) {
	string s = to_string(time);
	if (s.size() < 4) {
		s = string(4 - s.size(), '0') + s;
	}
	string res = s.substr(0, s.size() - 3) + '.' + s.substr(s.size() - 3, 3);
	return res;
}

// calc max absolute difference of elements of two vectors
template <typename T>
T calc_diff(const vector_format<T>& a, const vector_format<T>& b) {
	assert(b.N >= a.N);
	assert(a.N > 0);
	T ans = 0;
	for (int i = 0; i < a.N; i++) {
		ans = max(ans, abs(a.value[i] - b.value[i]));
	}
	return ans;
}

template<typename T>
string name_suffix();

template<>
string name_suffix<double>() {
	return "";
}

template<>
string name_suffix<float>() {
	return "_f";
}

template<typename T>
string type_name();

template<>
string type_name<double>() {
	return "double";
}

template<>
string type_name<float>() {
	return "float";
}

const string NONE = "---";

struct JacobiResult{
	string NAME = NONE;         // short name for algorithm
	string FUNCTION = NONE;     // full function name
	string TIME_MS = NONE;      // time in milliseconds
	string DIFF = NONE;         // difference in result between naive and this approach
	string NZ = NONE;           // number of non-zero elements in this format
	string CONV_TIME = NONE;    // time of convertion to format
	string PREPROC_TIME = NONE; // time of preprocessing
};

vector_format<double> b, naive_res;
vector_format<float> b_float, naive_res_float;
constexpr int WARM_UP_ITE = 2; // number of warm up runs before actual measurement

// test_jacobi_naive double
TestResult test_jacobi_naive(const int ite, const int threads_num, const matrix_CSR<double>& mtx_CSR) {
	MyTimer timer;
	const string name = "naive";
	const string function = "spmv_naive_noalloc<double>";
	cout << "run " << name << " " << function << endl;
	
	naive_res = move(alloc_vector_res(mtx_CSR));
	
	for (int it = 0; it < WARM_UP_ITE; it++) {
		timer.SetStartTime();
		spmv_naive_noalloc(mtx_CSR, v, threads_num, naive_res);
		timer.SetEndTime();
	}
	long long res = numeric_limits<long long>::max();
	for (int it = 0; it < ite; it++) {
		timer.SetStartTime();
		spmv_naive_noalloc(mtx_CSR, v, threads_num, naive_res);
		timer.SetEndTime();
		res = min(res, timer.GetDifferenceUs());
	}

	TestResult test_result;
	test_result.NAME = name;
	test_result.FUNCTION = function;
	test_result.TIME_MS = us_to_ms_string(res);
	test_result.NZ = to_string(mtx_CSR.row_id[mtx_CSR.N]);
	return test_result;
}

// test_jacobi_naive float
TestResult test_jacobi_naive(const int ite, const int threads_num, const matrix_CSR<float>& mtx_CSR) {
	MyTimer timer;
	const string name = "naive_f";
	const string function = "spmv_naive_noalloc<float>";
	cout << "run " << name << " " << function << endl;
	
	naive_res_float = move(alloc_vector_res(mtx_CSR));
	
	for (int it = 0; it < WARM_UP_ITE; it++) {
		timer.SetStartTime();
		spmv_naive_noalloc(mtx_CSR, v_float, threads_num, naive_res_float);
		timer.SetEndTime();
	}
	long long res = numeric_limits<long long>::max();
	for (int it = 0; it < ite; it++) {
		timer.SetStartTime();
		spmv_naive_noalloc(mtx_CSR, v_float, threads_num, naive_res_float);
		timer.SetEndTime();
		res = min(res, timer.GetDifferenceUs());
	}
	
	TestResult test_result;
	test_result.NAME = name;
	test_result.FUNCTION = function;
	test_result.TIME_MS = us_to_ms_string(res);
	test_result.NZ = to_string(mtx_CSR.row_id[mtx_CSR.N]);
	return test_result;
}

constexpr int SIGMA_SORTED = 9'999'999;

const double? eps = ???;

// jacobi_sell_c_sigma
template<typename T, int C, int sigma>
void jacobi_sell_c_sigma(const int max_ite, const int threads_num, const vector_format<T>& LEFT, const matrix_SELL_C_sigma<C, sigma, T>& RIGHT_MTX, vector_format<T>& RES, vector_format<T>& PREV_RES, vector_format<T>& MULT) {
	// TODO INIT PREVRES
	int it = 0;
	do {
		it++;
		spmv_sell_c_sigma_noalloc(mtx, prev_x, threads_num, MULT);
		res = local_b - MULT;
		if (DIST(res - prev_x) < eps) {
			break;
		}
		prev_x = res;
	} while (it < ite);
}

// test_jacobi_sell_c_sigma
template<typename T, int C, int sigma>
TestResult test_jacobi_sell_c_sigma(const int ite, const int threads_num, const matrix_CSR<T>& mtx_CSR) {
	MyTimer timer;
	const string name = "scs_" + to_string(C) + "_" + (sigma == SIGMA_SORTED ? string("S") : to_string(sigma)) + name_suffix<T>();
	const string function = "spmv_sell_c_sigma_noalloc<" + type_name<T>() + "," + to_string(C) + "," + to_string(sigma) + ">";
	cout << "run " << name << " " << function << endl;
	
	timer.SetStartTime();
	matrix_SELL_C_sigma<C, sigma, T> mtx = convert_CSR_to_SELL_C_sigma<C, sigma>(mtx_CSR);
	timer.SetEndTime();
	auto conv_time = timer.GetDifferenceUs();
	
	timer.SetStartTime();
	vector_format<T> local_b = alloc_vector_res(mtx);
	for (int i = 0; i < b.N; i++) {
		local_b[i] = b[mtx.rows_perm[i]];
	}
	vector<T> diag(mtx.N);
	for (int i = 0; i < mtx.N / C; i++) {
		for (int j = mtx.cs[i]; j < mtx.cs[i + 1]; j += C) {
			for (int k = 0; k < C; k++) {
				if (mtx.col[j + k] == i + k) {
					diag[i + k] = mtx.val[j + k];
					mtx.val[j + k] = 0;
				}
			}
		}
	}
	for (int i = 0; i < mtx.N / C; i++) {
		for (int j = mtx.cs[i]; j < mtx.cs[i + 1]; j += C) {
			for (int k = 0; k < C; k++) {
				if (mtx.col[j + k] != i + k) {
					if (diag[i + k] != 0) {
						mtx.val[j + k] /= diag[i + k];
					}
				}
			}
		}
	}
	for (int i = 0; i < local_b.N; i++) {
		if (diag[i] != 0) {
			local_b[i] /= diag[i];
		}
	}
	timer.SetEndTime();
	auto preproc_time = timer.GetDifferenceUs();
	
	vector_format<T> scs_res = alloc_vector_res(mtx);
	
	for (int it = 0; it < WARM_UP_ITE; it++) {
		timer.SetStartTime();
		spmv_sell_c_sigma_noalloc(mtx, v, threads_num, scs_res);
		timer.SetEndTime();
	}
	
	// mtx * x = b
	for (int it = 0; it < ite; it++) {
		timer.SetStartTime();
		jacobi_sell_c_sigma(/*       TODO      */);
		timer.SetEndTime();
		res = min(res, timer.GetDifferenceUs());
	}
	
	long long res = numeric_limits<long long>::max();
	for (int it = 0; it < ite; it++) {
		timer.SetStartTime();
		spmv_sell_c_sigma_noalloc(mtx, v, threads_num, scs_res);
		timer.SetEndTime();
		res = min(res, timer.GetDifferenceUs());
	}
	
	vector_format<T> real_scs_res = alloc_vector_res(mtx);
	for (int i = 0; i < mtx_CSR.N; i++) {
		real_scs_res.value[mtx.rows_perm[i]] = scs_res.value[i];
	}
	
	TestResult test_result;
	test_result.NAME = name;
	test_result.FUNCTION = function;
	test_result.TIME_MS = us_to_ms_string(res);
	test_result.NZ = to_string(mtx.cs[mtx.N / C]);
	test_result.DIFF = to_string(calc_diff(naive_res, real_scs_res));
	test_result.CONV_TIME = us_to_ms_string(conv_time);
	test_result.PREPROC_TIME = us_to_ms_string(preproc_time);
	return test_result;
}

matrix_CSR<float> mtx_CSR_double_to_mtx_CSR_float(const matrix_CSR<double>& mtx) {
	matrix_CSR<float> res;
	res.N = mtx.N;
	res.M = mtx.M;
	res.row_id = new int[res.N + 1];
	for (int i = 0; i < res.N + 1; i++) {
		res.row_id[i] = mtx.row_id[i];
	}
	int nz = res.row_id[res.N];
	res.col = new int[nz];
	res.value = new float[nz];
	for (int i = 0; i < nz; i++) {
		res.col[i] = mtx.col[i];
		res.value[i] = mtx.value[i];
	}
	return res;
}

void PrintTableJacobi(const vector<JacobiResult>& test_results, string filename, int threads_num, int ite) {
	cout << "table_start" << endl;
	cout << "mtx= " << filename << " threads_num= " << threads_num << " ite= " << ite << endl; 
	const string title_NAME = "NAME";
	const string title_FUNCTION = "FUNCTION";
	const string title_TIME_MS = "TIME_MS";
	const string title_NZ = "NZ";
	const string title_DIFF = "DIFF";
	const string title_CONV_TIME = "CONV_TIME";
	const string title_PREPROC_TIME = "PREPROC_TIME";
	size_t NAME_sz = title_NAME.size();
	size_t FUNCTION_sz = title_FUNCTION.size();
	size_t TIME_MS_sz = title_TIME_MS.size();
	size_t NZ_sz = title_NZ.size();
	size_t DIFF_sz = title_DIFF.size();
	size_t CONV_TIME_sz = title_CONV_TIME.size();
	size_t PREPROC_TIME_sz = title_PREPROC_TIME.size();
	for (auto res : test_results) {
		NAME_sz         = max(NAME_sz,         res.NAME.size());
		FUNCTION_sz     = max(FUNCTION_sz,     res.FUNCTION.size());
		TIME_MS_sz      = max(TIME_MS_sz,      res.TIME_MS.size());
		NZ_sz           = max(NZ_sz,           res.NZ.size());
		DIFF_sz         = max(DIFF_sz,         res.DIFF.size());
		CONV_TIME_sz    = max(CONV_TIME_sz,    res.CONV_TIME.size());
		PREPROC_TIME_sz = max(PREPROC_TIME_sz, res.PREPROC_TIME.size());
 	}
	
	cout << title_NAME << string(NAME_sz - title_NAME.size() + 1, ' ');
	cout << title_FUNCTION << string(FUNCTION_sz - title_FUNCTION.size() + 1, ' ');
	cout << title_TIME_MS << string(TIME_MS_sz - title_TIME_MS.size() + 1, ' ');
	cout << title_NZ << string(NZ_sz - title_NZ.size() + 1, ' ');
	cout << title_DIFF << string(DIFF_sz - title_DIFF.size() + 1, ' ');
	cout << title_CONV_TIME << string(CONV_TIME_sz - title_CONV_TIME.size() + 1, ' ');
	cout << title_PREPROC_TIME << string(PREPROC_TIME_sz - title_PREPROC_TIME.size() + 1, ' ');
	cout << endl;
	for (auto res : test_results) {
		cout << res.NAME         << string(NAME_sz - res.NAME.size() + 1, ' ');
		cout << res.FUNCTION     << string(FUNCTION_sz - res.FUNCTION.size() + 1, ' ');
		cout << res.TIME_MS      << string(TIME_MS_sz - res.TIME_MS.size() + 1, ' ');
		cout << res.NZ           << string(NZ_sz - res.NZ.size() + 1, ' ');
		cout << res.DIFF         << string(DIFF_sz  - res.DIFF.size() + 1, ' ');
		cout << res.CONV_TIME    << string(CONV_TIME_sz  - res.CONV_TIME.size() + 1, ' ');
		cout << res.PREPROC_TIME << string(PREPROC_TIME_sz  - res.PREPROC_TIME.size() + 1, ' ');
		cout << endl;
 	}
	cout << "table_end" << endl;
}

int main(int argc, char** argv) {
	
	cout << "------------------------------------------------------" << endl;
	
	if (argc <= 1) {
		cout << "error : no name of file with matrix!" << endl;
		return -1;
	}
	
	// file in BIN format
	char* filename_bin = argv[1];
	
	cout << "RUN MAIN_JACOBI_EXE WITH MATRIX " << filename_bin << endl;
	
	int threads_num = omp_get_max_threads();
	if (argc >= 3) {
		int thr_num = atoi(argv[2]);
		if (1 <= thr_num && thr_num <= omp_get_max_threads()) {
			threads_num = thr_num;
		}
		else {
			cout << "error : wrong threads_num : " << argv[2] << endl;
			return -1;
		}
	}
	omp_set_num_threads(threads_num);
	
	// number of iterations for each
	int ite = 1000;
	if (argc >= 4) {
		int it = atoi(argv[3]);
		if (1 <= it && it <= 100000) {
			ite = it;
		}
		else {
			cout << "error : wrong number of iterations : " << argv[3] << endl;
			return -1;
		}
	}
	
	cout << "threads_num = " << threads_num << "; ";
	cout << "ite = " << ite << "; ";
	cout << "WARM_UP_ITE = " << WARM_UP_ITE << ";" << endl;
	
	MyTimer timer;
	
	matrix_CSR<double> mtx_CSR;
	try {
		timer.SetStartTime();
		mtx_CSR = read_BIN_to_CSR(filename_bin);
		timer.SetEndTime();
	}
	catch (...) {
		cout << "error in reading matrix " << filename_bin << endl;
		return -1;
	}
	
	auto read_time = timer.GetDifferenceUs();
	
	timer.SetStartTime();
	transpose_CSR(mtx_CSR);
	transpose_CSR(mtx_CSR);
	timer.SetEndTime();
	
	auto transpose_time = timer.GetDifferenceUs();
	
	cout << "read in " << us_to_ms_string(read_time) << "ms; ";
	cout << "2x transpose in " << us_to_ms_string(transpose_time) << "ms; " << endl;
	
	int N = mtx_CSR.N;
	int M = mtx_CSR.M;
	cout << "N = " << N << "; ";
	cout << "M = " << M << "; ";
	cout << "nz = " << mtx_CSR.row_id[N] << ";" << endl;
	
	timer.SetStartTime();
	matrix_CSR<float> mtx_CSR_float = mtx_CSR_double_to_mtx_CSR_float(mtx_CSR);
	timer.SetEndTime();
	
	auto convert2float_time = timer.GetDifferenceUs();
	cout << "convert2float in " << us_to_ms_string(convert2float_time) << "ms;" << endl;
	
	// CHECK IF MATRIX DIAGONALLY DOMINANT
	
	bool is_diagonally_dominant = true;
	for (int i = 0; i < mtx_CSR.N; i++) {
		double diag_sum = 0;
		double non_diag_sum = 0;
		
		for (int j = mtx_CSR.row_id[i]; j < mtx_CSR.row_id[i + 1]; j++) {
			if (mtx_CSR.col[j] == i) {
				diag_sum += abs(mtx_CSR.value[j]);
			}
			else {
				non_diag_sum += abs(mtx_CSR.value[j]);
			}
		}
		
		if (non_diag_sum > diag_sum) {
			is_diagonally_dominant = false;
		}
	}
	
	if (is_diagonally_dominant) {
		cout << "MATRIX IS DIAGONALLY DOMINANT" << endl;
	}
	else {
		cout << "MATRIX IS NOT DIAGONALLY DOMINANT" << endl;
		cout << "END MAIN_JACOBI_EXE WITH MATRIX " << filename_bin << endl;
		exit(0);
	}
	
	if (N != M) {
		cout << "N IS NOT EQUAL TO M" << endl;
		cout << "END MAIN_JACOBI_EXE WITH MATRIX " << filename_bin << endl;
		exit(0);
	}
	
	// Ax = b
	
	// INIT b
	b.alloc(M, 32);
	// b.N = M;
	// b.value = new double[b.N];
	mt19937_64 gen(38);
	uniform_real_distribution<> dis(0.0, 1.0);
	for (int i = 0; i < b.N; i++) {
		b.value[i] = dis(gen);
	}
	
	// INIT b_float
	b_float.alloc(M, 64);
	for (int i = 0; i < b_float.N; i++) {
		b_float.value[i] = b.value[i];
	}
	
	cout << "vector b[" << b.N << "]: ";
	for (int i = 0; i < min(10, b.N); i++) {
		cout << b.value[i] << " ";
	}
	if (b.N > 10) {
		cout << "...";
	}
	cout << endl;
	cout << "------------------------------------------------------" << endl;
	
	// DOUBLE
	
	vector<JacobiResult> test_results_double;
	
	test_results_double.push_back(test_jacobi_naivenaive(ite, threads_num, mtx_CSR));
	
	cout << "vector naive_res[" << naive_res.N << "]: ";
	for (int i = 0; i < min(10, naive_res.N); i++) {
		cout << naive_res.value[i] << " ";
	}
	if (naive_res.N > 10) {
		cout << "...";
	}
	cout << endl;
	
	// test_results_double.push_back(test_jacobi_albus<double>(ite, threads_num, mtx_CSR));
	// test_results_double.push_back(test_jacobi_albus_v<double, 1>(ite, threads_num, mtx_CSR));
	// test_results_double.push_back(test_jacobi_albus_v<double, 2>(ite, threads_num, mtx_CSR));
	// test_results_double.push_back(test_jacobi_albus_v<double, 4>(ite, threads_num, mtx_CSR));
	// test_results_double.push_back(test_jacobi_albus_v<double, 8>(ite, threads_num, mtx_CSR));
	
	test_results_double.push_back(test_jacobi_sell_c_sigma<double,  4, 1>(ite, threads_num, mtx_CSR));
	test_results_double.push_back(test_jacobi_sell_c_sigma<double,  8, 1>(ite, threads_num, mtx_CSR));
	test_results_double.push_back(test_jacobi_sell_c_sigma<double, 16, 1>(ite, threads_num, mtx_CSR));
	test_results_double.push_back(test_jacobi_sell_c_sigma<double, 32, 1>(ite, threads_num, mtx_CSR));
	
	test_results_double.push_back(test_jacobi_sell_c_sigma<double,  4, 2>(ite, threads_num, mtx_CSR));
	test_results_double.push_back(test_jacobi_sell_c_sigma<double,  8, 2>(ite, threads_num, mtx_CSR));
	test_results_double.push_back(test_jacobi_sell_c_sigma<double, 16, 2>(ite, threads_num, mtx_CSR));
	test_results_double.push_back(test_jacobi_sell_c_sigma<double, 32, 2>(ite, threads_num, mtx_CSR));
	
	test_results_double.push_back(test_jacobi_sell_c_sigma<double,  4, 4>(ite, threads_num, mtx_CSR));
	test_results_double.push_back(test_jacobi_sell_c_sigma<double,  8, 4>(ite, threads_num, mtx_CSR));
	test_results_double.push_back(test_jacobi_sell_c_sigma<double, 16, 4>(ite, threads_num, mtx_CSR));
	test_results_double.push_back(test_jacobi_sell_c_sigma<double, 32, 4>(ite, threads_num, mtx_CSR));
	
	test_results_double.push_back(test_jacobi_sell_c_sigma<double,  4, 8>(ite, threads_num, mtx_CSR));
	test_results_double.push_back(test_jacobi_sell_c_sigma<double,  8, 8>(ite, threads_num, mtx_CSR));
	test_results_double.push_back(test_jacobi_sell_c_sigma<double, 16, 8>(ite, threads_num, mtx_CSR));
	test_results_double.push_back(test_jacobi_sell_c_sigma<double, 32, 8>(ite, threads_num, mtx_CSR));

	test_results_double.push_back(test_jacobi_sell_c_sigma<double,  4, SIGMA_SORTED>(ite, threads_num, mtx_CSR));
	test_results_double.push_back(test_jacobi_sell_c_sigma<double,  8, SIGMA_SORTED>(ite, threads_num, mtx_CSR));
	test_results_double.push_back(test_jacobi_sell_c_sigma<double, 16, SIGMA_SORTED>(ite, threads_num, mtx_CSR));
	test_results_double.push_back(test_jacobi_sell_c_sigma<double, 32, SIGMA_SORTED>(ite, threads_num, mtx_CSR));
	
	// test_results_double.push_back(test_jacobi_sell_c_sigma_novec<double,  4, 1>(ite, threads_num, mtx_CSR));
	// test_results_double.push_back(test_jacobi_sell_c_sigma_novec<double,  8, 1>(ite, threads_num, mtx_CSR));
	// test_results_double.push_back(test_jacobi_sell_c_sigma_novec<double, 16, 1>(ite, threads_num, mtx_CSR));
	// test_results_double.push_back(test_jacobi_sell_c_sigma_novec<double, 32, 1>(ite, threads_num, mtx_CSR));
	
	// FLOAT
	
	vector<JacobiResult> test_results_float;
	
	test_results_float.push_back(test_jacobi_naive(ite, threads_num, mtx_CSR_float));
	
	cout << "vector naive_res_float[" << naive_res_float.N << "]: ";
	for (int i = 0; i < min(10, naive_res_float.N); i++) {
		cout << naive_res_float.value[i] << " ";
	}
	if (naive_res_float.N > 10) {
		cout << "...";
	}
	cout << endl;
	
	// test_results_float.push_back(test_jacobi_albus<float>(ite, threads_num, mtx_CSR_float));
	// test_results_float.push_back(test_jacobi_albus_v<float, 1>(ite, threads_num, mtx_CSR_float));
	// test_results_float.push_back(test_jacobi_albus_v<float, 2>(ite, threads_num, mtx_CSR_float));
	// test_results_float.push_back(test_jacobi_albus_v<float, 4>(ite, threads_num, mtx_CSR_float));
	// test_results_float.push_back(test_jacobi_albus_v<float, 8>(ite, threads_num, mtx_CSR_float));
	
	test_results_float.push_back(test_jacobi_sell_c_sigma<float,  8, 1>(ite, threads_num, mtx_CSR_float));
	test_results_float.push_back(test_jacobi_sell_c_sigma<float, 16, 1>(ite, threads_num, mtx_CSR_float));
	test_results_float.push_back(test_jacobi_sell_c_sigma<float, 32, 1>(ite, threads_num, mtx_CSR_float));
	test_results_float.push_back(test_jacobi_sell_c_sigma<float, 64, 1>(ite, threads_num, mtx_CSR_float));
	
	test_results_float.push_back(test_jacobi_sell_c_sigma<float,  8, 2>(ite, threads_num, mtx_CSR_float));
	test_results_float.push_back(test_jacobi_sell_c_sigma<float, 16, 2>(ite, threads_num, mtx_CSR_float));
	test_results_float.push_back(test_jacobi_sell_c_sigma<float, 32, 2>(ite, threads_num, mtx_CSR_float));
	test_results_float.push_back(test_jacobi_sell_c_sigma<float, 64, 2>(ite, threads_num, mtx_CSR_float));
	
	test_results_float.push_back(test_jacobi_sell_c_sigma<float,  8, 4>(ite, threads_num, mtx_CSR_float));
	test_results_float.push_back(test_jacobi_sell_c_sigma<float, 16, 4>(ite, threads_num, mtx_CSR_float));
	test_results_float.push_back(test_jacobi_sell_c_sigma<float, 32, 4>(ite, threads_num, mtx_CSR_float));
	test_results_float.push_back(test_jacobi_sell_c_sigma<float, 64, 4>(ite, threads_num, mtx_CSR_float));
	
	test_results_float.push_back(test_jacobi_sell_c_sigma<float,  8, 8>(ite, threads_num, mtx_CSR_float));
	test_results_float.push_back(test_jacobi_sell_c_sigma<float, 16, 8>(ite, threads_num, mtx_CSR_float));
	test_results_float.push_back(test_jacobi_sell_c_sigma<float, 32, 8>(ite, threads_num, mtx_CSR_float));
	test_results_float.push_back(test_jacobi_sell_c_sigma<float, 64, 8>(ite, threads_num, mtx_CSR_float));
	
	test_results_float.push_back(test_jacobi_sell_c_sigma<float,  8, SIGMA_SORTED>(ite, threads_num, mtx_CSR_float));
	test_results_float.push_back(test_jacobi_sell_c_sigma<float, 16, SIGMA_SORTED>(ite, threads_num, mtx_CSR_float));
	test_results_float.push_back(test_jacobi_sell_c_sigma<float, 32, SIGMA_SORTED>(ite, threads_num, mtx_CSR_float));
	test_results_float.push_back(test_jacobi_sell_c_sigma<float, 64, SIGMA_SORTED>(ite, threads_num, mtx_CSR_float));
	
	// test_results_float.push_back(test_jacobi_sell_c_sigma_novec<float,  8, 1>(ite, threads_num, mtx_CSR_float));
	// test_results_float.push_back(test_jacobi_sell_c_sigma_novec<float, 16, 1>(ite, threads_num, mtx_CSR_float));
	// test_results_float.push_back(test_jacobi_sell_c_sigma_novec<float, 32, 1>(ite, threads_num, mtx_CSR_float));
	// test_results_float.push_back(test_jacobi_sell_c_sigma_novec<float, 64, 1>(ite, threads_num, mtx_CSR_float));
	
	string filename_string = filename_bin;
	cout << "------------------------------------------------------" << endl;
	PrintTableJacobi(test_results_double, filename_string + " type= double", threads_num, ite);
	cout << "------------------------------------------------------" << endl;
	PrintTableJacobi(test_results_float, filename_string + " type= float", threads_num, ite);
	cout << "------------------------------------------------------" << endl;
	
	cout << "END MAIN_JACOBI_EXE WITH MATRIX " << filename_bin << endl;
	cout << "------------------------------------------------------" << endl;
	cout << endl << endl << endl;
	
 	return 0;
}
