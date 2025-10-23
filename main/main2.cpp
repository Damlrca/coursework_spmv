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

const string NONE = "---";

struct TestResult{
	string NAME = NONE;      // short name for algorithm
	string FUNCTION = NONE;  // full function name
	string TIME_MS = NONE;   // time in milliseconds
	string DIFF = NONE;      // difference in result between naive and this approach
	string NZ = NONE;        // number of non-zero elements in this format
	string CONV_TIME = NONE; // time of convertion to format
};

vector_format<double> v, naive_res;
vector_format<float> v_float, naive_res_float;
vector<TestResult> test_results;
constexpr int WARM_UP_ITE = 2; // number of warm up runs before actual measurement

// test_naive double
TestResult test_naive(const int ite, const int threads_num, const matrix_CSR<double>& mtx_CSR) {
	MyTimer timer;
	const string name = "naive";
	const string function = "spmv_naive_noalloc<double>";
	cout << "run " << name << " " << function << " ..." << endl;
	
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

// test_naive float
TestResult test_naive(const int ite, const int threads_num, const matrix_CSR<float>& mtx_CSR) {
	MyTimer timer;
	const string name = "naive_f";
	const string function = "spmv_naive_noalloc<float>";
	cout << "run " << name << " " << function << " ..." << endl;
	
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

// test_albus double
TestResult test_albus(const int ite, const int threads_num, const matrix_CSR<double>& mtx_CSR) {
	MyTimer timer;
	const string name = "albus";
	const string function = "spmv_albus_omp_noalloc<double>";
	cout << "run " << name << " preproc_albus_balance ..." << endl;
	
	int start[100];
	int block_start[100];
	
	timer.SetStartTime();
	preproc_albus_balance(mtx_CSR, start, block_start, threads_num);
	timer.SetEndTime();
	auto balance_time = timer.GetDifferenceUs();
	
	cout << "run " << name << " " << function << " ..." << endl;
	
	vector_format<double> albus_omp_res = alloc_vector_res(mtx_CSR);
	
	for (int it = 0; it < WARM_UP_ITE; it++) {
		timer.SetStartTime();
		spmv_albus_omp_noalloc(mtx_CSR, v, start, block_start, threads_num, albus_omp_res);
		timer.SetEndTime();
	}
	long long res = numeric_limits<long long>::max();
	for (int it = 0; it < ite; it++) {
		timer.SetStartTime();
		spmv_albus_omp_noalloc(mtx_CSR, v, start, block_start, threads_num, albus_omp_res);
		timer.SetEndTime();
		res = min(res, timer.GetDifferenceUs());
	}
	
	TestResult test_result;
	test_result.NAME = name;
	test_result.FUNCTION = function;
	test_result.TIME_MS = us_to_ms_string(res);
	test_result.NZ = to_string(mtx_CSR.row_id[mtx_CSR.N]);
	test_result.DIFF = to_string(calc_diff(naive_res, albus_omp_res));
	test_result.CONV_TIME = us_to_ms_string(balance_time) + "(balance)";
	return test_result;
}

// test_albus float
TestResult test_albus(const int ite, const int threads_num, const matrix_CSR<float>& mtx_CSR) {
	MyTimer timer;
	const string name = "albus_f";
	const string function = "spmv_albus_omp_noalloc<float>";
	cout << "run " << name << " preproc_albus_balance ..." << endl;
	
	int start[100];
	int block_start[100];
	
	timer.SetStartTime();
	preproc_albus_balance(mtx_CSR, start, block_start, threads_num);
	timer.SetEndTime();
	auto balance_time = timer.GetDifferenceUs();
	
	cout << "run " << name << " " << function << " ..." << endl;
	
	vector_format<float> albus_omp_res = alloc_vector_res(mtx_CSR);
	
	for (int it = 0; it < WARM_UP_ITE; it++) {
		timer.SetStartTime();
		spmv_albus_omp_noalloc(mtx_CSR, v_float, start, block_start, threads_num, albus_omp_res);
		timer.SetEndTime();
	}
	long long res = numeric_limits<long long>::max();
	for (int it = 0; it < ite; it++) {
		timer.SetStartTime();
		spmv_albus_omp_noalloc(mtx_CSR, v_float, start, block_start, threads_num, albus_omp_res);
		timer.SetEndTime();
		res = min(res, timer.GetDifferenceUs());
	}
	
	TestResult test_result;
	test_result.NAME = name;
	test_result.FUNCTION = function;
	test_result.TIME_MS = us_to_ms_string(res);
	test_result.NZ = to_string(mtx_CSR.row_id[mtx_CSR.N]);
	test_result.DIFF = to_string(calc_diff(naive_res_float, albus_omp_res));
	test_result.CONV_TIME = us_to_ms_string(balance_time) + "(balance)";
	return test_result;
}

// test_albus_v double
template <int M>
TestResult test_albus_v(const int ite, const int threads_num, const matrix_CSR<double>& mtx_CSR) {
	MyTimer timer;
	const string name = "albus_v_m" + to_string(M);
	const string function = "spmv_albus_omp_v_noalloc<double, " + to_string(M) + ">";
	cout << "run " << name << " preproc_albus_balance ..." << endl;
	
	int start[100];
	int block_start[100];
	
	timer.SetStartTime();
	preproc_albus_balance(mtx_CSR, start, block_start, threads_num);
	timer.SetEndTime();
	auto balance_time = timer.GetDifferenceUs();
	
	cout << "run " << name << " " << function << " ..." << endl;
	
	vector_format<double> albus_omp_v_res = alloc_vector_res(mtx_CSR);
	
	for (int it = 0; it < WARM_UP_ITE; it++) {
		timer.SetStartTime();
		spmv_albus_omp_v_noalloc<double, M>(mtx_CSR, v, start, block_start, threads_num, albus_omp_v_res);
		timer.SetEndTime();
	}
	long long res_omp_v = numeric_limits<long long>::max();
	for (int it = 0; it < ite; it++) {
		timer.SetStartTime();
		spmv_albus_omp_v_noalloc<double, M>(mtx_CSR, v, start, block_start, threads_num, albus_omp_v_res);
		timer.SetEndTime();
		res_omp_v = min(res_omp_v, timer.GetDifferenceUs());
	}
	
	TestResult test_result;
	test_result.NAME = name;
	test_result.FUNCTION = function;
	test_result.TIME_MS = us_to_ms_string(res_omp_v);
	test_result.NZ = to_string(mtx_CSR.row_id[mtx_CSR.N]);
	test_result.DIFF = to_string(calc_diff(naive_res, albus_omp_v_res));
	test_result.CONV_TIME = us_to_ms_string(balance_time) + "(balance)";
	return test_result;
}

// test_albus_v float
template <int M>
TestResult test_albus_v(const int ite, const int threads_num, const matrix_CSR<float>& mtx_CSR) {
	MyTimer timer;
	const string name = "albus_v_m" + to_string(M) + "_f";
	const string function = "spmv_albus_omp_v_noalloc<float, " + to_string(M) + ">";
	cout << "run " << name << " preproc_albus_balance ..." << endl;
	
	int start[100];
	int block_start[100];
	
	timer.SetStartTime();
	preproc_albus_balance(mtx_CSR, start, block_start, threads_num);
	timer.SetEndTime();
	auto balance_time = timer.GetDifferenceUs();
	
	cout << "run " << name << " " << function << " ..." << endl;
	
	vector_format<float> albus_omp_v_res = alloc_vector_res(mtx_CSR);
	
	for (int it = 0; it < WARM_UP_ITE; it++) {
		timer.SetStartTime();
		spmv_albus_omp_v_noalloc<float, M>(mtx_CSR, v_float, start, block_start, threads_num, albus_omp_v_res);
		timer.SetEndTime();
	}
	long long res_omp_v = numeric_limits<long long>::max();
	for (int it = 0; it < ite; it++) {
		timer.SetStartTime();
		spmv_albus_omp_v_noalloc<float, M>(mtx_CSR, v_float, start, block_start, threads_num, albus_omp_v_res);
		timer.SetEndTime();
		res_omp_v = min(res_omp_v, timer.GetDifferenceUs());
	}
	
	TestResult test_result;
	test_result.NAME = name;
	test_result.FUNCTION = function;
	test_result.TIME_MS = us_to_ms_string(res_omp_v);
	test_result.NZ = to_string(mtx_CSR.row_id[mtx_CSR.N]);
	test_result.DIFF = to_string(calc_diff(naive_res_float, albus_omp_v_res));
	test_result.CONV_TIME = us_to_ms_string(balance_time) + "(balance)";
	return test_result;
}

constexpr int SIGMA_SORTED = 9'999'999;

// test_sell_c_sigma double
template<int C, int sigma>
TestResult test_sell_c_sigma(const int ite, const int threads_num, const matrix_CSR<double>& mtx_CSR) {
	MyTimer timer;
	const string name = "scs_" + to_string(C) + "_" + (sigma == SIGMA_SORTED ? string("S") : to_string(sigma));
	const string function = "spmv_sell_c_sigma_noalloc<double, " + to_string(C) + ", " + to_string(sigma) + ">";
	cout << "run " << name << " " << function << endl;
	
	timer.SetStartTime();
	matrix_SELL_C_sigma<C, sigma, double> mtx = convert_CSR_to_SELL_C_sigma<C, sigma>(mtx_CSR);
	timer.SetEndTime();
	auto conv_time = timer.GetDifferenceUs();
	
	vector_format<double> scs_res = alloc_vector_res(mtx);
	
	for (int it = 0; it < WARM_UP_ITE; it++) {
		timer.SetStartTime();
		spmv_sell_c_sigma_noalloc(mtx, v, threads_num, scs_res);
		timer.SetEndTime();
	}
	long long res = numeric_limits<long long>::max();
	for (int it = 0; it < ite; it++) {
		timer.SetStartTime();
		spmv_sell_c_sigma_noalloc(mtx, v, threads_num, scs_res);
		timer.SetEndTime();
		res = min(res, timer.GetDifferenceUs());
	}
	
	vector_format<double> real_scs_res = alloc_vector_res(mtx);
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
	return test_result;
}

// test_sell_c_sigma float
template<int C, int sigma>
TestResult test_sell_c_sigma(const int ite, const int threads_num, const matrix_CSR<float>& mtx_CSR) {
	MyTimer timer;
	const string name = "scs_" + to_string(C) + "_" + (sigma == SIGMA_SORTED ? string("S") : to_string(sigma)) + "_f";
	const string function = "spmv_sell_c_sigma_noalloc<float, " + to_string(C) + ", " + to_string(sigma) + ">";
	cout << "run " << name << " " << function << endl;
	
	timer.SetStartTime();
	matrix_SELL_C_sigma<C, sigma, float> mtx = convert_CSR_to_SELL_C_sigma<C, sigma>(mtx_CSR);
	timer.SetEndTime();
	auto conv_time = timer.GetDifferenceUs();
	
	vector_format<float> scs_res = alloc_vector_res(mtx);
	
	for (int it = 0; it < WARM_UP_ITE; it++) {
		timer.SetStartTime();
		spmv_sell_c_sigma_noalloc(mtx, v_float, threads_num, scs_res);
		timer.SetEndTime();
	}
	long long res = numeric_limits<long long>::max();
	for (int it = 0; it < ite; it++) {
		timer.SetStartTime();
		spmv_sell_c_sigma_noalloc(mtx, v_float, threads_num, scs_res);
		timer.SetEndTime();
		res = min(res, timer.GetDifferenceUs());
	}
	
	vector_format<float> real_scs_res = alloc_vector_res(mtx);
	for (int i = 0; i < mtx_CSR.N; i++) {
		real_scs_res.value[mtx.rows_perm[i]] = scs_res.value[i];
	}
	
	TestResult test_result;
	test_result.NAME = name;
	test_result.FUNCTION = function;
	test_result.TIME_MS = us_to_ms_string(res);
	test_result.NZ = to_string(mtx.cs[mtx.N / C]);
	test_result.DIFF = to_string(calc_diff(naive_res_float, real_scs_res));
	test_result.CONV_TIME = us_to_ms_string(conv_time);
	return test_result;
}

// test_sell_c_sigma_novec double
template<int C, int sigma>
TestResult test_sell_c_sigma_novec(const int ite, const int threads_num, const matrix_CSR<double>& mtx_CSR) {
	MyTimer timer;
	const string name = "scs_" + to_string(C) + "_" + (sigma == SIGMA_SORTED ? string("S") : to_string(sigma)) + "_novec";
	const string function = "spmv_sell_c_sigma_noalloc_novec<double, " + to_string(C) + ", " + to_string(sigma) + ">";
	cout << "run " << name << " " << function << endl;
	
	timer.SetStartTime();
	matrix_SELL_C_sigma<C, sigma, double> mtx = convert_CSR_to_SELL_C_sigma<C, sigma>(mtx_CSR);
	timer.SetEndTime();
	auto conv_time = timer.GetDifferenceUs();
	
	vector_format<double> scs_res = alloc_vector_res(mtx);
	
	for (int it = 0; it < WARM_UP_ITE; it++) {
		timer.SetStartTime();
		spmv_sell_c_sigma_noalloc_novec(mtx, v, threads_num, scs_res);
		timer.SetEndTime();
	}
	long long res = numeric_limits<long long>::max();
	for (int it = 0; it < ite; it++) {
		timer.SetStartTime();
		spmv_sell_c_sigma_noalloc_novec(mtx, v, threads_num, scs_res);
		timer.SetEndTime();
		res = min(res, timer.GetDifferenceUs());
	}
	
	vector_format<double> real_scs_res = alloc_vector_res(mtx);
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
	return test_result;
}

// test_sell_c_sigma_novec float
template<int C, int sigma>
TestResult test_sell_c_sigma_novec(const int ite, const int threads_num, const matrix_CSR<float>& mtx_CSR) {
	MyTimer timer;
	const string name = "scs_" + to_string(C) + "_" + (sigma == SIGMA_SORTED ? string("S") : to_string(sigma)) + "_novec" + "_f";
	const string function = "spmv_sell_c_sigma_noalloc_novec<float, " + to_string(C) + ", " + to_string(sigma) + ">";
	cout << "run " << name << " " << function << endl;
	
	timer.SetStartTime();
	matrix_SELL_C_sigma<C, sigma, float> mtx = convert_CSR_to_SELL_C_sigma<C, sigma>(mtx_CSR);
	timer.SetEndTime();
	auto conv_time = timer.GetDifferenceUs();
	
	vector_format<float> scs_res = alloc_vector_res(mtx);
	
	for (int it = 0; it < WARM_UP_ITE; it++) {
		timer.SetStartTime();
		spmv_sell_c_sigma_noalloc_novec(mtx, v_float, threads_num, scs_res);
		timer.SetEndTime();
	}
	long long res = numeric_limits<long long>::max();
	for (int it = 0; it < ite; it++) {
		timer.SetStartTime();
		spmv_sell_c_sigma_noalloc_novec(mtx, v_float, threads_num, scs_res);
		timer.SetEndTime();
		res = min(res, timer.GetDifferenceUs());
	}
	
	vector_format<float> real_scs_res = alloc_vector_res(mtx);
	for (int i = 0; i < mtx_CSR.N; i++) {
		real_scs_res.value[mtx.rows_perm[i]] = scs_res.value[i];
	}
	
	TestResult test_result;
	test_result.NAME = name;
	test_result.FUNCTION = function;
	test_result.TIME_MS = us_to_ms_string(res);
	test_result.NZ = to_string(mtx.cs[mtx.N / C]);
	test_result.DIFF = to_string(calc_diff(naive_res_float, real_scs_res));
	test_result.CONV_TIME = us_to_ms_string(conv_time);
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

int main(int argc, char** argv) {
	
	cout << "------------------------------------------------------" << endl;
	
	if (argc <= 1) {
		cout << "error : no name of file with matrix!" << endl;
		return -1;
	}
	
	// file in BIN format
	char* filename_bin = argv[1];
	
	cout << "RUN MAIN2_EXE WITH MATRIX " << filename_bin << endl;
	
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
	cout << "2x transpose in  " << us_to_ms_string(transpose_time) << "ms; " << endl;
	
	int N = mtx_CSR.N;
	int M = mtx_CSR.M;
	cout << "N = " << N << "; ";
	cout << "M = " << M << "; ";
	cout << "nz = " << mtx_CSR.row_id[N] << ";" << endl;
	
	timer.SetStartTime();
	matrix_CSR<float> mtx_CSR_float = mtx_CSR_double_to_mtx_CSR_float(mtx_CSR);
	timer.SetEndTime();
	
	auto convert2float_time = timer.GetDifferenceUs();
	cout << "convert2float in  " << us_to_ms_string(convert2float_time) << "ms;" << endl;
	
	// INIT v
	v.alloc(M, 32);
	// v.N = M;
	// v.value = new double[v.N];
	mt19937_64 gen(38);
	uniform_real_distribution<> dis(0.0, 1.0);
	for (int i = 0; i < v.N; i++) {
		v.value[i] = dis(gen);
	}
	
	// INIT v_float
	v_float.alloc(M, 64);
	for (int i = 0; i < v_float.N; i++) {
		v_float.value[i] = v.value[i];
	}
	
	cout << "vector v[" << v.N << "]: ";
	for (int i = 0; i < min(10, v.N); i++) {
		cout << v.value[i] << " ";
	}
	if (v.N > 10) {
		cout << "...";
	}
	cout << endl;
	cout << "------------------------------------------------------" << endl;
	
	// DOUBLE
	
	test_results.push_back(test_naive(ite, threads_num, mtx_CSR));
	
	cout << "vector naive_res[" << naive_res.N << "]: ";
	for (int i = 0; i < min(10, naive_res.N); i++) {
		cout << naive_res.value[i] << " ";
	}
	if (naive_res.N > 10) {
		cout << "...";
	}
	cout << endl;
	
	test_results.push_back(test_albus(ite, threads_num, mtx_CSR));
	test_results.push_back(test_albus_v<1>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_albus_v<2>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_albus_v<4>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_albus_v<8>(ite, threads_num, mtx_CSR));
	
	test_results.push_back(test_sell_c_sigma< 4, 1>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_sell_c_sigma< 8, 1>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_sell_c_sigma<16, 1>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_sell_c_sigma<32, 1>(ite, threads_num, mtx_CSR));
	
	test_results.push_back(test_sell_c_sigma< 4, 2>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_sell_c_sigma< 8, 2>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_sell_c_sigma<16, 2>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_sell_c_sigma<32, 2>(ite, threads_num, mtx_CSR));
	
	test_results.push_back(test_sell_c_sigma< 4, 4>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_sell_c_sigma< 8, 4>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_sell_c_sigma<16, 4>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_sell_c_sigma<32, 4>(ite, threads_num, mtx_CSR));
	
	test_results.push_back(test_sell_c_sigma< 4, 8>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_sell_c_sigma< 8, 8>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_sell_c_sigma<16, 8>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_sell_c_sigma<32, 8>(ite, threads_num, mtx_CSR));

	test_results.push_back(test_sell_c_sigma< 4, SIGMA_SORTED>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_sell_c_sigma< 8, SIGMA_SORTED>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_sell_c_sigma<16, SIGMA_SORTED>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_sell_c_sigma<32, SIGMA_SORTED>(ite, threads_num, mtx_CSR));
	
	test_results.push_back(test_sell_c_sigma_novec< 4, 1>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_sell_c_sigma_novec< 8, 1>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_sell_c_sigma_novec<16, 1>(ite, threads_num, mtx_CSR));
	test_results.push_back(test_sell_c_sigma_novec<32, 1>(ite, threads_num, mtx_CSR));
	
	// FLOAT
	
	test_results.push_back(test_naive(ite, threads_num, mtx_CSR_float));
	
	cout << "vector naive_res_float[" << naive_res_float.N << "]: ";
	for (int i = 0; i < min(10, naive_res_float.N); i++) {
		cout << naive_res_float.value[i] << " ";
	}
	if (naive_res_float.N > 10) {
		cout << "...";
	}
	cout << endl;
	
	test_results.push_back(test_albus(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_albus_v<1>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_albus_v<2>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_albus_v<4>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_albus_v<8>(ite, threads_num, mtx_CSR_float));
	
	test_results.push_back(test_sell_c_sigma< 8, 1>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_sell_c_sigma<16, 1>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_sell_c_sigma<32, 1>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_sell_c_sigma<64, 1>(ite, threads_num, mtx_CSR_float));
	
	test_results.push_back(test_sell_c_sigma< 8, 2>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_sell_c_sigma<16, 2>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_sell_c_sigma<32, 2>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_sell_c_sigma<64, 2>(ite, threads_num, mtx_CSR_float));
	
	test_results.push_back(test_sell_c_sigma< 8, 4>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_sell_c_sigma<16, 4>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_sell_c_sigma<32, 4>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_sell_c_sigma<64, 4>(ite, threads_num, mtx_CSR_float));
	
	test_results.push_back(test_sell_c_sigma< 8, 8>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_sell_c_sigma<16, 8>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_sell_c_sigma<32, 8>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_sell_c_sigma<64, 8>(ite, threads_num, mtx_CSR_float));
	
	test_results.push_back(test_sell_c_sigma< 8, SIGMA_SORTED>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_sell_c_sigma<16, SIGMA_SORTED>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_sell_c_sigma<32, SIGMA_SORTED>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_sell_c_sigma<64, SIGMA_SORTED>(ite, threads_num, mtx_CSR_float));
	
	test_results.push_back(test_sell_c_sigma_novec< 8, 1>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_sell_c_sigma_novec<16, 1>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_sell_c_sigma_novec<32, 1>(ite, threads_num, mtx_CSR_float));
	test_results.push_back(test_sell_c_sigma_novec<64, 1>(ite, threads_num, mtx_CSR_float));
	
	cout << "------------------------------------------------------" << endl;
	cout << "table_start" << endl;
	const string title_NAME = "NAME";
	const string title_FUNCTION = "FUNCTION";
	const string title_TIME_MS = "TIME_MS";
	const string title_NZ = "NZ";
	const string title_DIFF = "DIFF";
	const string title_CONV_TIME = "CONV_TIME";
	size_t NAME_sz = title_NAME.size();
	size_t FUNCTION_sz = title_FUNCTION.size();
	size_t TIME_MS_sz = title_TIME_MS.size();
	size_t NZ_sz = title_NZ.size();
	size_t DIFF_sz = title_DIFF.size();
	size_t CONV_TIME_sz = title_CONV_TIME.size();
	for (auto res : test_results) {
		NAME_sz      = max(NAME_sz,      res.NAME.size());
		FUNCTION_sz  = max(FUNCTION_sz,  res.FUNCTION.size());
		TIME_MS_sz   = max(TIME_MS_sz,   res.TIME_MS.size());
		NZ_sz        = max(NZ_sz,        res.NZ.size());
		DIFF_sz      = max(DIFF_sz,      res.DIFF.size());
		CONV_TIME_sz = max(CONV_TIME_sz, res.CONV_TIME.size());
 	}
	
	cout << title_NAME << string(NAME_sz - title_NAME.size() + 1, ' ');
	cout << title_FUNCTION << string(FUNCTION_sz - title_FUNCTION.size() + 1, ' ');
	cout << title_TIME_MS << string(TIME_MS_sz - title_TIME_MS.size() + 1, ' ');
	cout << title_NZ << string(NZ_sz - title_NZ.size() + 1, ' ');
	cout << title_DIFF << string(DIFF_sz - title_DIFF.size() + 1, ' ');
	cout << title_CONV_TIME << string(CONV_TIME_sz - title_CONV_TIME.size() + 1, ' ');
	cout << endl;
	for (auto res : test_results) {
		cout << res.NAME      << string(NAME_sz - res.NAME.size() + 1, ' ');
		cout << res.FUNCTION  << string(FUNCTION_sz - res.FUNCTION.size() + 1, ' ');
		cout << res.TIME_MS   << string(TIME_MS_sz - res.TIME_MS.size() + 1, ' ');
		cout << res.NZ        << string(NZ_sz - res.NZ.size() + 1, ' ');
		cout << res.DIFF      << string(DIFF_sz  - res.DIFF.size() + 1, ' ');
		cout << res.CONV_TIME << string(CONV_TIME_sz  - res.DIFF.size() + 1, ' ');
		cout << endl;
 	}
	cout << "table_end" << endl;
	cout << "------------------------------------------------------" << endl;
	
	cout << "END MAIN2_EXE WITH MATRIX " << filename_bin << endl;
	cout << "------------------------------------------------------" << endl;
	cout << endl << endl << endl;
	
 	return 0;
}
