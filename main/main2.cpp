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
#include "../mtx_input/mtx_input.hpp"
#include "../spmv_functions/spmv_functions.hpp"

using namespace std;

// class for measuring time
// MyTimer::SetStartTime();
// ... code ...
// MyTimer::SetEndTime();
// cout << MyTimer::GetDifferenceUs() << "us" << endl;
class MyTimer {
	using myclock = std::chrono::high_resolution_clock;
	static myclock::time_point start_time;
	static myclock::time_point end_time;
public:
	static void SetStartTime() {
		start_time = myclock::now();
	}
	static void SetEndTime() {
		end_time = myclock::now();
	}
	// difference in microseconds (us)
	static long long GetDifferenceUs() {
		return std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
	}
};

MyTimer::myclock::time_point MyTimer::start_time;
MyTimer::myclock::time_point MyTimer::end_time;

// time in microseconds to string of time in ms
string us_to_ms_string(long long time) {
	string s = to_string(time);
	if (s.size() < 4) {
		s = string(4 - s.size(), '0') + s;
	}
	string res = s.substr(0, s.size() - 3) + '.' + s.substr(s.size() - 3, 3);
	return res;
}

// calc max difference of two vectors
double calc_diff(const vector_format& a, const vector_format& b) {
	assert(b.N >= a.N);
	assert(a.N > 0);
	double ans = 0;
	for (int i = 0; i < a.N; i++) {
		ans = max(ans, abs(a.value[i] - b.value[i]));
	}
	return ans;
}

vector_format v, naive_res;
vector<string> results;
constexpr int WARM_UP_CNT = 2; // number of warm up runs before actual measurement

void test_naive(const int ite, const int threads_num, const matrix_CSR& mtx_CSR) {
	naive_res = move(alloc_vector_res(mtx_CSR));
	
	cout << "spmv_naive: ";
	
	// warm up
	for (int it = 0; it < WARM_UP_CNT; it++) {
		spmv_naive_noalloc(mtx_CSR, v, threads_num, naive_res);
	}
	long long res = numeric_limits<long long>::max();
	for (int it = 0; it < ite; it++) {
		MyTimer::SetStartTime();
		spmv_naive_noalloc(mtx_CSR, v, threads_num, naive_res);
		MyTimer::SetEndTime();
		res = min(res, MyTimer::GetDifferenceUs());
	}
	
	cout << res << "us per iteration (minimum)" << endl;
	
	results.push_back("naive");
	results.push_back(to_string(mtx_CSR.row_id[mtx_CSR.N]));
	results.push_back(us_to_ms_string(res));
}

void test_albus(const int ite, const int threads_num, const matrix_CSR& mtx_CSR) {
	int start[100];
	int block_start[100];
	preproc_albus_balance(mtx_CSR, start, block_start, threads_num);
	cout << "-------------------------" << endl;
	cout << "albus precalc" << endl;
	cout << "      start: ";
	for (int i = 0; i <= threads_num; i++) cout << start[i] << " "; cout << "\n";
	cout << "block_start: ";
	for (int i = 0; i <= threads_num; i++) cout << block_start[i] << " "; cout << "\n";
	cout << "-------------------------" << endl;
	
	cout << "spmv_albus_omp: ";
	vector_format albus_omp_res = alloc_vector_res(mtx_CSR);
	// warm up
	for (int it = 0; it < WARM_UP_CNT; it++) {
		spmv_albus_omp_noalloc(mtx_CSR, v, start, block_start, threads_num, albus_omp_res);
	}
	long long res_omp = numeric_limits<long long>::max();
	for (int it = 0; it < ite; it++) {
		MyTimer::SetStartTime();
		spmv_albus_omp_noalloc(mtx_CSR, v, start, block_start, threads_num, albus_omp_res);
		MyTimer::SetEndTime();
		res_omp = min(res_omp, MyTimer::GetDifferenceUs());
	}
	cout << res_omp << "us per iteration (minimum); diff = " << calc_diff(naive_res, albus_omp_res) << endl;
	
	results.push_back("albus");
	results.push_back(to_string(mtx_CSR.row_id[mtx_CSR.N]));
	results.push_back(us_to_ms_string(res_omp));
	
	cout << "spmv_albus_omp_v: ";
	vector_format albus_omp_v_res = alloc_vector_res(mtx_CSR);
	// warm up
	for (int it = 0; it < WARM_UP_CNT; it++) {
		spmv_albus_omp_v_noalloc(mtx_CSR, v, start, block_start, threads_num, albus_omp_v_res);
	}
	long long res_omp_v = numeric_limits<long long>::max();
	for (int it = 0; it < ite; it++) {
		MyTimer::SetStartTime();
		spmv_albus_omp_v_noalloc(mtx_CSR, v, start, block_start, threads_num, albus_omp_v_res);
		MyTimer::SetEndTime();
		res_omp_v = min(res_omp_v, MyTimer::GetDifferenceUs());
	}
	cout << res_omp_v << "us per iteration (minimum); diff = " << calc_diff(naive_res, albus_omp_v_res) << endl;
	
	results.push_back("albus_v");
	results.push_back(to_string(mtx_CSR.row_id[mtx_CSR.N]));
	results.push_back(us_to_ms_string(res_omp_v));
}

constexpr int SIGMA_SORTED = 9'999'999;

template<int C, int sigma>
void test_sell_c_sigma(const int ite, const int threads_num, const matrix_CSR& mtx_CSR, string name) {
	if (C != 4 && C != 8 && C != 16 && C != 32) {
		cout << "test_sell_c_sigma : wrong C == " << C << endl;
		return;
	}
	
	cout << "mtx_SELL_C_sigma<" << C << ", " << sigma << ">: ";
	matrix_SELL_C_sigma<C, sigma> mtx = convert_CSR_to_SELL_C_sigma<C, sigma>(mtx_CSR);
	
	vector_format scs_res = alloc_vector_res(mtx);
	// warm up
	for (int it = 0; it < WARM_UP_CNT; it++) {
		spmv_sell_c_sigma_noalloc(mtx, v, threads_num, scs_res);
	}
	long long res = numeric_limits<long long>::max();
	for (int it = 0; it < ite; it++) {
		MyTimer::SetStartTime();
		spmv_sell_c_sigma_noalloc(mtx, v, threads_num, scs_res);
		MyTimer::SetEndTime();
		res = min(res, MyTimer::GetDifferenceUs());
	}
	
	vector_format real_scs_res = alloc_vector_res(mtx);
	for (int i = 0; i < mtx_CSR.N; i++) {
		real_scs_res.value[mtx.rows_perm[i]] = scs_res.value[i];
	}
	
	cout << res << "us per iteration (minimum); diff = " << calc_diff(naive_res, real_scs_res) << endl;
	
	results.push_back(name);
	results.push_back(to_string(mtx.cs[mtx.N / C]));
	results.push_back(us_to_ms_string(res));
}

template<int C, int sigma>
void test_sell_c_sigmau(const int ite, const int threads_num, const matrix_CSR& mtx_CSR, string name) {
	if (C != 4 && C != 8 && C != 16 && C != 32) {
		cout << "test_sell_c_sigma : wrong C == " << C << endl;
		return;
	}
	
	cout << "mtx_SELL_C_sigma<" << C << ", " << sigma << ">: ";
	matrix_SELL_C_sigma<C, sigma> mtx = convert_CSR_to_SELL_C_sigma<C, sigma>(mtx_CSR);
	
	vector_format scs_res = alloc_vector_res(mtx);
	// warm up
	for (int it = 0; it < WARM_UP_CNT; it++) {
		spmv_sell_c_sigma_noalloc_unroll4(mtx, v, threads_num, scs_res);
	}
	long long res = numeric_limits<long long>::max();
	for (int it = 0; it < ite; it++) {
		MyTimer::SetStartTime();
		spmv_sell_c_sigma_noalloc_unroll4(mtx, v, threads_num, scs_res);
		MyTimer::SetEndTime();
		res = min(res, MyTimer::GetDifferenceUs());
	}
	
	vector_format real_scs_res = alloc_vector_res(mtx);
	for (int i = 0; i < mtx_CSR.N; i++) {
		real_scs_res.value[mtx.rows_perm[i]] = scs_res.value[i];
	}
	
	cout << res << "us per iteration (minimum); diff = " << calc_diff(naive_res, real_scs_res) << endl;
	
	results.push_back(name);
	results.push_back(to_string(mtx.cs[mtx.N / C]));
	results.push_back(us_to_ms_string(res));
}

template<int C, int sigma>
void test_sell_c_sigma_novec(const int ite, const int threads_num, const matrix_CSR& mtx_CSR, string name) {
	if (C != 4 && C != 8 && C != 16 && C != 32) {
		cout << "test_sell_c_sigma_novec : wrong C == " << C << endl;
		return;
	}
	
	cout << "mtx_SELL_C_sigma<" << C << ", " << sigma << "> novec: ";
	matrix_SELL_C_sigma<C, sigma> mtx = convert_CSR_to_SELL_C_sigma<C, sigma>(mtx_CSR);
	
	vector_format scs_res = alloc_vector_res(mtx);
	// warm up
	for (int it = 0; it < WARM_UP_CNT; it++) {
		spmv_sell_c_sigma_noalloc_novec(mtx, v, threads_num, scs_res);
	}
	long long res = numeric_limits<long long>::max();
	for (int it = 0; it < ite; it++) {
		MyTimer::SetStartTime();
		spmv_sell_c_sigma_noalloc_novec(mtx, v, threads_num, scs_res);
		MyTimer::SetEndTime();
		res = min(res, MyTimer::GetDifferenceUs());
	}
	
	vector_format real_scs_res = alloc_vector_res(mtx);
	for (int i = 0; i < mtx_CSR.N; i++) {
		real_scs_res.value[mtx.rows_perm[i]] = scs_res.value[i];
	}
	
	cout << res << "us per iteration (minimum); diff = " << calc_diff(naive_res, real_scs_res) << endl;
	
	results.push_back(name);
	results.push_back(to_string(mtx.cs[mtx.N / C]));
	results.push_back(us_to_ms_string(res));
}

int main(int argc, char** argv) {
	
	cout << "------------------------------------------------------" << endl;
	
	if (argc <= 1) {
		cout << "error : no name of file with matrix!" << endl;
		return -1;
	}
	
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
	cout << "threads_num: " << threads_num << endl;
	
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
	cout << "ite: " << ite << endl;
	
	char* filename = argv[1];
	matrix_CSR mtx_CSR;
	try {
		mtx_CSR = read_MTX_as_CSR(filename);
	}
	catch (...) {
		cout << "error in reading matrix " << filename << endl;
		return -1;
	}
	transpose_CSR(mtx_CSR);
	transpose_CSR(mtx_CSR);
	cout << "matrix: " << filename << endl;
 	cout << "N: " << mtx_CSR.N << " M: " << mtx_CSR.M << endl;
	cout << "nz: " << mtx_CSR.row_id[mtx_CSR.N] << endl;
	int N = mtx_CSR.N;
	int M = mtx_CSR.M;
	cout << "-------------------------" << endl;
	
	// INIT v
	v.alloc(M, 32);
	v.N = M;
	v.value = new double[v.N];
	mt19937_64 gen(38);
	uniform_real_distribution<> dis(0.0, 1.0);
	for (int i = 0; i < v.N; i++) {
		v.value[i] = dis(gen);
	}
	cout << "vector v[" << v.N << "]: ";
	for (int i = 0; i < 10; i++) {
		cout << v.value[i] << " ";
	}
	cout << "..." << endl;
	cout << "-------------------------" << endl;
	
	
	
	test_naive(ite, threads_num, mtx_CSR);
	
	test_albus(ite, threads_num, mtx_CSR);
	
	test_sell_c_sigma< 4, 1>(ite, threads_num, mtx_CSR, "scs_4_1");
	test_sell_c_sigmau< 4, 1>(ite, threads_num, mtx_CSR, "scs_4_1_u4");
	test_sell_c_sigma< 8, 1>(ite, threads_num, mtx_CSR, "scs_8_1");
	test_sell_c_sigmau< 8, 1>(ite, threads_num, mtx_CSR, "scs_8_1_u4");
	test_sell_c_sigma<16, 1>(ite, threads_num, mtx_CSR, "scs16_1");
	test_sell_c_sigmau<16, 1>(ite, threads_num, mtx_CSR, "scs16_1_u4");
	test_sell_c_sigma<32, 1>(ite, threads_num, mtx_CSR, "scs32_1");
	test_sell_c_sigmau<32, 1>(ite, threads_num, mtx_CSR, "scs32_1_u4");
	test_sell_c_sigma< 4, 2>(ite, threads_num, mtx_CSR, "scs_4_2");
	test_sell_c_sigmau< 4, 2>(ite, threads_num, mtx_CSR, "scs_4_2_u4");
	test_sell_c_sigma< 8, 2>(ite, threads_num, mtx_CSR, "scs_8_2");
	test_sell_c_sigmau< 8, 2>(ite, threads_num, mtx_CSR, "scs_8_2_u4");
	test_sell_c_sigma<16, 2>(ite, threads_num, mtx_CSR, "scs16_2");
	test_sell_c_sigmau<16, 2>(ite, threads_num, mtx_CSR, "scs16_2_u4");
	test_sell_c_sigma<32, 2>(ite, threads_num, mtx_CSR, "scs32_2");
	test_sell_c_sigmau<32, 2>(ite, threads_num, mtx_CSR, "scs32_2_u4");
	test_sell_c_sigma< 4, 4>(ite, threads_num, mtx_CSR, "scs_4_4");
	test_sell_c_sigmau< 4, 4>(ite, threads_num, mtx_CSR, "scs_4_4_u4");
	test_sell_c_sigma< 8, 4>(ite, threads_num, mtx_CSR, "scs_8_4");
	test_sell_c_sigmau< 8, 4>(ite, threads_num, mtx_CSR, "scs_8_4_u4");
	test_sell_c_sigma<16, 4>(ite, threads_num, mtx_CSR, "scs16_4");
	test_sell_c_sigmau<16, 4>(ite, threads_num, mtx_CSR, "scs16_4_u4");
	test_sell_c_sigma<32, 4>(ite, threads_num, mtx_CSR, "scs32_4");
	test_sell_c_sigmau<32, 4>(ite, threads_num, mtx_CSR, "scs32_4_u4");
	test_sell_c_sigma< 4, 8>(ite, threads_num, mtx_CSR, "scs_4_8");
	test_sell_c_sigmau< 4, 8>(ite, threads_num, mtx_CSR, "scs_4_8_u4");
	test_sell_c_sigma< 8, 8>(ite, threads_num, mtx_CSR, "scs_8_8");
	test_sell_c_sigmau< 8, 8>(ite, threads_num, mtx_CSR, "scs_8_8_u4");
	test_sell_c_sigma<16, 8>(ite, threads_num, mtx_CSR, "scs16_8");
	test_sell_c_sigmau<16, 8>(ite, threads_num, mtx_CSR, "scs16_8_u4");
	test_sell_c_sigma<32, 8>(ite, threads_num, mtx_CSR, "scs32_8");
	test_sell_c_sigmau<32, 8>(ite, threads_num, mtx_CSR, "scs32_8_u4");
	
	test_sell_c_sigma_novec< 4, 1>(ite, threads_num, mtx_CSR, "scs_4_1_nv");
	test_sell_c_sigma_novec< 8, 1>(ite, threads_num, mtx_CSR, "scs_8_1_nv");
	test_sell_c_sigma_novec<16, 1>(ite, threads_num, mtx_CSR, "scs16_1_nv");
	test_sell_c_sigma_novec<32, 1>(ite, threads_num, mtx_CSR, "scs32_1_nv");
	
	test_sell_c_sigma< 4, SIGMA_SORTED>(ite, threads_num, mtx_CSR, "scs_4_S");
	test_sell_c_sigmau< 4, SIGMA_SORTED>(ite, threads_num, mtx_CSR, "scs_4_S_u4");
	test_sell_c_sigma< 8, SIGMA_SORTED>(ite, threads_num, mtx_CSR, "scs_8_S");
	test_sell_c_sigmau< 8, SIGMA_SORTED>(ite, threads_num, mtx_CSR, "scs_8_S_u4");
	test_sell_c_sigma<16, SIGMA_SORTED>(ite, threads_num, mtx_CSR, "scs16_S");
	test_sell_c_sigmau<16, SIGMA_SORTED>(ite, threads_num, mtx_CSR, "scs16_S_u4");
	test_sell_c_sigma<32, SIGMA_SORTED>(ite, threads_num, mtx_CSR, "scs32_S");
	test_sell_c_sigmau<32, SIGMA_SORTED>(ite, threads_num, mtx_CSR, "scs32_S_u4");
	
	cout << "-------------------------" << endl;
	
	cout << "data2 ";
	cout << "mtx " << filename << " ";
	cout << "threads " << threads_num << " ";
	cout << "ite " << ite << " ";
	for (auto& i : results) {
		cout << i << " ";
	}
	cout << endl;
	
 	return 0;
}
