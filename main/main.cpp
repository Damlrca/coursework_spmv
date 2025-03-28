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

vector_format v;
double mx_diff_albus_omp = 0;
double mx_diff_albus_omp_v = 0;
double mx_diff_scs__4_1 = 0;
double mx_diff_scs__8_1 = 0;
double mx_diff_scs_16_1 = 0;
double mx_diff_scs_32_1 = 0;
double mx_diff_scs__4_1_novec = 0;
double mx_diff_scs__8_1_novec = 0;
double mx_diff_scs_16_1_novec = 0;
double mx_diff_scs_32_1_novec = 0;
double mx_diff_scs__4_sorted = 0;
double mx_diff_scs__8_sorted = 0;
double mx_diff_scs_16_sorted = 0;
double mx_diff_scs_32_sorted = 0;
vector_format naive_res;
long long spmv_naive_time = numeric_limits<long long>::max();
long long spmv_albus_omp_time = numeric_limits<long long>::max();
long long spmv_albus_omp_v_time = numeric_limits<long long>::max();
long long spmv_scs__4_1_time = numeric_limits<long long>::max();
long long spmv_scs__8_1_time = numeric_limits<long long>::max();
long long spmv_scs_16_1_time = numeric_limits<long long>::max();
long long spmv_scs_32_1_time = numeric_limits<long long>::max();
long long spmv_scs__4_sorted_time = numeric_limits<long long>::max();
long long spmv_scs__8_sorted_time = numeric_limits<long long>::max();
long long spmv_scs_16_sorted_time = numeric_limits<long long>::max();
long long spmv_scs_32_sorted_time = numeric_limits<long long>::max();
long long spmv_scs__4_1_novec_time = numeric_limits<long long>::max();
long long spmv_scs__8_1_novec_time = numeric_limits<long long>::max();
long long spmv_scs_16_1_novec_time = numeric_limits<long long>::max();
long long spmv_scs_32_1_novec_time = numeric_limits<long long>::max();

// calc max difference of two vectors
double calc_diff(const vector_format& a, const vector_format& b) {
	double ans = 0;
	for (int i = 0; i < a.N; i++) {
		ans = max(ans, abs(a.value[i] - b.value[i]));
	}
	return ans;
}

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
	spmv_naive_time = res;
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
	cout << res_omp << "us per iteration (minimum)" << endl;
	
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
	cout << res_omp_v << "us per iteration (minimum)" << endl;
	
	spmv_albus_omp_time = res_omp;
	spmv_albus_omp_v_time = res_omp_v;
	mx_diff_albus_omp = calc_diff(naive_res, albus_omp_res);
	mx_diff_albus_omp_v = calc_diff(naive_res, albus_omp_v_res);
}

constexpr int SIGMA_SORTED = 9'999'999;

template<int C, int sigma>
void test_sell_c_sigma(const int ite, const int threads_num, const matrix_CSR& mtx_CSR, long long& result_time, double& result_diff) {
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
	
	cout << res << "us per iteration (minimum)" << endl;
	
	result_time = res;
	result_diff = calc_diff(naive_res, real_scs_res);
}

template<int C, int sigma>
void test_sell_c_sigma_novec(const int ite, const int threads_num, const matrix_CSR& mtx_CSR, long long& result_time, double& result_diff) {
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
	
	cout << res << "us per iteration (minimum)" << endl;
	
	result_time = res;
	result_diff = calc_diff(naive_res, real_scs_res);
}

template<int C, int sigma>
void print_mtx_stat_scs(const matrix_CSR& mtx_CSR) {
	matrix_SELL_C_sigma<C, 1> mtx = convert_CSR_to_SELL_C_sigma<C, 1>(mtx_CSR);
	matrix_SELL_C_sigma<C, SIGMA_SORTED> mtx_sorted = convert_CSR_to_SELL_C_sigma<C, SIGMA_SORTED>(mtx_CSR);
	
	cout << "mtx_scs" << C << " : N=" << mtx.N << " M=" << mtx.M << " nz=" << mtx.cs[mtx.N / C] << endl;
	cout << "mtx_scs" << C << "S: N=" << mtx_sorted.N << " M=" << mtx_sorted.M << " nz=" << mtx_sorted.cs[mtx_sorted.N / C] << endl;
}

void print_mtx_stat(const matrix_CSR& mtx_CSR) {
	cout << "mtx_CSR  : N=" << mtx_CSR.N << " M=" << mtx_CSR.M << " nz=" << mtx_CSR.row_id[mtx_CSR.N] << endl;
	print_mtx_stat_scs<4, 1>(mtx_CSR);
	print_mtx_stat_scs<8, 1>(mtx_CSR);
	print_mtx_stat_scs<16, 1>(mtx_CSR);
	print_mtx_stat_scs<32, 1>(mtx_CSR);
}

// time in microseconds to string of time in ms
string us_to_ms_string(long long time) {
	string s = to_string(time);
	if (s.size() < 4) {
		s = string(4 - s.size(), '0') + s;
	}
	string res = s.substr(0, s.size() - 3) + '.' + s.substr(s.size() - 3, 3);
	return res;
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
	//set<int> rs = random_sample(M, M);
	v.N = M;
	v.value = new double[v.N];
	//auto rs_i = rs.begin();
	for (int i = 0; i < v.N; i++) {
		//v.value[i] = (double)*rs_i;
		//++rs_i;
		v.value[i] = 1;
	}
	cout << "vector v[" << v.N << "]: ";
	for (int i = 0; i < 10; i++) {
		cout << v.value[i] << " ";
	}
	cout << "..." << endl;
	cout << "-------------------------" << endl;
	
	
	
	test_naive(ite, threads_num, mtx_CSR);
	
	test_albus(ite, threads_num, mtx_CSR);
	
	test_sell_c_sigma< 4, 1>(ite, threads_num, mtx_CSR, spmv_scs__4_1_time, mx_diff_scs__4_1);
	test_sell_c_sigma< 8, 1>(ite, threads_num, mtx_CSR, spmv_scs__8_1_time, mx_diff_scs__8_1);
	test_sell_c_sigma<16, 1>(ite, threads_num, mtx_CSR, spmv_scs_16_1_time, mx_diff_scs_16_1);
	test_sell_c_sigma<32, 1>(ite, threads_num, mtx_CSR, spmv_scs_32_1_time, mx_diff_scs_32_1);
	
	test_sell_c_sigma_novec< 4, 1>(ite, threads_num, mtx_CSR, spmv_scs__4_1_novec_time, mx_diff_scs__4_1_novec);
	test_sell_c_sigma_novec< 8, 1>(ite, threads_num, mtx_CSR, spmv_scs__8_1_novec_time, mx_diff_scs__8_1_novec);
	test_sell_c_sigma_novec<16, 1>(ite, threads_num, mtx_CSR, spmv_scs_16_1_novec_time, mx_diff_scs_16_1_novec);
	test_sell_c_sigma_novec<32, 1>(ite, threads_num, mtx_CSR, spmv_scs_32_1_novec_time, mx_diff_scs_32_1_novec);
	
	test_sell_c_sigma< 4, SIGMA_SORTED>(ite, threads_num, mtx_CSR, spmv_scs__4_sorted_time, mx_diff_scs__4_sorted);
	test_sell_c_sigma< 8, SIGMA_SORTED>(ite, threads_num, mtx_CSR, spmv_scs__8_sorted_time, mx_diff_scs__8_sorted);
	test_sell_c_sigma<16, SIGMA_SORTED>(ite, threads_num, mtx_CSR, spmv_scs_16_sorted_time, mx_diff_scs_16_sorted);
	test_sell_c_sigma<32, SIGMA_SORTED>(ite, threads_num, mtx_CSR, spmv_scs_32_sorted_time, mx_diff_scs_32_sorted);
	
	cout << "-------------------------" << endl;
	
	
	cout << "mx_diff_albus_omp      : " << mx_diff_albus_omp << endl;
	cout << "mx_diff_albus_omp_v    : " << mx_diff_albus_omp_v << endl;
	cout << "mx_diff_scs__4_1       : " << mx_diff_scs__4_1 << endl;
	cout << "mx_diff_scs__8_1       : " << mx_diff_scs__8_1 << endl;
	cout << "mx_diff_scs_16_1       : " << mx_diff_scs_16_1 << endl;
	cout << "mx_diff_scs_32_1       : " << mx_diff_scs_32_1 << endl;
	cout << "mx_diff_scs__4_1_novec : " << mx_diff_scs__4_1_novec << endl;
	cout << "mx_diff_scs__8_1_novec : " << mx_diff_scs__8_1_novec << endl;
	cout << "mx_diff_scs_16_1_novec : " << mx_diff_scs_16_1_novec << endl;
	cout << "mx_diff_scs_32_1_novec : " << mx_diff_scs_32_1_novec << endl;
	cout << "mx_diff_scs__4_sorted  : " << mx_diff_scs__4_sorted << endl;
	cout << "mx_diff_scs__8_sorted  : " << mx_diff_scs__8_sorted << endl;
	cout << "mx_diff_scs_16_sorted  : " << mx_diff_scs_16_sorted << endl;
	cout << "mx_diff_scs_32_sorted  : " << mx_diff_scs_32_sorted << endl;
	
	/*
	int cnt = 0;
	for (int i = 0; i < res_naive.N; i++) {
		if (res_naive.value[i] != res_albus.value[i]) {
			if (cnt < 10) {
				cout << i << " ";
				cout << res_naive.value[i] << " " << res_albus.value[i] << " ";
				cout << abs(res_naive.value[i] - res_albus.value[i]) << endl;
				// cout << "values in i-th row: ";
				// for (int j = mtx_CSR.row_id[i]; j < mtx_CSR.row_id[i + 1]; j++) {
				// 	cout << mtx_CSR.value[j] << " ";
				// }
				// cout << endl;
			}
			else if (cnt == 10) {
				cout << "... ..." << endl;
			}
			cnt++;	
		}
 	}
	cout << "cnt: " << cnt << endl;
	*/
	
	cout << "-------------------------" << endl;
	cout << "!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
	cout << "threads_num : " << threads_num << endl;
	cout << "ite         : " << ite << endl;
	cout << "matrix      : " << filename << endl;
	
	print_mtx_stat(mtx_CSR);
	
	cout << "spmv_naive             : " << us_to_ms_string(spmv_naive_time) << "ms per iteration (minimum)" << endl;
	cout << "spmv_albus_omp         : " << us_to_ms_string(spmv_albus_omp_time) << "ms per iteration (minimum)" << endl;
	cout << "spmv_albus_omp_v       : " << us_to_ms_string(spmv_albus_omp_v_time) << "ms per iteration (minimum)" << endl;
	cout << "spmv_scs__4_1          : " << us_to_ms_string(spmv_scs__4_1_time) << "ms per iteration (minimum)" << endl;
	cout << "spmv_scs__8_1          : " << us_to_ms_string(spmv_scs__8_1_time) << "ms per iteration (minimum)" << endl;
	cout << "spmv_scs_16_1          : " << us_to_ms_string(spmv_scs_16_1_time) << "ms per iteration (minimum)" << endl;
	cout << "spmv_scs_32_1          : " << us_to_ms_string(spmv_scs_32_1_time) << "ms per iteration (minimum)" << endl;
	cout << "spmv_scs__4_1_novec    : " << us_to_ms_string(spmv_scs__4_1_novec_time) << "ms per iteration (minimum)" << endl;
	cout << "spmv_scs__8_1_novec    : " << us_to_ms_string(spmv_scs__8_1_novec_time) << "ms per iteration (minimum)" << endl;
	cout << "spmv_scs_16_1_novec    : " << us_to_ms_string(spmv_scs_16_1_novec_time) << "ms per iteration (minimum)" << endl;
	cout << "spmv_scs_32_1_novec    : " << us_to_ms_string(spmv_scs_32_1_novec_time) << "ms per iteration (minimum)" << endl;
	cout << "spmv_scs__4_sorted     : " << us_to_ms_string(spmv_scs__4_sorted_time) << "ms per iteration (minimum)" << endl;
	cout << "spmv_scs__8_sorted     : " << us_to_ms_string(spmv_scs__8_sorted_time) << "ms per iteration (minimum)" << endl;
	cout << "spmv_scs_16_sorted     : " << us_to_ms_string(spmv_scs_16_sorted_time) << "ms per iteration (minimum)" << endl;
	cout << "spmv_scs_32_sorted     : " << us_to_ms_string(spmv_scs_32_sorted_time) << "ms per iteration (minimum)" << endl;
	
	vector<string> RES;
	RES.push_back("data");
	RES.push_back("mtx");
	RES.push_back(filename);
	RES.push_back("threads");
	RES.push_back(to_string(threads_num));
	RES.push_back("ite");
	RES.push_back(to_string(ite));
	RES.push_back("naive");
	RES.push_back(us_to_ms_string(spmv_naive_time));
	RES.push_back("albus_omp");
	RES.push_back(us_to_ms_string(spmv_albus_omp_time));
	RES.push_back("albus_omp_v");
	RES.push_back(us_to_ms_string(spmv_albus_omp_v_time));
	RES.push_back("scs");
	RES.push_back(us_to_ms_string(spmv_scs__4_1_time));
	RES.push_back(us_to_ms_string(spmv_scs__8_1_time));
	RES.push_back(us_to_ms_string(spmv_scs_16_1_time));
	RES.push_back(us_to_ms_string(spmv_scs_32_1_time));
	RES.push_back("scs_novec");
	RES.push_back(us_to_ms_string(spmv_scs__4_1_novec_time));
	RES.push_back(us_to_ms_string(spmv_scs__8_1_novec_time));
	RES.push_back(us_to_ms_string(spmv_scs_16_1_novec_time));
	RES.push_back(us_to_ms_string(spmv_scs_32_1_novec_time));
	RES.push_back("scs_sorted");
	RES.push_back(us_to_ms_string(spmv_scs__4_sorted_time));
	RES.push_back(us_to_ms_string(spmv_scs__8_sorted_time));
	RES.push_back(us_to_ms_string(spmv_scs_16_sorted_time));
	RES.push_back(us_to_ms_string(spmv_scs_32_sorted_time));
	for (auto& i : RES) {
		cout << i << " ";
	}
	cout << endl;
	cout << "!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
	
 	return 0;
}
