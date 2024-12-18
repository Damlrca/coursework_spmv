// Copyright (C) 2024 Sadikov Damir
// github.com/Damlrca/coursework_spmv

#include <iostream>
#include <chrono>
#include <set>
#include <random>
#include <omp.h>
#include <cstdlib>

#include "../storage_formats/storage_formats.hpp"
#include "../mtx_input/mtx_input.hpp"
#include "../spmv_functions/spmv_functions.hpp"

using namespace std;

// class for measuring time
// MyTimer::SetStartTime();
// ... code ...
// MyTimer::SetEndTime();
// usage:
// cout << MyTimer::GetDifferenceMs() << "ms" << endl;
class MyTimer {
	using myclock = std::chrono::system_clock;
	static myclock::time_point start_time;
	static myclock::time_point end_time;
public:
	static void SetStartTime() {
		start_time = myclock::now();
	}
	static void SetEndTime() {
		end_time = myclock::now();
	}
	static long long GetDifferenceMs() {
		return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
	}
};

MyTimer::myclock::time_point MyTimer::start_time;
MyTimer::myclock::time_point MyTimer::end_time;

random_device rd;
mt19937 gen(rd());

set<int> random_sample(int n, int m) {
	set<int> res;
	for (int r = n - m; r < n; r++) {
		int i = uniform_int_distribution<int>(0, r)(gen);
		if (res.count(i))
			res.insert(r);
		else
			res.insert(i);
	}
	return res;
}

vector_format v;
double mx_diff_albus_omp = 0;
double mx_diff_albus_omp_v = 0;
double mx_diff_scs81 = 0;
double mx_diff_scs41 = 0;
vector_format naive_res;
long long spmv_naive_result;
long long spmv_albus_omp_result;
long long spmv_albus_omp_v_result;
long long spmv_sell_8_1_result;
long long spmv_sell_4_1_result;

double calc_diff(const vector_format& a, const vector_format& b) {
	double ans = 0;
	for (int i = 0; i < a.N; i++) {
		ans = max(ans, abs(a.value[i] - b.value[i]));
	}
	return ans;
}

void test_naive(const int ite, const int threads_num, const matrix_CSR& mtx_CSR) {
	naive_res = move(alloc_vector_res(mtx_CSR));
	// warm up
	for (int it = 0; it < 5; it++) {
		spmv_naive_noalloc(mtx_CSR, v, threads_num, naive_res);
	}
	MyTimer::SetStartTime();
	for (int it = 0; it < ite; it++) {
		spmv_naive_noalloc(mtx_CSR, v, threads_num, naive_res);
	}
	MyTimer::SetEndTime();
	spmv_naive_result = MyTimer::GetDifferenceMs();
	cout << "spmv_naive: " << spmv_naive_result / ite << "ms per iteration" << endl;
	cout << "(" << spmv_naive_result << "ms for all iterations)" << endl;
	cout << "-------------------------" << endl;
}

void test_albus(const int ite, const int threads_num, const matrix_CSR& mtx_CSR) {
	int start[100];
	int block_start[100];
	preproc_albus_balance(mtx_CSR, start, block_start, threads_num);
	cout << "albus precalc" << endl;
	cout << "      start: ";
	for (int i = 0; i <= threads_num; i++) cout << start[i] << " "; cout << "\n";
	cout << "block_start: ";
	for (int i = 0; i <= threads_num; i++) cout << block_start[i] << " "; cout << "\n";
	cout << "-------------------------" << endl;
	
	vector_format albus_omp_res = alloc_vector_res(mtx_CSR);
	// warm up
	for (int it = 0; it < 5; it++) {
		spmv_albus_omp_noalloc(mtx_CSR, v, start, block_start, threads_num, albus_omp_res);
	}
	MyTimer::SetStartTime();
	for (int it = 0; it < ite; it++) {
		spmv_albus_omp_noalloc(mtx_CSR, v, start, block_start, threads_num, albus_omp_res);
	}
	MyTimer::SetEndTime();
	spmv_albus_omp_result = MyTimer::GetDifferenceMs();
	cout << "spmv_albus_omp: " << spmv_albus_omp_result / ite << "ms per iteration" << endl;
	cout << "(" << spmv_albus_omp_result << "ms for all iterations)" << endl;
	cout << "-------------------------" << endl;
	
	vector_format albus_omp_v_res = alloc_vector_res(mtx_CSR);
	// warm up
	for (int it = 0; it < 5; it++) {
		spmv_albus_omp_v_noalloc(mtx_CSR, v, start, block_start, threads_num, albus_omp_v_res);
	}
	MyTimer::SetStartTime();
	for (int it = 0; it < ite; it++) {
		spmv_albus_omp_v_noalloc(mtx_CSR, v, start, block_start, threads_num, albus_omp_v_res);
	}
	MyTimer::SetEndTime();
	spmv_albus_omp_v_result = MyTimer::GetDifferenceMs();
	cout << "spmv_albus_omp_v: " << spmv_albus_omp_v_result / ite << "ms per iteration" << endl;
	cout << "(" << spmv_albus_omp_v_result << "ms for all iterations)" << endl;
	cout << "-------------------------" << endl;
	
	// calc mx_diff !
	mx_diff_albus_omp = calc_diff(naive_res, albus_omp_res);
	mx_diff_albus_omp_v = calc_diff(naive_res, albus_omp_v_res);
}

void test_sell_c_sigma(const int ite, const int threads_num, const matrix_CSR& mtx_CSR) {
	cout << "mtx_SELL_C_sigma<8, 1>" << endl;
	matrix_SELL_C_sigma<8, 1> mtx_SELL_8_1 = convert_CSR_to_SELL_C_sigma<8, 1>(mtx_CSR);
	
	vector_format scs_8_1_res = alloc_vector_res(mtx_SELL_8_1);
	// warm up
	for (int it = 0; it < 5; it++) {
		spmv_sell_c_sigma_noalloc(mtx_SELL_8_1, v, threads_num, scs_8_1_res);
	}
	MyTimer::SetStartTime();
	for (int it = 0; it < ite; it++) {
		spmv_sell_c_sigma_noalloc(mtx_SELL_8_1, v, threads_num, scs_8_1_res);
	}
	MyTimer::SetEndTime();
	spmv_sell_8_1_result = MyTimer::GetDifferenceMs();
	cout << "mtx_SELL_C_sigma<8, 1>: " << spmv_sell_8_1_result / ite << "ms per iteration" << endl;
	cout << "(" << spmv_sell_8_1_result << "ms for all iterations)" << endl;
	cout << "-------------------------" << endl;
	
	
	cout << "mtx_SELL_C_sigma<4, 1>" << endl;
	matrix_SELL_C_sigma<4, 1> mtx_SELL_4_1 = convert_CSR_to_SELL_C_sigma<4, 1>(mtx_CSR);
	
	vector_format scs_4_1_res = alloc_vector_res(mtx_SELL_4_1);
	// warm up
	for (int it = 0; it < 5; it++) {
		spmv_sell_c_sigma_noalloc(mtx_SELL_4_1, v, threads_num, scs_4_1_res);
	}	
	MyTimer::SetStartTime();
	for (int it = 0; it < ite; it++) {
		spmv_sell_c_sigma_noalloc(mtx_SELL_4_1, v, threads_num, scs_4_1_res);
	}
	MyTimer::SetEndTime();
	spmv_sell_4_1_result = MyTimer::GetDifferenceMs();
	cout << "mtx_SELL_C_sigma<4, 1>: " << spmv_sell_4_1_result / ite << "ms per iteration" << endl;
	cout << "(" << spmv_sell_4_1_result << "ms for all iterations)" << endl;
	cout << "-------------------------" << endl;
	
	// calc mx_diff !
	mx_diff_scs81 = calc_diff(naive_res, scs_8_1_res);
	mx_diff_scs41 = calc_diff(naive_res, scs_4_1_res);
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
	cout << "-------------------------" << endl;
	
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
	cout << "-------------------------" << endl;
	
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
	
	
	
	test_naive(ite, threads_num, mtx_CSR);
	
	test_albus(ite, threads_num, mtx_CSR);
	
	test_sell_c_sigma(ite, threads_num, mtx_CSR);
	
	
	
	cout << "mx_diff_albus_omp   : " << mx_diff_albus_omp << endl;
	cout << "mx_diff_albus_omp_v : " << mx_diff_albus_omp_v << endl;
	cout << "mx_diff_scs81       : " << mx_diff_scs81 << endl;
	cout << "mx_diff_scs41       : " << mx_diff_scs41 << endl;
	
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
	cout << "             threads_num: " << threads_num << endl;
	cout << "                     ite: " << ite << endl;
	cout << "                  matrix: " << filename << endl;
	cout << "              spmv_naive: " << spmv_naive_result / ite << "ms per iteration" << endl;
	cout << "          spmv_albus_omp: " << spmv_albus_omp_result / ite << "ms per iteration" << endl;
	cout << "        spmv_albus_omp_v: " << spmv_albus_omp_v_result / ite << "ms per iteration" << endl;
	cout << "    spmv_sell_8_1_result: " << spmv_sell_8_1_result / ite << "ms per iteration" << endl;
	cout << "    spmv_sell_4_1_result: " << spmv_sell_4_1_result / ite << "ms per iteration" << endl;
	cout << "!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
	
 	return 0;
}
