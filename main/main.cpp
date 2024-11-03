// Copyright (C) 2024 Sadikov Damir
// github.com/Damlrca/coursework_spmv

#include <iostream>
#include <chrono>
#include <set>
#include <random>
#include <omp.h>

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

int main(int argc, char** argv) {
	
	cout << "------------------------------------------------------" << endl;
	
	if (argc <= 1) {
		cout << "error : no name of file with matrix!" << endl;
		return -1;
	}
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
	
	//set<int> rs = random_sample(M, M);
	vector_format v;
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
	
	int threads_num = omp_get_max_threads();
	omp_set_num_threads(threads_num);
	cout << "threads_num: " << threads_num << endl;
	cout << "-------------------------" << endl;
	
	int ite = 1000;
	cout << "ite: " << ite << endl;
	cout << "-------------------------" << endl;
	
	// warm up
	for (int it = 0; it < 5; it++) {
		vector_format res = spmv_naive(mtx_CSR, v, threads_num);
	}
	MyTimer::SetStartTime();
	for (int it = 0; it < ite; it++) {
		vector_format res = spmv_naive(mtx_CSR, v, threads_num);
	}
	MyTimer::SetEndTime();
	int spmv_naive_result = MyTimer::GetDifferenceMs();
	cout << "spmv_naive: " << spmv_naive_result / ite << "ms per iteration" << endl;
	cout << "(" << spmv_naive_result << "ms for all iterations)" << endl;
	cout << "-------------------------" << endl;
	
	int start[100];
	int block_start[100];
	preproc_albus_balance(mtx_CSR, start, block_start, threads_num);
	cout << "albus precalc" << endl;
	cout << "      start: ";
	for (int i = 0; i <= threads_num; i++) cout << start[i] << " "; cout << "\n";
	cout << "block_start: ";
	for (int i = 0; i <= threads_num; i++) cout << block_start[i] << " "; cout << "\n";
	cout << "-------------------------" << endl;
	
	// warm up
	for (int it = 0; it < 5; it++) {
		vector_format res = spmv_albus_omp(mtx_CSR, v, start, block_start, threads_num);
	}
	MyTimer::SetStartTime();
	for (int it = 0; it < ite; it++) {
		vector_format res = spmv_albus_omp(mtx_CSR, v, start, block_start, threads_num);
	}
	MyTimer::SetEndTime();
	int spmv_albus_omp_result = MyTimer::GetDifferenceMs();
	cout << "spmv_albus_omp: " << spmv_albus_omp_result / ite << "ms per iteration" << endl;
	cout << "(" << spmv_albus_omp_result << "ms for all iterations)" << endl;
	cout << "-------------------------" << endl;
	
	// warm up
	for (int it = 0; it < 5; it++) {
		vector_format res = spmv_albus_omp_v(mtx_CSR, v, start, block_start, threads_num);
	}
	MyTimer::SetStartTime();
	for (int it = 0; it < ite; it++) {
		vector_format res = spmv_albus_omp_v(mtx_CSR, v, start, block_start, threads_num);
	}
	MyTimer::SetEndTime();
	int spmv_albus_omp_v_result = MyTimer::GetDifferenceMs();
	cout << "spmv_albus_omp_v: " << spmv_albus_omp_v_result / ite << "ms per iteration" << endl;
	cout << "(" << spmv_albus_omp_v_result << "ms for all iterations)" << endl;
	cout << "-------------------------" << endl;
	
	
	cout << "mtx_SELL_C_sigma" << endl;
	matrix_SELL_C_sigma mtx_SELL_C_sigma = convert_CSR_to_SELL_C_sigma(mtx_CSR);
	
	for (int it = 0; it < 5; it++) {
		vector_format res = spmv_sell_c_sigma(mtx_SELL_C_sigma, v, threads_num);
	}
	MyTimer::SetStartTime();
	for (int it = 0; it < ite; it++) {
		vector_format res = spmv_sell_c_sigma(mtx_SELL_C_sigma, v, threads_num);
	}
	MyTimer::SetEndTime();
	int spmv_sell_c_sigma_result = MyTimer::GetDifferenceMs();
	cout << "spmv_sell_c_sigma: " << spmv_sell_c_sigma_result / ite << "ms per iteration" << endl;
	cout << "(" << spmv_sell_c_sigma_result << "ms for all iterations)" << endl;
	cout << "-------------------------" << endl;
	
	vector_format res_naive = spmv_naive(mtx_CSR, v, threads_num);
	vector_format res_albus = spmv_albus_omp(mtx_CSR, v, start, block_start, threads_num);
	vector_format res_albus_v = spmv_albus_omp_v(mtx_CSR, v, start, block_start, threads_num);
	vector_format res_scs = spmv_sell_c_sigma(mtx_SELL_C_sigma, v, threads_num);
	
	double mx_diff = 0;
	for (int i = 0; i < res_naive.N; i++) {
		mx_diff = max(mx_diff, abs(res_naive.value[i] - res_albus.value[i]));
	}
	cout << "mx_diff: " << mx_diff << endl;
	
	double mx_diff_2 = 0;
	for (int i = 0; i < res_naive.N; i++) {
		mx_diff_2 = max(mx_diff_2, abs(res_naive.value[i] - res_albus_v.value[i]));
	}
	cout << "mx_diff_2: " << mx_diff_2 << endl;
	
	double mx_diff_3 = 0;
	for (int i = 0; i < res_naive.N; i++) {
		mx_diff_3 = max(mx_diff_3, abs(res_naive.value[i] - res_scs.value[i]));
	}
	cout << "mx_diff_3: " << mx_diff_3 << endl;
	
	int cnt = 0;
	for (int i = 0; i < res_naive.N; i++) {
		if (res_naive.value[i] != res_albus.value[i]) {
			if (cnt < 10) {
				cout << i << " ";
				cout << res_naive.value[i] << " " << res_albus.value[i] << " ";
				cout << abs(res_naive.value[i] - res_albus.value[i]) << endl;
				/*
				cout << "values in i-th row: ";
				for (int j = mtx_CSR.row_id[i]; j < mtx_CSR.row_id[i + 1]; j++) {
					cout << mtx_CSR.value[j] << " ";
				}
				cout << endl;
				*/
			}
			else if (cnt == 10) {
				cout << "... ..." << endl;
			}
			cnt++;	
		}
 	}
	cout << "cnt: " << cnt << endl;
	
	cout << "-------------------------" << endl;
	cout << "!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
	cout << "                  matrix: " << filename << endl;
	cout << "              spmv_naive: " << spmv_naive_result / ite << "ms per iteration" << endl;
	cout << "          spmv_albus_omp: " << spmv_albus_omp_result / ite << "ms per iteration" << endl;
	cout << "        spmv_albus_omp_v: " << spmv_albus_omp_v_result / ite << "ms per iteration" << endl;
	cout << "spmv_sell_c_sigma_result: " << spmv_sell_c_sigma_result / ite << "ms per iteration" << endl;
	cout << "!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
	
	/*
	cout << "-------------------------" << endl;
	cout << "test sell-c-sigma" << endl;
	
	matrix_SELL_C_sigma mtx_SELL_C_sigma = convert_CSR_to_SELL_C_sigma(mtx_CSR);
	
	cout << "csr" << endl;
	for (int i = 0; i < 8; i++) {
		cout << "i=" << i << " ";
		for (int j = mtx_CSR.row_id[i]; j < mtx_CSR.row_id[i + 1]; j++) {
			cout << "(" << mtx_CSR.col[j] << ", " << mtx_CSR.value[j] << ")" << " ";
		}
		cout << endl;
	}
	cout << "..." << endl;
	
	cout << "sell-c-sigma" << endl;
	for (int j = mtx_SELL_C_sigma.cs[0]; j < mtx_SELL_C_sigma.cs[1]; j += 8) {
		for (int i = 0; i < 8; i++) {
			cout << "(" << mtx_SELL_C_sigma.col[j + i] << ", " << mtx_SELL_C_sigma.value[j + i] << ") ";
		}
		cout << endl;
	}
	
	cout << "cl cs" << endl;
	for (int i = 0; i < 10; i++) {
		cout << mtx_SELL_C_sigma.cl[i] << " " << mtx_SELL_C_sigma.cs[i] << endl;
	}
	cout << "... ..." << endl;
 	
	cout << "convert_CSR_to_SELL_C_sigma successfully?" << endl;
	
	vector_format res_scs = spmv_sell_c_sigma(mtx_SELL_C_sigma, v, threads_num);
	
	cout << "spmv_sell_c_sigma successfully?" << endl;
	
	double mx_diff_3 = 0;
	for (int i = 0; i < res_naive.N; i++) {
		mx_diff_3 = max(mx_diff_3, abs(res_naive.value[i] - res_scs.value[i]));
	}
	cout << "mx_diff_3: " << mx_diff_3 << endl;
	
	cout << "res_naive res_scs" << endl;
	for (int i = 0; i < 100; i++) {
		cout << res_naive.value[i] << " " << res_scs.value[i] << endl;
	}
	cout << "... ..." << endl;
	
	*/
 	return 0;
}
