// Copyright (C) 2025 Sadikov Damir
// github.com/Damlrca/coursework_spmv

// SOME UNUSED CODE

// unrolled versions of spmv_sell_c_sigma_noalloc:

// spmv_sell_c_sigma_noalloc_unroll4 <double>

template<int sigma>
void spmv_sell_c_sigma_noalloc_unroll4(const matrix_SELL_C_sigma<4, sigma, double>& mtx, const vector_format<double>& vec, int threads_num, vector_format<double>& res) {
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / 4; i++) {
		vfloat64m1_t v_summ_t1 = __riscv_vfmv_v_f_f64m1(0.0, 4);
		vfloat64m1_t v_summ_t2 = __riscv_vfmv_v_f_f64m1(0.0, 4);
		vfloat64m1_t v_summ_t3 = __riscv_vfmv_v_f_f64m1(0.0, 4);
		vfloat64m1_t v_summ_t4 = __riscv_vfmv_v_f_f64m1(0.0, 4);
		int j = mtx.cs[i];
		for (; j + 4 * 3 < mtx.cs[i + 1]; j += 4 * 4) {
			vuint32mf2_t index_shftd_1 = __riscv_vle32_v_u32mf2(reinterpret_cast<uint32_t *>(mtx.col + j + 4 * 0), 4);
			vuint32mf2_t index_shftd_2 = __riscv_vle32_v_u32mf2(reinterpret_cast<uint32_t *>(mtx.col + j + 4 * 1), 4);
			vuint32mf2_t index_shftd_3 = __riscv_vle32_v_u32mf2(reinterpret_cast<uint32_t *>(mtx.col + j + 4 * 2), 4);
			vuint32mf2_t index_shftd_4 = __riscv_vle32_v_u32mf2(reinterpret_cast<uint32_t *>(mtx.col + j + 4 * 3), 4);
			vfloat64m1_t v_1_1 = __riscv_vluxei32_v_f64m1(vec.value, index_shftd_1, 4);
			vfloat64m1_t v_1_2 = __riscv_vluxei32_v_f64m1(vec.value, index_shftd_2, 4);
			vfloat64m1_t v_1_3 = __riscv_vluxei32_v_f64m1(vec.value, index_shftd_3, 4);
			vfloat64m1_t v_1_4 = __riscv_vluxei32_v_f64m1(vec.value, index_shftd_4, 4);
			vfloat64m1_t v_2_1 = __riscv_vle64_v_f64m1(mtx.value + j + 4 * 0, 4);
			vfloat64m1_t v_2_2 = __riscv_vle64_v_f64m1(mtx.value + j + 4 * 1, 4);
			vfloat64m1_t v_2_3 = __riscv_vle64_v_f64m1(mtx.value + j + 4 * 2, 4);
			vfloat64m1_t v_2_4 = __riscv_vle64_v_f64m1(mtx.value + j + 4 * 3, 4);
			v_summ_t1 = __riscv_vfmacc_vv_f64m1(v_summ_t1, v_1_1, v_2_1, 4);
			v_summ_t2 = __riscv_vfmacc_vv_f64m1(v_summ_t2, v_1_2, v_2_2, 4);
			v_summ_t3 = __riscv_vfmacc_vv_f64m1(v_summ_t3, v_1_3, v_2_3, 4);
			v_summ_t4 = __riscv_vfmacc_vv_f64m1(v_summ_t4, v_1_4, v_2_4, 4);
		}
		vfloat64m1_t temp_summ_1 = __riscv_vfadd_vv_f64m1(v_summ_t1, v_summ_t2, 4);
		vfloat64m1_t temp_summ_2 = __riscv_vfadd_vv_f64m1(v_summ_t3, v_summ_t4, 4);
		vfloat64m1_t v_summ = __riscv_vfadd_vv_f64m1(temp_summ_1, temp_summ_2, 4);
		for (; j < mtx.cs[i + 1]; j += 4) {
			vuint32mf2_t index_shftd = __riscv_vle32_v_u32mf2(reinterpret_cast<uint32_t *>(mtx.col + j), 4);
			vfloat64m1_t v_1 = __riscv_vluxei32_v_f64m1(vec.value, index_shftd, 4);
			vfloat64m1_t v_2 = __riscv_vle64_v_f64m1(mtx.value + j, 4);
			v_summ = __riscv_vfmacc_vv_f64m1(v_summ, v_1, v_2, 4);
		}
		__riscv_vse64_v_f64m1(res.value + i * 4, v_summ, 4);
	}
}

template<int sigma>
void spmv_sell_c_sigma_noalloc_unroll4(const matrix_SELL_C_sigma<8, sigma, double>& mtx, const vector_format<double>& vec, int threads_num, vector_format<double>& res) {
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / 8; i++) {
		vfloat64m2_t v_summ_t1 = __riscv_vfmv_v_f_f64m2(0.0, 8);
		vfloat64m2_t v_summ_t2 = __riscv_vfmv_v_f_f64m2(0.0, 8);
		vfloat64m2_t v_summ_t3 = __riscv_vfmv_v_f_f64m2(0.0, 8);
		vfloat64m2_t v_summ_t4 = __riscv_vfmv_v_f_f64m2(0.0, 8);
		int j = mtx.cs[i];
		for (; j + 8 * 3 < mtx.cs[i + 1]; j += 8 * 4) {
			vuint32m1_t index_shftd_1 = __riscv_vle32_v_u32m1(reinterpret_cast<uint32_t *>(mtx.col + j + 8 * 0), 8);
			vuint32m1_t index_shftd_2 = __riscv_vle32_v_u32m1(reinterpret_cast<uint32_t *>(mtx.col + j + 8 * 1), 8);
			vuint32m1_t index_shftd_3 = __riscv_vle32_v_u32m1(reinterpret_cast<uint32_t *>(mtx.col + j + 8 * 2), 8);
			vuint32m1_t index_shftd_4 = __riscv_vle32_v_u32m1(reinterpret_cast<uint32_t *>(mtx.col + j + 8 * 3), 8);
			vfloat64m2_t v_1_1 = __riscv_vluxei32_v_f64m2(vec.value, index_shftd_1, 8);
			vfloat64m2_t v_1_2 = __riscv_vluxei32_v_f64m2(vec.value, index_shftd_2, 8);
			vfloat64m2_t v_1_3 = __riscv_vluxei32_v_f64m2(vec.value, index_shftd_3, 8);
			vfloat64m2_t v_1_4 = __riscv_vluxei32_v_f64m2(vec.value, index_shftd_4, 8);
			vfloat64m2_t v_2_1 = __riscv_vle64_v_f64m2(mtx.value + j + 8 * 0, 8);
			vfloat64m2_t v_2_2 = __riscv_vle64_v_f64m2(mtx.value + j + 8 * 1, 8);
			vfloat64m2_t v_2_3 = __riscv_vle64_v_f64m2(mtx.value + j + 8 * 2, 8);
			vfloat64m2_t v_2_4 = __riscv_vle64_v_f64m2(mtx.value + j + 8 * 3, 8);
			v_summ_t1 = __riscv_vfmacc_vv_f64m2(v_summ_t1, v_1_1, v_2_1, 8);
			v_summ_t2 = __riscv_vfmacc_vv_f64m2(v_summ_t2, v_1_2, v_2_2, 8);
			v_summ_t3 = __riscv_vfmacc_vv_f64m2(v_summ_t3, v_1_3, v_2_3, 8);
			v_summ_t4 = __riscv_vfmacc_vv_f64m2(v_summ_t4, v_1_4, v_2_4, 8);
		}
		vfloat64m2_t temp_summ_1 = __riscv_vfadd_vv_f64m2(v_summ_t1, v_summ_t2, 8);
		vfloat64m2_t temp_summ_2 = __riscv_vfadd_vv_f64m2(v_summ_t3, v_summ_t4, 8);
		vfloat64m2_t v_summ = __riscv_vfadd_vv_f64m2(temp_summ_1, temp_summ_2, 8);
		for (; j < mtx.cs[i + 1]; j += 8) {
			vuint32m1_t index_shftd = __riscv_vle32_v_u32m1(reinterpret_cast<uint32_t *>(mtx.col + j), 8);
			vfloat64m2_t v_1 = __riscv_vluxei32_v_f64m2(vec.value, index_shftd, 8);
			vfloat64m2_t v_2 = __riscv_vle64_v_f64m2(mtx.value + j, 8);
			v_summ = __riscv_vfmacc_vv_f64m2(v_summ, v_1, v_2, 8);
		}
		__riscv_vse64_v_f64m2(res.value + i * 8, v_summ, 8);
	}
}

template<int sigma>
void spmv_sell_c_sigma_noalloc_unroll4(const matrix_SELL_C_sigma<16, sigma, double>& mtx, const vector_format<double>& vec, int threads_num, vector_format<double>& res) {
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / 16; i++) {
		vfloat64m4_t v_summ_t1 = __riscv_vfmv_v_f_f64m4(0.0, 16);
		vfloat64m4_t v_summ_t2 = __riscv_vfmv_v_f_f64m4(0.0, 16);
		vfloat64m4_t v_summ_t3 = __riscv_vfmv_v_f_f64m4(0.0, 16);
		vfloat64m4_t v_summ_t4 = __riscv_vfmv_v_f_f64m4(0.0, 16);
		int j = mtx.cs[i];
		for (; j + 16 * 3 < mtx.cs[i + 1]; j += 16 * 4) {
			vuint32m2_t index_shftd_1 = __riscv_vle32_v_u32m2(reinterpret_cast<uint32_t *>(mtx.col + j + 16 * 0), 16);
			vuint32m2_t index_shftd_2 = __riscv_vle32_v_u32m2(reinterpret_cast<uint32_t *>(mtx.col + j + 16 * 1), 16);
			vuint32m2_t index_shftd_3 = __riscv_vle32_v_u32m2(reinterpret_cast<uint32_t *>(mtx.col + j + 16 * 2), 16);
			vuint32m2_t index_shftd_4 = __riscv_vle32_v_u32m2(reinterpret_cast<uint32_t *>(mtx.col + j + 16 * 3), 16);
			vfloat64m4_t v_1_1 = __riscv_vluxei32_v_f64m4(vec.value, index_shftd_1, 16);
			vfloat64m4_t v_1_2 = __riscv_vluxei32_v_f64m4(vec.value, index_shftd_2, 16);
			vfloat64m4_t v_1_3 = __riscv_vluxei32_v_f64m4(vec.value, index_shftd_3, 16);
			vfloat64m4_t v_1_4 = __riscv_vluxei32_v_f64m4(vec.value, index_shftd_4, 16);
			vfloat64m4_t v_2_1 = __riscv_vle64_v_f64m4(mtx.value + j + 16 * 0, 16);
			vfloat64m4_t v_2_2 = __riscv_vle64_v_f64m4(mtx.value + j + 16 * 1, 16);
			vfloat64m4_t v_2_3 = __riscv_vle64_v_f64m4(mtx.value + j + 16 * 2, 16);
			vfloat64m4_t v_2_4 = __riscv_vle64_v_f64m4(mtx.value + j + 16 * 3, 16);
			v_summ_t1 = __riscv_vfmacc_vv_f64m4(v_summ_t1, v_1_1, v_2_1, 16);
			v_summ_t2 = __riscv_vfmacc_vv_f64m4(v_summ_t2, v_1_2, v_2_2, 16);
			v_summ_t3 = __riscv_vfmacc_vv_f64m4(v_summ_t3, v_1_3, v_2_3, 16);
			v_summ_t4 = __riscv_vfmacc_vv_f64m4(v_summ_t4, v_1_4, v_2_4, 16);
		}
		vfloat64m4_t temp_summ_1 = __riscv_vfadd_vv_f64m4(v_summ_t1, v_summ_t2, 16);
		vfloat64m4_t temp_summ_2 = __riscv_vfadd_vv_f64m4(v_summ_t3, v_summ_t4, 16);
		vfloat64m4_t v_summ = __riscv_vfadd_vv_f64m4(temp_summ_1, temp_summ_2, 16);
		for (; j < mtx.cs[i + 1]; j += 16) {
			vuint32m2_t index_shftd = __riscv_vle32_v_u32m2(reinterpret_cast<uint32_t *>(mtx.col + j), 16);
			vfloat64m4_t v_1 = __riscv_vluxei32_v_f64m4(vec.value, index_shftd, 16);
			vfloat64m4_t v_2 = __riscv_vle64_v_f64m4(mtx.value + j, 16);
			v_summ = __riscv_vfmacc_vv_f64m4(v_summ, v_1, v_2, 16);
		}
		__riscv_vse64_v_f64m4(res.value + i * 16, v_summ, 16);
	}
}

template<int sigma>
void spmv_sell_c_sigma_noalloc_unroll4(const matrix_SELL_C_sigma<32, sigma, double>& mtx, const vector_format<double>& vec, int threads_num, vector_format<double>& res) {
#pragma omp parallel for num_threads(threads_num) schedule(dynamic)
	for (int i = 0; i < mtx.N / 32; i++) {
		vfloat64m8_t v_summ_t1 = __riscv_vfmv_v_f_f64m8(0.0, 32);
		vfloat64m8_t v_summ_t2 = __riscv_vfmv_v_f_f64m8(0.0, 32);
		vfloat64m8_t v_summ_t3 = __riscv_vfmv_v_f_f64m8(0.0, 32);
		vfloat64m8_t v_summ_t4 = __riscv_vfmv_v_f_f64m8(0.0, 32);
		int j = mtx.cs[i];
		for (; j + 32 * 3 < mtx.cs[i + 1]; j += 32 * 4) {
			vuint32m4_t index_shftd_1 = __riscv_vle32_v_u32m4(reinterpret_cast<uint32_t *>(mtx.col + j + 32 * 0), 32);
			vuint32m4_t index_shftd_2 = __riscv_vle32_v_u32m4(reinterpret_cast<uint32_t *>(mtx.col + j + 32 * 1), 32);
			vuint32m4_t index_shftd_3 = __riscv_vle32_v_u32m4(reinterpret_cast<uint32_t *>(mtx.col + j + 32 * 2), 32);
			vuint32m4_t index_shftd_4 = __riscv_vle32_v_u32m4(reinterpret_cast<uint32_t *>(mtx.col + j + 32 * 3), 32);
			vfloat64m8_t v_1_1 = __riscv_vluxei32_v_f64m8(vec.value, index_shftd_1, 32);
			vfloat64m8_t v_1_2 = __riscv_vluxei32_v_f64m8(vec.value, index_shftd_2, 32);
			vfloat64m8_t v_1_3 = __riscv_vluxei32_v_f64m8(vec.value, index_shftd_3, 32);
			vfloat64m8_t v_1_4 = __riscv_vluxei32_v_f64m8(vec.value, index_shftd_4, 32);
			vfloat64m8_t v_2_1 = __riscv_vle64_v_f64m8(mtx.value + j + 32 * 0, 32);
			vfloat64m8_t v_2_2 = __riscv_vle64_v_f64m8(mtx.value + j + 32 * 1, 32);
			vfloat64m8_t v_2_3 = __riscv_vle64_v_f64m8(mtx.value + j + 32 * 2, 32);
			vfloat64m8_t v_2_4 = __riscv_vle64_v_f64m8(mtx.value + j + 32 * 3, 32);
			v_summ_t1 = __riscv_vfmacc_vv_f64m8(v_summ_t1, v_1_1, v_2_1, 32);
			v_summ_t2 = __riscv_vfmacc_vv_f64m8(v_summ_t2, v_1_2, v_2_2, 32);
			v_summ_t3 = __riscv_vfmacc_vv_f64m8(v_summ_t3, v_1_3, v_2_3, 32);
			v_summ_t4 = __riscv_vfmacc_vv_f64m8(v_summ_t4, v_1_4, v_2_4, 32);
		}
		vfloat64m8_t temp_summ_1 = __riscv_vfadd_vv_f64m8(v_summ_t1, v_summ_t2, 32);
		vfloat64m8_t temp_summ_2 = __riscv_vfadd_vv_f64m8(v_summ_t3, v_summ_t4, 32);
		vfloat64m8_t v_summ = __riscv_vfadd_vv_f64m8(temp_summ_1, temp_summ_2, 32);
		for (; j < mtx.cs[i + 1]; j += 32) {
			vuint32m4_t index_shftd = __riscv_vle32_v_u32m4(reinterpret_cast<uint32_t *>(mtx.col + j), 32);
			vfloat64m8_t v_1 = __riscv_vluxei32_v_f64m8(vec.value, index_shftd, 32);
			vfloat64m8_t v_2 = __riscv_vle64_v_f64m8(mtx.value + j, 32);
			v_summ = __riscv_vfmacc_vv_f64m8(v_summ, v_1, v_2, 32);
		}
		__riscv_vse64_v_f64m8(res.value + i * 32, v_summ, 32);
	}
}

// TESTING UROLLED FUNCTIONS

template<int C, int sigma>
void test_sell_c_sigmau(const int ite, const int threads_num, const matrix_CSR<double>& mtx_CSR, string name) {
	if (C != 4 && C != 8 && C != 16 && C != 32) {
		cout << "test_sell_c_sigma : wrong C == " << C << endl;
		return;
	}
	
	cout << "mtx_SELL_C_sigma<" << C << ", " << sigma << ">: ";
	matrix_SELL_C_sigma<C, sigma, double> mtx = convert_CSR_to_SELL_C_sigma<C, sigma>(mtx_CSR);
	
	vector_format<double> scs_res = alloc_vector_res(mtx);
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
	
	vector_format<double> real_scs_res = alloc_vector_res(mtx);
	for (int i = 0; i < mtx_CSR.N; i++) {
		real_scs_res.value[mtx.rows_perm[i]] = scs_res.value[i];
	}
	
	cout << res << "us per iteration (minimum); diff = " << calc_diff(naive_res, real_scs_res) << endl;
	
	results.push_back(name);
	results.push_back(to_string(mtx.cs[mtx.N / C]));
	results.push_back(us_to_ms_string(res));
}

{
	test_sell_c_sigmau< 4, 1>(ite, threads_num, mtx_CSR, "scs_4_1_u4");
	test_sell_c_sigmau< 8, 1>(ite, threads_num, mtx_CSR, "scs_8_1_u4");
	test_sell_c_sigmau<16, 1>(ite, threads_num, mtx_CSR, "scs16_1_u4");
	test_sell_c_sigmau<32, 1>(ite, threads_num, mtx_CSR, "scs32_1_u4");
	test_sell_c_sigmau< 4, 2>(ite, threads_num, mtx_CSR, "scs_4_2_u4");
	test_sell_c_sigmau< 8, 2>(ite, threads_num, mtx_CSR, "scs_8_2_u4");
	test_sell_c_sigmau<16, 2>(ite, threads_num, mtx_CSR, "scs16_2_u4");
	test_sell_c_sigmau<32, 2>(ite, threads_num, mtx_CSR, "scs32_2_u4");
	test_sell_c_sigmau< 4, 4>(ite, threads_num, mtx_CSR, "scs_4_4_u4");
	test_sell_c_sigmau< 8, 4>(ite, threads_num, mtx_CSR, "scs_8_4_u4");
	test_sell_c_sigmau<16, 4>(ite, threads_num, mtx_CSR, "scs16_4_u4");
	test_sell_c_sigmau<32, 4>(ite, threads_num, mtx_CSR, "scs32_4_u4");
	test_sell_c_sigmau< 4, 8>(ite, threads_num, mtx_CSR, "scs_4_8_u4");
	test_sell_c_sigmau< 8, 8>(ite, threads_num, mtx_CSR, "scs_8_8_u4");
	test_sell_c_sigmau<16, 8>(ite, threads_num, mtx_CSR, "scs16_8_u4");
	test_sell_c_sigmau<32, 8>(ite, threads_num, mtx_CSR, "scs32_8_u4");
}

{
	test_sell_c_sigmau< 4, SIGMA_SORTED>(ite, threads_num, mtx_CSR, "scs_4_S_u4");
	test_sell_c_sigmau< 8, SIGMA_SORTED>(ite, threads_num, mtx_CSR, "scs_8_S_u4");
	test_sell_c_sigmau<16, SIGMA_SORTED>(ite, threads_num, mtx_CSR, "scs16_S_u4");
	test_sell_c_sigmau<32, SIGMA_SORTED>(ite, threads_num, mtx_CSR, "scs32_S_u4");
}

{
	test_sell_c_sigmau< 8, 1>(ite, threads_num, mtx_CSR_float, "scs_8_1_u4_f");
	test_sell_c_sigmau<16, 1>(ite, threads_num, mtx_CSR_float, "scs16_1_u4_f");
	test_sell_c_sigmau<32, 1>(ite, threads_num, mtx_CSR_float, "scs32_1_u4_f");
	test_sell_c_sigmau<64, 1>(ite, threads_num, mtx_CSR_float, "scs64_1_u4_f");
	test_sell_c_sigmau< 8, 2>(ite, threads_num, mtx_CSR_float, "scs_8_2_u4_f");
	test_sell_c_sigmau<16, 2>(ite, threads_num, mtx_CSR_float, "scs16_2_u4_f");
	test_sell_c_sigmau<32, 2>(ite, threads_num, mtx_CSR_float, "scs32_2_u4_f");
	test_sell_c_sigmau<64, 2>(ite, threads_num, mtx_CSR_float, "scs64_2_u4_f");
	test_sell_c_sigmau< 8, 4>(ite, threads_num, mtx_CSR_float, "scs_8_4_u4_f");
	test_sell_c_sigmau<16, 4>(ite, threads_num, mtx_CSR_float, "scs16_4_u4_f");
	test_sell_c_sigmau<32, 4>(ite, threads_num, mtx_CSR_float, "scs32_4_u4_f");
	test_sell_c_sigmau<64, 4>(ite, threads_num, mtx_CSR_float, "scs64_4_u4_f");
	test_sell_c_sigmau< 8, 8>(ite, threads_num, mtx_CSR_float, "scs_8_8_u4_f");
	test_sell_c_sigmau<16, 8>(ite, threads_num, mtx_CSR_float, "scs16_8_u4_f");
	test_sell_c_sigmau<32, 8>(ite, threads_num, mtx_CSR_float, "scs32_8_u4_f");
	test_sell_c_sigmau<64, 8>(ite, threads_num, mtx_CSR_float, "scs64_8_u4_f");
}

{
	test_sell_c_sigmau< 8, SIGMA_SORTED>(ite, threads_num, mtx_CSR_float, "scs_8_S_u4_f");
	test_sell_c_sigmau<16, SIGMA_SORTED>(ite, threads_num, mtx_CSR_float, "scs16_S_u4_f");
	test_sell_c_sigmau<32, SIGMA_SORTED>(ite, threads_num, mtx_CSR_float, "scs32_S_u4_f");
	test_sell_c_sigmau<64, SIGMA_SORTED>(ite, threads_num, mtx_CSR_float, "scs64_S_u4_f");
}
