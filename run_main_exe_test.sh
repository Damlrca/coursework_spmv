#!/bin/sh

#SBATCH --time=180 --partition=k1_8gb
#SBATCH --output=out_test-%j.out

echo start_run_main_exe_test

date

# ./main_exe ../test_mtx_3/consph.mtx 8 10

# ./main_exe ../test_mtx_3/mc2depi.mtx 8 10

# ./main_exe ../test_mtx_3/Hamrle3.mtx 8 10

# ./main_exe ../test_mtx_3/RM07R/RM07R.mtx 1 10

./main_exe ../test_mtx_3/mac_econ_fwd500/mac_econ_fwd500.mtx 8 10

# ./main_exe ../test_mtx_3/kkt_power.mtx 8 10

echo end_run_main_exe_test

date
