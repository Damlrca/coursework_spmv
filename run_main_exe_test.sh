#!/bin/sh

#SBATCH --time=180 --partition=k1

echo start_run_main_exe_test

date

./main_exe ../test_mtx_3/kkt_power.mtx 8 10

echo end_run_main_exe_test

date
