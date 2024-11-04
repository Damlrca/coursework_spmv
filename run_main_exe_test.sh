#!/bin/sh

#SBATCH --time=180 --partition=k1

echo start_run_main_exe_test

date

./main_exe ../test_mtx/cant.mtx 4 5000

echo end_run_main_exe_test

date
