#!/bin/sh

#SBATCH --time=400 --partition=k1

echo start_run_1_t1

date

./main_exe ../test_mtx/cant.mtx 1 100

./main_exe ../test_mtx/pdb1HYS.mtx 1 100

./main_exe ../test_mtx/rma10.mtx 1 100

./main_exe ../test_mtx/scircuit.mtx 1 100

./main_exe ../test_mtx_3/kkt_power.mtx 1 100

echo end_run_1_t1

date
