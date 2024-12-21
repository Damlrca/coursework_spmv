#!/bin/sh

#SBATCH --time=400 --partition=k1

echo start_run_1_t2

date

./main_exe ../test_mtx/cant.mtx 2 100

./main_exe ../test_mtx/pdb1HYS.mtx 2 100

./main_exe ../test_mtx/rma10.mtx 2 100

./main_exe ../test_mtx/scircuit.mtx 2 100

./main_exe ../test_mtx_3/kkt_power.mtx 2 100

echo end_run_1_t2

date
