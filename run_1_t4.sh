#!/bin/sh

#SBATCH --time=400 --partition=k1

echo start_run_1_t4

date

./main_exe ../test_mtx/cant.mtx 4 50

./main_exe ../test_mtx/pdb1HYS.mtx 4 50

./main_exe ../test_mtx/rma10.mtx 4 50

./main_exe ../test_mtx/scircuit.mtx 4 50

./main_exe ../test_mtx_3/kkt_power.mtx 4 50

echo end_run_1_t4

date
