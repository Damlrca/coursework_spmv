#!/bin/sh

#SBATCH --time=180 --partition=k1

echo start_run_main_exe

date

./main_exe ../test_mtx/cant.mtx

./main_exe ../test_mtx/pdb1HYS.mtx

./main_exe ../test_mtx/rma10.mtx

./main_exe ../test_mtx/scircuit.mtx

./main_exe ../test_mtx_3/kkt_power.mtx

./main_exe ../test_mtx/road_central.mtx

echo end_run_main_exe

date
