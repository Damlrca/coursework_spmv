#!/bin/sh

#SBATCH --time=400 --partition=k1

echo start_run_2_t2

date

./main_exe ../test_mtx_3/mac_econ_fwd500/mac_econ_fwd500.mtx 2 50

./main_exe ../test_mtx_3/RM07R/RM07R.mtx 2 50

./main_exe ../test_mtx_3/Hamrle3.mtx 2 50

./main_exe ../test_mtx_3/mc2depi.mtx 2 50

./main_exe ../test_mtx_3/consph.mtx 2 50

echo end_run_2_t2

date
