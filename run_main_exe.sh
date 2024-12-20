#!/bin/sh

#SBATCH --time=400 --partition=k1

echo start_run_main_exe

date

./main_exe ../test_mtx/cant.mtx 1 50

./main_exe ../test_mtx/pdb1HYS.mtx 1 50

./main_exe ../test_mtx/rma10.mtx 1 50

./main_exe ../test_mtx/scircuit.mtx 1 50

./main_exe ../test_mtx_3/kkt_power.mtx 1 50

./main_exe ../test_mtx/cant.mtx 2 50

./main_exe ../test_mtx/pdb1HYS.mtx 2 50

./main_exe ../test_mtx/rma10.mtx 2 50

./main_exe ../test_mtx/scircuit.mtx 2 50

./main_exe ../test_mtx_3/kkt_power.mtx 2 50

./main_exe ../test_mtx/cant.mtx 4 50

./main_exe ../test_mtx/pdb1HYS.mtx 4 50

./main_exe ../test_mtx/rma10.mtx 4 50

./main_exe ../test_mtx/scircuit.mtx 4 50

./main_exe ../test_mtx_3/kkt_power.mtx 4 50

./main_exe ../test_mtx/cant.mtx 8 50

./main_exe ../test_mtx/pdb1HYS.mtx 8 50

./main_exe ../test_mtx/rma10.mtx 8 50

./main_exe ../test_mtx/scircuit.mtx 8 50

./main_exe ../test_mtx_3/kkt_power.mtx 8 50

# ./main_exe ../test_mtx/cant.mtx

# ./main_exe ../test_mtx/pdb1HYS.mtx

# ./main_exe ../test_mtx/rma10.mtx

# ./main_exe ../test_mtx/scircuit.mtx

# ./main_exe ../test_mtx_3/kkt_power.mtx

# ./main_exe ../test_mtx/road_central.mtx

echo end_run_main_exe

date
