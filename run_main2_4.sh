#!/bin/sh

#SBATCH --time=400 --partition=k1_8gb
#SBATCH --output=out-%j-run_main2_4.out

echo start_run_main2_4

date

./main2_exe ../test_mtx/cant.mtx 4 25

./main2_exe ../test_mtx/pdb1HYS.mtx 4 25

./main2_exe ../test_mtx/rma10.mtx 4 25

./main2_exe ../test_mtx/scircuit.mtx 4 25

./main2_exe ../test_mtx_3/kkt_power.mtx 4 25

./main2_exe ../test_mtx_3/mac_econ_fwd500/mac_econ_fwd500.mtx 4 25

./main2_exe ../test_mtx_3/RM07R/RM07R.mtx 4 25

./main2_exe ../test_mtx_3/Hamrle3.mtx 4 25

./main2_exe ../test_mtx_3/mc2depi.mtx 4 25

./main2_exe ../test_mtx_3/consph.mtx 4 25

echo end_run_main2_4

date
