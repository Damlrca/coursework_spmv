#!/bin/sh

#SBATCH --time=400 --partition=k1_8gb
#SBATCH --output=out-%j-run_mtx2bin.out

echo start_run_mtx2bin

date

./mtx2bin_exe ../test_mtx/cant.mtx ../test_bin/cant.bin

./mtx2bin_exe ../test_mtx/pdb1HYS.mtx ../test_bin/pdb1HYS.bin

./mtx2bin_exe ../test_mtx/rma10.mtx ../test_bin/rma10.bin

./mtx2bin_exe ../test_mtx/scircuit.mtx ../test_bin/scircuit.bin

./mtx2bin_exe ../test_mtx_3/kkt_power.mtx ../test_bin/kkt_power.bin

./mtx2bin_exe ../test_mtx_3/mac_econ_fwd500/mac_econ_fwd500.mtx ../test_bin/mac_econ_fwd500.bin

./mtx2bin_exe ../test_mtx_3/RM07R/RM07R.mtx ../test_bin/RM07R.bin

./mtx2bin_exe ../test_mtx_3/Hamrle3.mtx ../test_bin/Hamrle3.bin

./mtx2bin_exe ../test_mtx_3/mc2depi.mtx ../test_bin/mc2depi.bin

./mtx2bin_exe ../test_mtx_3/consph.mtx ../test_bin/consph.bin

echo end_run_mtx2bin

date
