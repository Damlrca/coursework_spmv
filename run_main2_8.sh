#!/bin/sh

#SBATCH --time=400 --partition=k1_8gb
#SBATCH --output=out-%j-run_main2_8.out

echo start_run_main2_8

date

./main2_exe ../test_bin/cant.bin 8 25

./main2_exe ../test_bin/pdb1HYS.bin 8 25

./main2_exe ../test_bin/rma10.bin 8 25

./main2_exe ../test_bin/scircuit.bin 8 25

./main2_exe ../test_bin/kkt_power.bin 8 25

./main2_exe ../test_bin/mac_econ_fwd500.bin 8 25

./main2_exe ../test_bin/RM07R.bin 8 25

./main2_exe ../test_bin/Hamrle3.bin 8 25

./main2_exe ../test_bin/mc2depi.bin 8 25

./main2_exe ../test_bin/consph.bin 8 25

echo end_run_main2_8

date
