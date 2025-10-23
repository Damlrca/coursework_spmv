#!/bin/sh

#SBATCH --time=400 --partition=k1_8gb
#SBATCH --output=out-%j-run_main2_1.out

echo -n "START RUNNING MAIN2_EXE (run_main2_1.sh) " && date

./main2_exe ../test_bin/cant.bin 1 25

./main2_exe ../test_bin/pdb1HYS.bin 1 25

./main2_exe ../test_bin/rma10.bin 1 25

./main2_exe ../test_bin/scircuit.bin 1 25

./main2_exe ../test_bin/kkt_power.bin 1 25

./main2_exe ../test_bin/mac_econ_fwd500.bin 1 25

./main2_exe ../test_bin/RM07R.bin 1 25

./main2_exe ../test_bin/Hamrle3.bin 1 25

./main2_exe ../test_bin/mc2depi.bin 1 25

./main2_exe ../test_bin/consph.bin 1 25

echo -n "END RUNNING MAIN2_EXE (run_main2_1.sh) " && date
