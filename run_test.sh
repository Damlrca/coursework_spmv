#!/bin/sh

#SBATCH --time=180 --partition=k1_8gb
#SBATCH --output=out_test-%j.out

echo -n "START RUNNING (run_test.sh) " && date

./main2_exe ../test_bin/mac_econ_fwd500.bin 8 10

echo -n "END RUNNING (run_test.sh) " && date
