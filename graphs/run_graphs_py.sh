#!/bin/sh

#SBATCH --time=400 --partition=k1_8gb
#SBATCH --output=out-%j-graphs-py.out

echo -n "START RUNNING graphs.py " && date

python3 ./graphs.py ../out-XXXXXX-run_main2_1.out ../out-XXXXXX-run_main2_2.out ../out-XXXXXX-run_main2_3.out ../out-XXXXXX-run_main2_4.out

echo -n "END RUNNING graphs.py " && date
