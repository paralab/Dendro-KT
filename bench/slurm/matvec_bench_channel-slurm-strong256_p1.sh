#!/bin/bash
#SBATCH --time=0:30:00     # walltime, abbreviated by -t
#SBATCH -o ../results/ipdps21/out-job-%j-node-%N.tsv      # name of the stdout, using the job number (%j) and 
                           # the first node (%N)
#SBATCH -e ../results/ipdps21/err-job-%j-node-%N.log      # name of the stderr, using job and first node values

#SBATCH --nodes=10         #
#SBATCH --ntasks=512      # maximum total number of mpi tasks across all nodes.
#SBATCH -p normal

#SBATCH --job-name=matvec_bench_channel-strong256_p1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=masado@cs.utah.edu


##
## Note: Run this from the build directory.
##

# load appropriate modules

module load petsc/3.13
module load intel/19.0.5

RUNPROGRAM=/home1/07803/masado/Dendro-KT/build/matvecBenchChannel

PTS_TOTAL_STRONG=4096
PTS_PER_PROC_WEAK=1024

# If there are not enough levels then we may get errors.
MAX_DEPTH=15

ELE_ORDER=1
ibrun -n   2  -o   0 $RUNPROGRAM $(($PTS_TOTAL_STRONG /   2)) $MAX_DEPTH $ELE_ORDER "strong_p1" &
ibrun -n   4  -o   2 $RUNPROGRAM $(($PTS_TOTAL_STRONG /   4)) $MAX_DEPTH $ELE_ORDER "strong_p1" &
ibrun -n   8  -o   6 $RUNPROGRAM $(($PTS_TOTAL_STRONG /   8)) $MAX_DEPTH $ELE_ORDER "strong_p1" &
ibrun -n  16  -o  14 $RUNPROGRAM $(($PTS_TOTAL_STRONG /  16)) $MAX_DEPTH $ELE_ORDER "strong_p1" &
ibrun -n  32  -o  30 $RUNPROGRAM $(($PTS_TOTAL_STRONG /  32)) $MAX_DEPTH $ELE_ORDER "strong_p1" &
ibrun -n  64  -o  62 $RUNPROGRAM $(($PTS_TOTAL_STRONG /  64)) $MAX_DEPTH $ELE_ORDER "strong_p1" &
ibrun -n 128  -o 126 $RUNPROGRAM $(($PTS_TOTAL_STRONG / 128)) $MAX_DEPTH $ELE_ORDER "strong_p1" &
ibrun -n 256  -o 254 $RUNPROGRAM $(($PTS_TOTAL_STRONG / 256)) $MAX_DEPTH $ELE_ORDER "strong_p1" &
wait
