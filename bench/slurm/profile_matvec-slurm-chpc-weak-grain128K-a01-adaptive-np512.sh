#!/bin/bash
#SBATCH --time=2:00:00     # walltime, abbreviated by -t
#SBATCH -o /uufs/chpc.utah.edu/common/home/u1210678/paralab/Dendro-KT/results/commScaling/out-job-%j-node-%N.tsv      # name of the stdout, using the job number (%j) and 
                           # the first node (%N)
#SBATCH -e /uufs/chpc.utah.edu/common/home/u1210678/paralab/Dendro-KT/results/commScaling/err-job-%j-node-%N.log      # name of the stderr, using job and first node values

#SBATCH --nodes=4
#SBATCH --ntasks=112

#SBATCH --account=soc-gpu-kp    # account - abbreviated by -A
#SBATCH --partition=soc-gpu-kp  # partition, abbreviated by -p
#SBATCH --mem=0             ## All the memory of the node


#SBATCH --job-name=profileMatvec
#SBATCH --mail-type=ALL
#SBATCH --mail-user=masado@cs.utah.edu

# load appropriate modules
module load intel/2019.5.281 impi/2019.5.281
module load petsc/3.12.0

RUNPROGRAM=/uufs/chpc.utah.edu/common/home/u1210678/paralab/Dendro-KT/build/profileMatvec
TRANSPOSEIT=/uufs/chpc.utah.edu/common/home/u1210678/paralab/Dendro-KT/helper/transposeIt

PTS_PER_PROC_WEAK=131072

# If there are not enough levels then we may get errors.
MAX_DEPTH=20

TASKS=1023
NUM_WARMUP=0
NUM_RUNS=1
ELE_ORDER=1
SFC_TOL=0.3
LENPOW=0
ISADAPTIVE=1

OUT_PREFIX1="out-${SLURM_JOB_ID}-weak-128K-a01-adap"
OUT_PREFIX4="out-${SLURM_JOB_ID}-weak-512K-a01-adap"

# Supposedly good for oversubscribing
export I_MPI_WAIT_MODE=1

mpirun -np $TASKS $RUNPROGRAM $NUM_WARMUP $NUM_RUNS $PTS_PER_PROC $ELE_ORDER $SFC_TOL $LENPOW $ISADAPTIVE > "${OUT_PREFIX1}-transpose.tsv"

$TRANSPOSEIT 73 "${OUT_PREFIX1}-transpose.tsv" > "${OUT_PREFIX1}.tsv"
cat "${OUT_PREFIX1}.tsv" | egrep 'npes|.min' > "${OUT_PREFIX1}-min.tsv"
cat "${OUT_PREFIX1}.tsv" | egrep 'npes|.mean' > "${OUT_PREFIX1}-mean.tsv"
cat "${OUT_PREFIX1}.tsv" | egrep 'npes|.max' > "${OUT_PREFIX1}-max"


mpirun -np $TASKS $RUNPROGRAM $NUM_WARMUP $NUM_RUNS $(( 4 * $PTS_PER_PROC )) $ELE_ORDER $SFC_TOL $LENPOW $ISADAPTIVE > "${OUT_PREFIX4}-transpose.tsv"

$TRANSPOSEIT 73 "${OUT_PREFIX4}-transpose.tsv" > "${OUT_PREFIX4}.tsv"
cat "${OUT_PREFIX4}.tsv" | egrep 'npes|.min' > "${OUT_PREFIX4}-min.tsv"
cat "${OUT_PREFIX4}.tsv" | egrep 'npes|.mean' > "${OUT_PREFIX4}-mean.tsv"
cat "${OUT_PREFIX4}.tsv" | egrep 'npes|.max' > "${OUT_PREFIX4}-max"

