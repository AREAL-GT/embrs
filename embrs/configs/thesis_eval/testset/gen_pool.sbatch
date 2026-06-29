#!/bin/bash
#SBATCH -J testset_gen
#SBATCH --account=gts-jrogers8-paid
#SBATCH -q inferno
#SBATCH -p cpu-small
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2G
#SBATCH -t 02:00:00
#SBATCH -o /storage/project/r-jrogers8-0/rjdp3/rui/embrs/embrs/configs/thesis_eval/testset/slurm/gen_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rjdp3@gatech.edu
set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate embrs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export MPLBACKEND=Agg
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$NUMBA_CACHE_DIR"
# Per-task WindNinja temp dir — the default WINDNINJA_TEMP_PATH is shared and
# run_windninja() CLEARS it at start, so concurrent tasks corrupt each other's
# wind fields (FileNotFoundError / SIGSEGV). Node-local + job/array-unique.
export WINDNINJA_TEMP_PATH="${TMPDIR:-/tmp}/windninja_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$WINDNINJA_TEMP_PATH"
cd "/storage/home/hcoda1/0/rjdp3/r-jrogers8-0/rui/ra-cbba-core"
python -m applications.firefighting.tuning.make_testset --phase generate-pool --out-dir /storage/project/r-jrogers8-0/rjdp3/rui/embrs/embrs/configs/thesis_eval/testset --regions a_flint_hills b_front_range_foothills c_northern_sierra_nevada --classes mild moderate extreme --base-cfg-suffix _pace --noise-seed-start 100 --tolerance 0.3 --pool-size 10 
