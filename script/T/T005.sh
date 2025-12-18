#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=32GB           
#PBS -l walltime=5:00:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3

source /scratch/rp06/sl5952/RL-test/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/RL-test/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 -m retro.import ./roms
python3 src/main.py \
	--game "SuperMarioWorld-Snes" \
	--state "YoshiIsland1" \
	--total-steps 250000 \
	--train-chunk 25000 \
	--n-envs 1 \
	--n-steps 2048 \
	--batch-size 256 \
	--n-epochs 4 \
	--learning-rate 3e-4 \
	--gamma 0.995 \
	--kl-coef 0.02 \
	--ent-coef 0.01 \
	--clip-range 0.2 \
	--eval-episodes 2 \
	--eval-max-steps 6000 \
	--record-steps 3000 \
	--backbone "resnet18" \
	--log-dir "./runs_T005" \
	--device "cuda:0" \
	>> T005.log 2>&1