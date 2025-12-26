#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=32GB           
#PBS -l walltime=38:00:00  
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
	--total-steps 10000000 \
	--train-chunk 25000 \
	--n-envs 1 \
	--n-steps 4096 \
	--batch-size 128 \
	--n-epochs 10 \
	--learning-rate 5e-4 \
	--gamma 0.995 \
	--kl-coef 0.01 \
	--ent-coef 0.01 \
	--clip-range 0.3 \
	--eval-episodes 3 \
	--eval-max-steps 18000 \
	--record-steps 18000 \
    --reward-scale 0.05 \
	--intrinsic-enable \
	--intrinsic-scale 0.05 \
	--intrinsic-w-curiosity 0.0 \
	--intrinsic-w-novelty 0.0 \
	--intrinsic-w-surprise 1.0 \
	--log-dir "./runs_IR013" \
	--device "cuda:0" \
	>> IR013.log 2>&1