#!/bin/bash
#PBS -P kf09
#PBS -q dgxa100
#PBS -l ngpus=1            
#PBS -l ncpus=16            
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
	--eval-episodes 2 \
	--eval-max-steps 6000 \
	--record-steps 3000 \
    --reward-scale 0.05 \
	--intrinsic-enable \
	--intrinsic-scale 0.05 \
	--intrinsic-w-curiosity 0.1 \
	--intrinsic-w-novelty 0.1 \
	--intrinsic-w-surprise 0.8 \
	--backbone "resnet34d.ra2_in1k" \
	--secret-stage1-bonus 2 \
	--secret-stage1-x-min 1886 --secret-stage1-x-max 1944 \
	--secret-stage2-spin-bonus 0.5 \
	--secret-stage2-spin-required 2 \
	--secret-stage2-spin-button A \
	--secret-stage3-bonus 100 \
	--log-dir "./runs_IR008" \
	--device "cuda:0" \
	>> IR008.log 2>&1