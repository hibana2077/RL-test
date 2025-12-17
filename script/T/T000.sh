#!/bin/bash
#PBS -P rp06
#PBS -q gpuhopper
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=32GB           
#PBS -l walltime=10:00:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3

source /scratch/rp06/sl5952/RL-test/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/RL-test/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 -m retro.import ./roms
python3 src/main.py >> T000.log 2>&1