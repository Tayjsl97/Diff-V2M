#!/bin/bash
#SBATCH --job-name=infer_odf
#SBATCH --partition=a10
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=300:00:10
#SBATCH --mail-user=xxx@gmail.com
#SBATCH --mail-type=ALL

srun -u python inference/generation.py \
    --model_config_path saved_models/model_config.json \
    --dataset_config_path stable_audio_tools/configs/dataset_configs/dataset_inference.json \
    --ckpt_path saved_models/model.ckpt \
    --output_dir ./output_odf