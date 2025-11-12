#!/bin/bash
#SBATCH --job-name=dit_odf
#SBATCH --partition=a100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=300:00:10
#SBATCH --mail-user=xxx@gmail.com
#SBATCH --mail-type=ALL


srun -u python train.py \
    --dataset-config stable_audio_tools/configs/dataset_configs/dataset_train.json \
    --model-config saved_models/stable_audio/model_config.json \
    --val-dataset-config stable_audio_tools/configs/dataset_configs/dataset_val.json \
    --pretrained-ckpt-path saved_models/stable_audio/model.safetensors \
    --save-dir saved_models/ \
    --checkpoint-every 150 \
    --val-every 602 \
    --accum-batches 4 \
    --batch-size 32 \
    --num-gpus 2 \
    --save-top-k 1 \
    --name dit_odf
