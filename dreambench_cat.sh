#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-cpu=10G
#SBATCH -C gmem24
#SBATCH --job-name=cat
#SBATCH --output=outputs_svdiff/cat.out

source activate svdiff
export MODEL_NAME="runwayml/stable-diffusion-v1-5"

##################################################################
#                                cat
##################################################################

export INSTANCE_DIR="/home/shyam/svdiff-pytorch/Data/cat/input"
export CLASS_DIR="/home/shyam/class_data/cat"
export OUTPUT_DIR="/home/shyam/svdiff_output/cat"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks cat" \
    --class_prompt="photo of a cat" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-3 \
    --learning_rate_1d=1e-6 \
    --train_text_encoder \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
    --max_train_steps=1000 \
    --checkpointing_steps=500 \
    --use_8bit_adam \
    --seed=42 \
    --enable_xformers_memory_efficient_attention \
    --gradient_checkpointing


# Dogs
export INSTANCE_DIR="/home/shyam/svdiff-pytorch/Data/cat2/input"
export CLASS_DIR="/home/shyam/class_data/cat"
export OUTPUT_DIR="/home/shyam/svdiff_output/cat2"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks cat" \
    --class_prompt="photo of a cat" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-3 \
    --learning_rate_1d=1e-6 \
    --train_text_encoder \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
    --max_train_steps=1000 \
    --checkpointing_steps=500 \
    --use_8bit_adam \
    --seed=42 \
    --enable_xformers_memory_efficient_attention \
    --gradient_checkpointing


# Dogs
export INSTANCE_DIR="/home/shyam/svdiff-pytorch/Data/cat3/input"
export CLASS_DIR="/home/shyam/class_data/cat"
export OUTPUT_DIR="/home/shyam/svdiff_output/cat3"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks cat" \
    --class_prompt="photo of a cat" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-3 \
    --learning_rate_1d=1e-6 \
    --train_text_encoder \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
    --max_train_steps=1000 \
    --checkpointing_steps=500 \
    --use_8bit_adam \
    --seed=42 \
    --enable_xformers_memory_efficient_attention \
    --gradient_checkpointing


# Dogs
export INSTANCE_DIR="/home/shyam/svdiff-pytorch/Data/cat4/input"
export CLASS_DIR="/home/shyam/class_data/cat"
export OUTPUT_DIR="/home/shyam/svdiff_output/cat4"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks cat" \
    --class_prompt="photo of a cat" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-3 \
    --learning_rate_1d=1e-6 \
    --train_text_encoder \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
    --max_train_steps=1000 \
    --checkpointing_steps=500 \
    --use_8bit_adam \
    --seed=42 \
    --enable_xformers_memory_efficient_attention \
    --gradient_checkpointing
