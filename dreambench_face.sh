#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-cpu=10G
#SBATCH -C gmem24
#SBATCH --job-name=face
#SBATCH --output=outputs_svdiff/face.out

source activate svdiff
export MODEL_NAME="runwayml/stable-diffusion-v1-5"

##################################################################
#                                Anime
##################################################################

export INSTANCE_DIR="/home/shyam/svdiff-pytorch/Data/anime_Kakashi/input"
export CLASS_DIR="/home/shyam/class_data/anime"
export OUTPUT_DIR="/home/shyam/svdiff_output/anime_Kakashi"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks kakashi_anime" \
    --class_prompt="photo of a anime" \
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

export INSTANCE_DIR="/home/shyam/svdiff-pytorch/Data/anime_kiriko/input"
export CLASS_DIR="/home/shyam/class_data/anime"
export OUTPUT_DIR="/home/shyam/svdiff_output/anime_kiriko"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks kiriko_anime" \
    --class_prompt="photo of a anime" \
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


export INSTANCE_DIR="/home/shyam/svdiff-pytorch/Data/anime_nami/input"
export CLASS_DIR="/home/shyam/class_data/anime"
export OUTPUT_DIR="/home/shyam/svdiff_output/anime_nami"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks nami_anime" \
    --class_prompt="photo of a anime" \
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

export INSTANCE_DIR="/home/shyam/svdiff-pytorch/Data/anime_shokokomi/input"
export CLASS_DIR="/home/shyam/class_data/anime"
export OUTPUT_DIR="/home/shyam/svdiff_output/anime_shokokomi"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks shokokomi_anime" \
    --class_prompt="photo of a anime" \
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


export INSTANCE_DIR="/home/shyam/svdiff-pytorch/Data/HuggingFace/input"
export CLASS_DIR="/home/shyam/class_data/logo"
export OUTPUT_DIR="/home/shyam/svdiff_output/HuggingFace"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks huggingface logo" \
    --class_prompt="photo of a logo" \
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


export INSTANCE_DIR="/home/shyam/svdiff-pytorch/Data/human_harshit/input"
export CLASS_DIR="/home/shyam/class_data/face"
export OUTPUT_DIR="/home/shyam/svdiff_output/human_harshit"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks harshit_face" \
    --class_prompt="photo of a face" \
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


export INSTANCE_DIR="/home/shyam/svdiff-pytorch/Data/human_shyam/input"
export CLASS_DIR="/home/shyam/class_data/face"
export OUTPUT_DIR="/home/shyam/svdiff_output/human_shyam"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks shyam_face" \
    --class_prompt="photo of a face" \
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


export INSTANCE_DIR="/home/shyam/svdiff-pytorch/Data/human_nityanand/input"
export CLASS_DIR="/home/shyam/class_data/face"
export OUTPUT_DIR="/home/shyam/svdiff_output/human_nityanand"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks nityanand_face" \
    --class_prompt="photo of a face" \
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