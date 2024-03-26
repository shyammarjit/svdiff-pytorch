#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-cpu=10G
#SBATCH -C gmem16
#SBATCH --job-name=stuff2
#SBATCH --output=outputs_svdiff/stuff2.out

source activate svdiff
export MODEL_NAME="runwayml/stable-diffusion-v1-5"

##################################################################
#                                STUFF
##################################################################

export INSTANCE_DIR="/home/shyam/svdiff-pytorch/Data/plushie/input"
export CLASS_DIR="/home/shyam/class_data/plushie"
export OUTPUT_DIR="/home/shyam/svdiff_output/plushie"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks plushie" \
    --class_prompt="photo of a plushie" \
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

export INSTANCE_DIR="/home/shyam/svdiff-pytorch/Data/plushie1/input"
export CLASS_DIR="/home/shyam/class_data/plushie"
export OUTPUT_DIR="/home/shyam/svdiff_output/plushie1"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks plushie" \
    --class_prompt="photo of a plushie" \
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


export INSTANCE_DIR="/home/shyam/svdiff-pytorch/Data/plushie2/input"
export CLASS_DIR="/home/shyam/class_data/plushie"
export OUTPUT_DIR="/home/shyam/svdiff_output/plushie2"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks plushie" \
    --class_prompt="photo of a plushie" \
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


export INSTANCE_DIR="/home/shyam/svdiff-pytorch/Data/plushie3/input"
export CLASS_DIR="/home/shyam/class_data/plushie"
export OUTPUT_DIR="/home/shyam/svdiff_output/plushie3"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks plushie" \
    --class_prompt="photo of a plushie" \
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


export INSTANCE_DIR="/home/shyam/svdiff-pytorch/Data/rc_car/input"
export CLASS_DIR="/home/shyam/class_data/toy"
export OUTPUT_DIR="/home/shyam/svdiff_output/rc_car"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks rc_car toy" \
    --class_prompt="photo of a toy" \
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


export INSTANCE_DIR="/home/shyam/svdiff-pytorch/Data/robotToy/input"
export CLASS_DIR="/home/shyam/class_data/toy"
export OUTPUT_DIR="/home/shyam/svdiff_output/robotToy"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks robot_toy" \
    --class_prompt="photo of a toy" \
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

export INSTANCE_DIR="/home/shyam/svdiff-pytorch/Data/shinyShoes/input"
export CLASS_DIR="/home/shyam/class_data/shoes"
export OUTPUT_DIR="/home/shyam/svdiff_output/shinyShoes"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks shiny_shoes" \
    --class_prompt="photo of a shoes" \
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