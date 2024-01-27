export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/shyam/dataset_svdiff/dog"
export CLASS_DIR="/home/shyam/class_data/dog"
export OUTPUT_DIR="/home/shyam/svdiff_output/dog"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks dog" \
    --class_prompt="photo of a dog" \
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

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/shyam/dataset_svdiff/cat"
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


export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/shyam/dataset_svdiff/chair"
export CLASS_DIR="/home/shyam/class_data/chair"
export OUTPUT_DIR="/home/shyam/svdiff_output/chair"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks chair" \
    --class_prompt="photo of a chair" \
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


export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/shyam/dataset_svdiff/tortoise_plushy"
export CLASS_DIR="/home/shyam/class_data/tortoise_plushy"
export OUTPUT_DIR="/home/shyam/svdiff_output/tortoise_plushy"

accelerate launch train_svdiff.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="photo of a sks tortoise plushy" \
    --class_prompt="photo of a tortoise plushy" \
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