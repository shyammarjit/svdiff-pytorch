import os
import sys
import io
import requests
import PIL
import torch
from torch import autocast
import random
from tqdm import tqdm
from transformers import CLIPTextModel
from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionPipeline
)
from PIL import Image
from svdiff_pytorch import load_unet_for_svdiff, load_text_encoder_for_svdiff, SCHEDULER_MAPPING, image_grid
MODEL_NAME="runwayml/stable-diffusion-v1-5"

#====================================================================
## Please change this one only.
spectral_shifts_ckpt = "/home/shyam/svdiff_output_rank/robot_toy"
output_folder_path = "./output_rank/robot_toy"
# seeds = [1, 2, 3]
seeds=1
prompts = ['A sks toy on top of a wooden floor', 'A sks toy in the snow']
#====================================================================


def load_text_encoder(pretrained_model_name_or_path, spectral_shifts_ckpt, device, fp16=False):
    if os.path.isdir(spectral_shifts_ckpt):
        spectral_shifts_ckpt = os.path.join(spectral_shifts_ckpt, "spectral_shifts_te.safetensors")
    elif not os.path.exists(spectral_shifts_ckpt):
        # download from hub
        hf_hub_kwargs = {} if hf_hub_kwargs is None else hf_hub_kwargs
        try:
            spectral_shifts_ckpt = huggingface_hub.hf_hub_download(spectral_shifts_ckpt, filename="spectral_shifts_te.safetensors", **hf_hub_kwargs)
        except huggingface_hub.utils.EntryNotFoundError:
            return CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch.float16 if fp16 else None).to(device)
    if not os.path.exists(spectral_shifts_ckpt):
            return CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch.float16 if fp16 else None).to(device)
    text_encoder = load_text_encoder_for_svdiff(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        spectral_shifts_ckpt=spectral_shifts_ckpt,
        subfolder="text_encoder",
    )
    # first perform svd and cache
    for module in text_encoder.modules():
        if hasattr(module, "perform_svd"):
            module.perform_svd()
    if fp16:
        text_encoder = text_encoder.to(device, dtype=torch.float16)
    return text_encoder

scheduler_type = "dpm_solver++" #@param ["ddim", "plms", "lms", "euler", "euler_ancestral", "dpm_solver++"]

device = "cuda" if torch.cuda.is_available() else "cpu"
unet = load_unet_for_svdiff(MODEL_NAME, spectral_shifts_ckpt=spectral_shifts_ckpt, subfolder="unet")
unet = unet.to(device)
for module in unet.modules():
  if hasattr(module, "perform_svd"):
    module.perform_svd()

unet = unet.to(device, dtype=torch.float16)
text_encoder = load_text_encoder(
    pretrained_model_name_or_path=MODEL_NAME,
    spectral_shifts_ckpt=spectral_shifts_ckpt,
    device=device,
    fp16=True,
)
# load pipe
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    unet=unet,
    text_encoder=text_encoder,
    vae=AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse"),
    requires_safety_checker=False,
    safety_checker=None,
    feature_extractor=None,
    scheduler=SCHEDULER_MAPPING[scheduler_type].from_pretrained(MODEL_NAME, subfolder="scheduler"),
    torch_dtype=torch.float16
)

pipe = pipe.to(device)
print("loaded pipeline")

num_images_per_prompt = 2 # @param {type: "integer"}
guidance_scale = 7.5 # @param {type: "number"}
num_inference_steps = 25 # @param {type: "integer"}
height = 512 # @param {type: "integer"}
width = 512 # @param {type: "integer"}
spectral_shifts_scale = 1.0 #@param {type: "number"}


if pipe.unet.conv_out.scale != spectral_shifts_scale:
  for module in pipe.unet.modules():
    if hasattr(module, "set_scale"):
      module.set_scale(scale=spectral_shifts_scale)
  for module in pipe.text_encoder.modules():
    if hasattr(module, "set_scale"):
      module.set_scale(scale=spectral_shifts_scale)
  print(f"Set spectral_shifts_scale to {spectral_shifts_scale}!")


if not os.path.isdir(output_folder_path):
    os.mkdir(output_folder_path)

all_images = []
for prompt in tqdm(prompts):
    if type(seeds)==int:
        seeds = [seeds]
    for seed in seeds:
        g_cuda = torch.Generator(device='cuda').manual_seed(seed)
        print(f"{prompt}")
        print(f"seed: {seed}")
        with torch.autocast(device), torch.inference_mode():
            images = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=g_cuda
            ).images
        images[0].save(os.path.join(output_folder_path, f"{seed}. {prompt}.png"))