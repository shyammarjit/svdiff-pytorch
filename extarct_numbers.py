import torch, os, clip, glob, json
from PIL import Image
from tqdm import trange
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import shutil
import numpy as np
import pandas as pd
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from pathlib import Path


###################################################################################
##                                  Code
###################################################################################

# load the clip model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, preprocess = clip.load('ViT-B/32', device=device, jit=False)
dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)


def Convert(image):
    return image.convert("RGB")

class DatasetWrapper(Dataset):
    def __init__(self, images_path, input_size = 224, eval = "clip"):
        self.images_path = images_path; self.input_size = input_size
        # Build transform
        self.trans = T.Compose([T.Resize(size=(self.input_size, self.input_size)), T.ToTensor()])
        self.trans = T.Compose([
            Resize(input_size, interpolation=Image.BICUBIC),
            CenterCrop(input_size),
            Convert,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

        if eval=="dino":
          self.trans =  Compose([
              Resize(self.input_size, interpolation=Image.BICUBIC),
              CenterCrop(self.input_size),
              Convert,
              ToTensor(),
              Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
          ])
        # total_images = args.images
        # distribution = [i for i in range(total_images)]
        # num_selected_images = int(selection_p * total_images)
        # sampled_elements = random.sample(distribution, num_selected_images)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        # print(idx)
        img = preprocess(Image.open(self.images_path[idx]))
        return img


def CLIP_I(org_folder_path = None, gen_folder_path = None):
    org_image_files = sorted([os.path.join(org_folder_path, f) for f in os.listdir(org_folder_path)])
    # gen_image_files = sorted([os.path.join(gen_folder_path, f) for f in os.listdir(gen_folder_path)])

    gen_image_files = glob.glob(os.path.join(gen_folder_path, '*.png'))
    # print(gen_image_files)

    # create dataloader for both the folder
    if(len(org_image_files)>128): batch_size = 128
    else: batch_size = len(org_image_files)
    org_datloader = DataLoader(DatasetWrapper(org_image_files), batch_size=batch_size, shuffle=False)
    if(len(gen_image_files)>128): batch_size = 128
    else: batch_size = len(gen_image_files)
    gen_dataloader = DataLoader(DatasetWrapper(gen_image_files), batch_size=batch_size, shuffle=False)

    clipi = []
    for i_batch in org_datloader:
        i_batch = model.encode_image(i_batch.to(device)).to(device) # pass this to CLIP model
    for j_batch in gen_dataloader:
        j_batch = model.encode_image(j_batch.to(device)).to(device) # pass this to CLIP model

    i_batch = i_batch.unsqueeze(0).expand(j_batch.size(0), -1, -1)  # shape: (8, 4, 12)

    # Compute cosine similarity
    with torch.no_grad():
        cosine_sim = torch.nn.functional.cosine_similarity(i_batch, j_batch.unsqueeze(1), dim=2)  # shape: (8, 4)
    cosine_sim = cosine_sim.mean(dim=1)
    cosine_sim = cosine_sim.cpu().detach().numpy()
    return np.mean(cosine_sim), np.std(cosine_sim)



def CLIP_T(gen_folder_path = None, prompts = None):
    total_similarity, num_images = [], 0
    gen_folder_path = glob.glob(os.path.join(gen_folder_path, '*.png'))

    # Iterate over the images and prompts simultaneously
    for image_filename in gen_folder_path:
        # Load and preprocess the image
        # image_path = os.path.join(gen_folder_path, image_filename)
        image = preprocess(Image.open(image_filename)).unsqueeze(0).to(device)

        prompt = image_filename[:-4]
        # Tokenize and encode the text prompt
        text_input = clip.tokenize(prompt).to(device)
        text_embedding = model.encode_text(text_input).to(device)

        image_embedding = model.encode_image(image).to(device) # Encode the image
        similarity = torch.cosine_similarity(image_embedding, text_embedding)
        # Accumulate the similarity score
        total_similarity.append(similarity.item())

    # Calculate the average similarity score
    # print(total_similarity)
    return np.mean(total_similarity), np.std(total_similarity)

def DINO(org_folder_path = None, gen_folder_path = None):
    org_image_files = sorted([os.path.join(org_folder_path, f) for f in os.listdir(org_folder_path)])
    gen_image_files = glob.glob(os.path.join(gen_folder_path, '*.png'))

    # gen_image_files = sorted([os.path.join(gen_folder_path, f) for f in os.listdir(gen_folder_path)])

    # create dataloader for both the folder
    if(len(org_image_files)>128): batch_size = 128
    else: batch_size = len(org_image_files)
    org_datloader = DataLoader(DatasetWrapper(org_image_files, eval="dino"), batch_size=batch_size, shuffle=False)
    if(len(gen_image_files)>128): batch_size = 128
    else: batch_size = len(gen_image_files)
    gen_dataloader = DataLoader(DatasetWrapper(gen_image_files, eval="dino"), batch_size=batch_size, shuffle=False)

    clipi = []
    dino_model.eval()
    for i_batch in org_datloader:
        i_batch = dino_model(i_batch.to(device)).to(device) # pass this to CLIP model
    for j_batch in gen_dataloader:
        j_batch = dino_model(j_batch.to(device)).to(device) # pass this to CLIP model

    i_batch = i_batch.unsqueeze(0).expand(j_batch.size(0), -1, -1)  # shape: (8, 4, 12)

    # Compute cosine similarity
    with torch.no_grad():
        cosine_sim = torch.nn.functional.cosine_similarity(i_batch, j_batch.unsqueeze(1), dim=2)  # shape: (8, 4)
    cosine_sim = cosine_sim.mean(dim=1)
    cosine_sim = cosine_sim.cpu().detach().numpy()
    return np.mean(cosine_sim), np.std(cosine_sim)


def evaluator(image_dir, org_data_path):
    clipi = CLIP_I(org_folder_path = org_data_path, gen_folder_path = image_dir)
    clipt = CLIP_T(gen_folder_path = image_dir)
    dino = DINO(org_folder_path = org_data_path, gen_folder_path = image_dir)
    return clipi, clipt, dino


# import argparse
# parser = argparse.ArgumentParser(description="Simple example of a training script.")
# parser.add_argument(
#         "--subject",
#         type=str,
#         default=None,
#         required=True,
#         help="The prompt with identifier specifying the instance",
#     )
# args = parser.parse_args()
subjects = ['human_harshit','human_nityanand','human_shyam','car','anime_shokokomi','anime_nami','anime_Kakashi','anime_kiriko','HuggingFace','dog','dog2','dog3','dog5','dog6','dog7','dog8','doggy','cat','cat2','cat3','cat4','teapot','robotToy','backpack','book','dog_backpack','rc_car','shinyShoes','duck','clock','plushie3','monstertoy','plushie1','plushie','plushie2','building','vase']
dataset_path = "/home/shyam/svdiff-pytorch/Data/"
instance_path = "/home/shyam/svdiff-pytorch/svdiff_output/"

CLIPI, CLIPT, DINO_ = [], [], []
for subject in subjects:
    org_data_path = os.path.join(dataset_path, subject, 'input')
    instance_data = os.path.join(instance_path, subject, 'checkpoint-500')

    # compute the quantiative results (CLIP-I, CLIP-T)
    clipi, clipt, dino = evaluator(instance_data, org_data_path)
    CLIPI.append(clipi[0])
    CLIPT.append(clipt[0])
    DINO_.append(dino[0])
    print(f"subject: {clipi}, {clipt}, {dino}")

print(np.mean(CLIPI), np.std(CLIPI))
print(np.mean(CLIPT), np.std(CLIPT))
print(np.mean(DINO_), np.std(DINO_))


