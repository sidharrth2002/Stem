# %% [markdown]
# requirements: UNI & CONCH installed  
# https://github.com/mahmoodlab/UNI/tree/main   
# https://github.com/mahmoodlab/CONCH   
#    
# Search for "TODO" and fill in your HuggingFace login token.

# %%
import os
from os import listdir
from os.path import isfile, join
import math
import time
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import scanpy as sc
import anndata

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from huggingface_hub import login

import torchvision
from torchvision import datasets, models, transforms
from torchvision.io import read_image
from torchvision.transforms import ToTensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from tempfile import TemporaryDirectory

# %%
def get_img_patch_embd(img, 
                       adata,
                       samplename,
                       device,
                       save_path=None):
    # CONCH model
    from conch.open_clip_custom import create_model_from_pretrained
    pretrained_CONCH, preprocess_CONCH = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", device=device,
                                        hf_auth_token="") # TODO: need to replace "" by HuggingFace Login Token

    # UNI model
    from uni import get_encoder
    login(token="") # TODO: need to replace "" by HuggingFace Login Token
    model_UNI, transform_UNI = get_encoder(enc_name='uni', device=device)


    def get_img_embd_conch(patch, model=pretrained_CONCH, preprocess=preprocess_CONCH):
        # resize to 256 by 256
        base_width = 256
        patch_resized = patch.resize((base_width, base_width), Image.Resampling.LANCZOS)
        patch_processed = preprocess(patch_resized).unsqueeze(0)
        with torch.inference_mode():
            feature_emb = model.encode_image(patch_processed.to(device), 
                                             proj_contrast=False, 
                                             normalize=False)        # [1, 512]
        return torch.clone(feature_emb)
    
    def get_img_embd_uni(patch, model=model_UNI, transform=transform_UNI):
        # resize to 224 by 224
        base_width = 224
        patch_resized = patch.resize((base_width, base_width), 
                                     Image.Resampling.LANCZOS)       # [224, 224]
        img_transformed = transform(patch_resized).unsqueeze(dim=0)  # [1, 3, 224, 224]
        with torch.inference_mode():
            feature_emb = model(img_transformed.to(device))          # [1, 1024]
        return torch.clone(feature_emb)

    def patch_augmentation_embd(patch, conch_or_uni,
                                num_transpose = 7):
        if conch_or_uni == "conch":
            embd_dim = 512
        elif conch_or_uni == "uni":
            embd_dim = 1024
        patch_aug_embd = torch.zeros(num_transpose, embd_dim)
        for trans in range(num_transpose):    # apply augmentations to the image patch
            patch_transposed = patch.transpose(trans)
            if conch_or_uni == "conch":
                patch_embd = get_img_embd_conch(patch_transposed)
            elif conch_or_uni == "uni":
                patch_embd = get_img_embd_uni(patch_transposed)
            patch_aug_embd[trans, :] = torch.clone(patch_embd)
        return patch_aug_embd.unsqueeze(0)


    # process spot
    spot_diameter = adata.uns["spatial"]["ST"]["scalefactors"]["spot_diameter_fullres"]
    print("Spot diameter: ", spot_diameter)  # Spot diameter for Visium
    if spot_diameter < 224: 
        radius = 112                         # minimum patch size: 224 by 224
    else:
        radius = int(spot_diameter // 2)
    x = adata.obsm["spatial"][:, 0]          # x coordinate in H&E image
    y = adata.obsm["spatial"][:, 1]          # y coordinate in H&E image

    all_patch_ebd_conch = None
    all_patch_ebd_uni = None
    all_patch_ebd_conch_aug = None
    all_patch_ebd_uni_aug = None
    first = True

    for spot_idx in tqdm(range(len(x))):
        patch = img.crop((x[spot_idx]-radius, y[spot_idx]-radius, 
                          x[spot_idx]+radius, y[spot_idx]+radius))
        patch_ebd_conch = get_img_embd_conch(patch)
        patch_ebd_uni   = get_img_embd_uni(patch)
        patch_ebd_conch_aug = patch_augmentation_embd(patch, "conch")
        patch_ebd_uni_aug   = patch_augmentation_embd(patch, "uni")

        if first:
            all_patch_ebd_conch = patch_ebd_conch
            all_patch_ebd_uni   = patch_ebd_uni
            all_patch_ebd_conch_aug = patch_ebd_conch_aug
            all_patch_ebd_uni_aug   = patch_ebd_uni_aug
            first = False
        else:
            all_patch_ebd_conch = torch.cat((all_patch_ebd_conch, patch_ebd_conch), dim=0)
            all_patch_ebd_uni   = torch.cat((all_patch_ebd_uni, patch_ebd_uni), dim=0)
            all_patch_ebd_conch_aug = torch.cat((all_patch_ebd_conch_aug, patch_ebd_conch_aug), dim=0)
            all_patch_ebd_uni_aug   = torch.cat((all_patch_ebd_uni_aug, patch_ebd_uni_aug), dim=0)
    print("Final data size: ", 
          all_patch_ebd_conch.shape, 
          all_patch_ebd_uni.shape,
          all_patch_ebd_conch_aug.shape,
          all_patch_ebd_uni_aug.shape)    

    if save_path != None:
        torch.save(all_patch_ebd_conch.detach().cpu(), save_path + "1spot_conch_ebd/" + samplename + "_conch.pt")
        torch.save(all_patch_ebd_uni.detach().cpu(),   save_path + "1spot_uni_ebd/"   + samplename + "_uni.pt")
        torch.save(all_patch_ebd_conch_aug.detach().cpu(), save_path + "1spot_conch_ebd_aug/" + samplename + "_conch_aug.pt")
        torch.save(all_patch_ebd_uni_aug.detach().cpu(),   save_path + "1spot_uni_ebd_aug/"   + samplename + "_uni_aug.pt")

# %% [markdown]
# Load dataset downloaded from hest1k: [example: PRAD]

# %%
# define paths
data_path = "/auto/archive/tcga/sn666/hest1k_datasets/kidney/"   # local path to dataset
tif_path = data_path + 'wsis/'          # H&E image path
st_path = data_path + "st/"             # ST data path

# %%
# load ST adata
adata_lst = []
fn_lst = ["NCBI" + str(i) for i in range(692, 713)] # filename for kidney dataset
# fn_lst = ["MEND"+str(i) for i in range(139, 163)] # filename for PRAD dataset
# fn_lst.remove("MEND155")                          # no MEND155 in the dataset

first = True
for fn in fn_lst:
    adata = anndata.read_h5ad(st_path + fn + ".h5ad")
    adata_lst.append(adata)
    if first:
        common_genes = adata.var_names 
        first = False
        print(fn, adata.shape)
        continue
    common_genes = set(common_genes).intersection(set(adata.var_names))
    print(fn, adata.shape, end="\t")

# keep common genes
print("Length of common genes: ", len(common_genes))
common_genes = sorted(list(common_genes))
for fni in range(len(fn_lst)):
    adata = adata_lst[fni].copy()
    adata_lst[fni] = adata[:, common_genes].copy()
    print(fn_lst[fni], " ", adata_lst[fni].shape)
print("Only keep common genes across the slides.")

# %%
# generate image patch embeddings (UNI and CONCH, original and augmented)

# os.makedirs(data_path + "processed_data/", exist_ok=True)
# os.makedirs(data_path + "processed_data/1spot_conch_ebd/")
# os.makedirs(data_path + "processed_data/1spot_uni_ebd/")
# os.makedirs(data_path + "processed_data/1spot_conch_ebd_aug/")
# os.makedirs(data_path + "processed_data/1spot_uni_ebd_aug/")
# if error, folder exists.

for i in range(len(fn_lst)):
    fn = fn_lst[i]
    adata = adata_lst[i]
    print(fn)
    image = Image.open(tif_path + fn + ".tif")
    get_img_patch_embd(image, adata, fn, device="cuda", 
                       save_path=data_path + "processed_data/")
    print("#" * 20)

# %% [markdown]
# Select genes  
# - Following code as an example
# - Could be any given gene list saved in: data_path + "processed_data/selected_gene_list.txt"

# %%
union_hvg = set()

for fn_idx in range(len(fn_lst)):
    adata = adata_lst[fn_idx].copy()
    fn = fn_lst[fn_idx]
    
    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)

    union_hvg = union_hvg.union(set(adata.var_names[adata.var["highly_variable"]]))
    print(fn, len(union_hvg))

union_hvg = sorted([gene for gene in union_hvg if not gene.startswith(("MT", "mt", "RPS", "RPL"))]) # [optional] remove mitochondrial genes and ribosomal genes
print(len(union_hvg))

# %%
# select union_hvg and concat all slides
all_count_df = pd.DataFrame(adata_lst[0][:, union_hvg].X.toarray(), 
                            columns=union_hvg, 
                            index=[fn_lst[0] + "_" + str(i) for i in range(adata_lst[0].shape[0])]).T

for fn_idx in range(1, len(fn_lst)):
    adata = adata_lst[fn_idx]
    df = pd.DataFrame(adata[:, union_hvg].X.toarray(), 
                      columns=union_hvg, 
                      index=[fn_lst[fn_idx] + "_" + str(i) for i in range(adata.shape[0])]).T
    all_count_df = pd.concat([all_count_df, df], axis=1)
    print(fn_lst[fn_idx], adata.shape, all_count_df.shape)

all_count_df.fillna(0, inplace=True)
all_count_df = all_count_df.T

# %%
# order selected genes by mean and std
all_gene_order_by_mean = all_count_df.mean(axis=0).sort_values(ascending=False).index
all_gene_order_by_std = all_count_df.std(axis=0).sort_values(ascending=False).index

# %%
# select top intersection of high mean and high variance genes

num_genes = 241 # to make final gene list of length 200

selected_genes = sorted(list(set(all_gene_order_by_mean[:num_genes]).intersection(set(all_gene_order_by_std[:num_genes]))))
print(len(selected_genes))

# %%
with open(data_path + "processed_data/selected_gene_list.txt", "w") as f:
    for gene in selected_genes:
        f.write(gene + "\n")

with open(data_path + "processed_data/all_slide_lst.txt", "w") as f:
    for fn in fn_lst:
        f.write(fn + "\n")

# %% [markdown]
# # Additional Pathology Foundation Models:

# %% [markdown]
# H-optimus-0: https://huggingface.co/bioptimus/H-optimus-0

# %%
# from huggingface_hub import login
# login(token="") # TODO: need to replace "" by HuggingFace Login Token

# from torchvision import transforms

# def get_img_patch_embd(img, 
#                        adata,
#                        samplename,
#                        device,
#                        save_path=None):

#     # load model
#     model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, 
#                               init_values=1e-5, dynamic_img_size=False)
#     model = model.to(device)
#     model = model.eval()
#     transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=(0.707223, 0.578729, 0.703617), 
#         std=(0.211883, 0.230117, 0.177517)
#     ),
# ])

    
#     def get_img_embd(patch, model=model, transform=transform):
#         # resize to 224 by 224
#         base_width = 224
#         patch_resized = patch.resize((base_width, base_width), 
#                                      Image.Resampling.LANCZOS) # [224, 224]
#         img_transformed = transform(transforms.ToPILImage()(torch.tensor(np.array(patch_resized)).permute(2, 0, 1))).unsqueeze(0)  # [1, 3, 224, 224]
#         with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16):
#             features = model(img_transformed.to(device))
#         return torch.clone(features).detach().cpu()

#     def patch_augmentation_embd(patch, num_transpose = 7):
#         embd_dim = 1536
#         patch_aug_embd = torch.zeros(num_transpose, embd_dim)
#         for trans in range(num_transpose):
#             patch_transposed = patch.transpose(trans)
#             patch_embd = get_img_embd(patch_transposed)
#             patch_aug_embd[trans, :] = torch.clone(patch_embd)
#         return patch_aug_embd.unsqueeze(0)
    
#     # process spot
#     spot_diameter = adata.uns["spatial"]["ST"]["scalefactors"]["spot_diameter_fullres"]
#     print("Spot diameter: ", spot_diameter)  # Spot diameter for Visium
#     if spot_diameter < 224: 
#         radius = 112                         # minimum patch size: 224 by 224
#     else:
#         radius = int(spot_diameter // 2)
#     x = adata.obsm["spatial"][:, 0]          # x coordinate in H&E image
#     y = adata.obsm["spatial"][:, 1]          # y coordinate in H&E image


#     img_embd = None
#     img_embd_aug = None
#     first_patch = True

#     for spot_idx in tqdm(range(len(x))):
#         patch = img.crop((x[spot_idx]-radius, y[spot_idx]-radius, 
#                           x[spot_idx]+radius, y[spot_idx]+radius))
#         patch_embd     = get_img_embd(patch)
#         patch_embd_aug = patch_augmentation_embd(patch)

#         if first_patch:
#             img_embd = patch_embd
#             img_embd_aug = patch_embd_aug
#             first_patch = False
#         else:
#             img_embd = torch.cat((img_embd, patch_embd), dim=0)
#             img_embd_aug = torch.cat((img_embd_aug, patch_embd_aug), dim=0)

#     print("Final size:", img_embd.shape, img_embd_aug.shape)

#     if save_path is not None:
#         torch.save(img_embd,     save_path + f"1spot_hopt0_ebd/{samplename}_hopt0.pt")
#         torch.save(img_embd_aug, save_path + f"1spot_hopt0_ebd_aug/{samplename}_hopt0.pt")


# %% [markdown]
# Virchow2: https://huggingface.co/paige-ai/Virchow2

# %%
# from huggingface_hub import login
# login(token="") # TODO: need to replace "" by HuggingFace Login Token
# import timm
# import torch
# from timm.data import resolve_data_config
# from timm.data.transforms_factory import create_transform
# from timm.layers import SwiGLUPacked

# def get_img_patch_embd(img, 
#                        adata,
#                        samplename,
#                        device,
#                        save_path=None):

#     # load model
#     # need to specify MLP layer and activation function for proper init
#     model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, 
#                               mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
#     model = model.to(device)
#     model = model.eval()
#     transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    
#     def get_img_embd(patch, model=model, transform=transforms):
#         # resize to 224 by 224
#         base_width = 224
#         patch_resized = patch.resize((base_width, base_width), 
#                                      Image.Resampling.LANCZOS) # [224, 224]
#         img_transformed = transform(patch_resized).unsqueeze(0)  # [1, 3, 224, 224]
#         with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
#             output = model(img_transformed.to(device))  # [1, 2560]
#             class_token = output[:, 0]
#             patch_tokens = output[:, 5:]
#             embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
#         return torch.clone(embedding).detach().cpu()

#     def patch_augmentation_embd(patch, num_transpose = 7):
#         embd_dim = 2560
#         patch_aug_embd = torch.zeros(num_transpose, embd_dim)
#         for trans in range(num_transpose):
#             patch_transposed = patch.transpose(trans)
#             patch_embd = get_img_embd(patch_transposed)
#             patch_aug_embd[trans, :] = torch.clone(patch_embd)
#         return patch_aug_embd.unsqueeze(0)
    
    
#     # process spot
#     spot_diameter = adata.uns["spatial"]["ST"]["scalefactors"]["spot_diameter_fullres"]
#     print("Spot diameter: ", spot_diameter)  # Spot diameter for Visium
#     if spot_diameter < 224: 
#         radius = 112                         # minimum patch size: 224 by 224
#     else:
#         radius = int(spot_diameter // 2)
#     x = adata.obsm["spatial"][:, 0]          # x coordinate in H&E image
#     y = adata.obsm["spatial"][:, 1]          # y coordinate in H&E image


#     # initialize variables
#     first_patch = True
#     img_embd = None
#     img_embd_aug = None

#     for spot_idx in tqdm(range(len(x))):
#         patch = img.crop((x[spot_idx]-radius, y[spot_idx]-radius, 
#                           x[spot_idx]+radius, y[spot_idx]+radius))        
#         patch_embd     = get_img_embd(patch)
#         patch_embd_aug = patch_augmentation_embd(patch)

#         if first_patch:
#             img_embd = patch_embd
#             img_embd_aug = patch_embd_aug
#             first_patch = False
#         else:
#             img_embd = torch.cat((img_embd, patch_embd), dim=0)
#             img_embd_aug = torch.cat((img_embd_aug, patch_embd_aug), dim=0)

#     print("Final size:", img_embd.shape, img_embd_aug.shape)

#     if save_path is not None:
#         torch.save(img_embd,     save_path + f"1spot_virchow2_ebd/{samplename}_virchow2.pt")
#         torch.save(img_embd_aug, save_path + f"1spot_virchow2_ebd_aug/{samplename}_virchow2.pt")



