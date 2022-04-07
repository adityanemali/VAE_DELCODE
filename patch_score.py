import pandas as pd
from patch_model import VAE
import torch.nn as nn
import torch
import nibabel as nib
import os
import matplotlib.pyplot as plt
from sim3D import ssim3D
import  get_data
from loss import L1Loss, KLDivergence
from torch.utils import data
from tqdm import tqdm
import numpy as np
from monai.transforms import CropForeground
import torch.nn.functional as F
import glob
from patchify import patchify, unpatchify
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


mask  = 'data/icvmask_4mm.nii'
masked_data = nib.load(mask).get_fdata()

def threshold_at_one(x):
    # threshold at 1
    return x > 0


def patch_img(image, patch_size, step, reconstruct=False):
    # padding need for patches (works only for 16)
    image = np.pad(image, ((2, 3), (1, 2), (4, 4)), 'constant')
    patches = patchify(image, (patch_size, patch_size, patch_size), step)
    patches_shape = patches.shape
    patches = np.ascontiguousarray(patches).reshape(-1, patch_size, patch_size, patch_size)
    if(reconstruct):
        reconstructed_image = patches.reshape(patches_shape)
        reconstructed_image = unpatchify(reconstructed_image, image.shape)
        return patches, reconstructed_image
    else:
        return patches


class NiftiDataloader(data.Dataset):
    def __init__(self, path, patch_size, patch_step):
        self.files = glob.glob(path)
        self.files = np.sort(self.files)
        self.patch_size = patch_size
        self.patch_step = patch_step
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        image_name = self.files[idx]
        subject_id= os.path.basename(image_name).split('_')[3]
        subject_id = os.path.basename(subject_id).split('.')[0]
        img = nib.load(image_name).get_fdata()
        img = img*masked_data
        cropper = CropForeground(select_fn=threshold_at_one, margin=0)
        img = cropper(img)
        img_patches = patch_img(img, patch_size, patch_step, reconstruct=True)
        img_patches = img_patches.reshape(1, img_patches[0].shape[0], img_patches[1].shape[1], img_patches[2].shape[2], img_patches[3].shape[3])
        img_patches = torch.from_numpy(img_patches)
        return img_patches

l1_loss = L1Loss()
kl_loss = KLDivergence()

path = '/home/aditya/Documents/DELCODE-GP_MKL/data/unmodulated_segments/4mm/wp1*'


batch_size = 1
learning_rate =  0.001
patch_size = 16
patch_step = 8
full_dataset = NiftiDataloader(path, patch_size, patch_step)

train_size = int(0 * len(full_dataset))
val_size = len(full_dataset)
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
val_loader = data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size)


results = pd.DataFrame()
latent_dims = [1, 2, 4, 8]
for lat in latent_dims:

    latent_dim = lat # latent dimension for sampling

    model = VAE(latent_dim)
    model= nn.DataParallel(model)
    model.to(device)

    # select best model
    best_model = torch.load('outputs1/dim_'+str(lat)+'/best_model.pth')
    best_model_epoch = best_model['epoch']
    print(f"Best model was saved at {best_model_epoch} epochs\n")
    model.load_state_dict(best_model['model_state_dict'])

    sim_score = []
    def evaluate(model, dataloader, dataset, device):
        model.eval()
        running_loss = 0.0
        counter = 0
        latent_space_val = {}
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
                counter += 1
                latent_space_patch = {}
                for patch_index in range(0, data.shape[1]):
                    batch_data = data[:, patch_index, :, :, :]
                    batch_data = batch_data.view(batch_data.shape[0], 1, patch_size, patch_size, patch_size)
                    batch_data = batch_data.to(device, dtype=torch.float)
                    recon, mu, logvar = model(batch_data)
                    loss = l1_loss(batch_data, recon) + kl_loss(mu, logvar)
                    mu = mu.cpu().detach().numpy()
                    latent_space_patch['patch_' + str(patch_index)] = mu
                    running_loss += loss.item()
                val_loss = running_loss / counter
                latent_space_val['batch_' + str(i)] = latent_space_patch
        return val_loss, latent_space_val


    temp_score = ssim3D(data, recon)
    sim_score.append(temp_score.cpu().detach().numpy())
    loss_valid, recon_images, original_images = evaluate(model, val_loader, val_dataset, device)
    results['latent_dim_'+str(latent_dim)] = sim_score


results.to_csv('sim_score_m00.csv', index=0)

