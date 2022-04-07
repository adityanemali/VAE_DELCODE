import torch
import get_data
from torch.utils import data
import nibabel as nib
import glob
import numpy as np
import os
from tqdm import tqdm
from patch_model import VAE
import torch.nn as nn
import pandas as pd
from save_model import SaveBestModel, save_model
from loss import L1Loss, KLDivergence
import matplotlib.pyplot as plt
from monai.transforms import CropForeground
import torch.nn.functional as F
import pickle
import os
from patchify import patchify, unpatchify

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mask  = 'data/icvmask_4mm.nii'
masked_data = nib.load(mask).get_fdata()

def threshold_at_one(x):
    # threshold at 1
    return x > 0

def patch_img(image, patch_size, step, reconstruct=False):
    # padding need for patches (works only for 16)
    image = np.pad(image, ((2, 3), (0, 0), (0, 0)), 'constant')
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
        cropper = CropForeground(select_fn=threshold_at_one, k_divisible=16, margin=0)
        img = cropper(img)
        img_patches = patch_img(img, patch_size, patch_step, reconstruct=False)
        img_patches = torch.from_numpy(img_patches)
        return img_patches, subject_id

path = '/home/aditya/Documents/DELCODE-GP_MKL/data/unmodulated_segments/4mm/wp1*'

delcode_cov = 'data/delcode_cov1079.mat'
delcode_data, hippo_data = get_data.get_data_mat(delcode_cov)

l1_loss = L1Loss()
kl_loss = KLDivergence()

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="jet", origin="lower")

def plot_slices_save(data):
        shp1 = round(data.shape[0] / 2)
        shp2 = round(data.shape[1] / 2)
        shp3 = round(data.shape[2] / 2)
        slice_0 = data[shp1, :, :]
        slice_1 = data[:, shp2, :]
        slice_2 = data[:, :, shp3]
        show_slices([slice_0, slice_1, slice_2])
        plt.show()
        plt.close()


# Training
def train(model, dataloader, dataset, device, optimizer):
    model.train()
    running_loss = 0.0
    counter = 0
    latent_space_train = {}
    subjects_train = {}
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        y = data[1]
        data = data[0]
        counter += 1
        latent_space_patch = {}
        for patch_index in range(0, data.shape[1]):
            batch_data = data[:, patch_index, :, :, :]
            batch_data = batch_data.view(batch_data.shape[0], 1, patch_size, patch_size, patch_size)
            batch_data = batch_data.to(device, dtype=torch.float)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch_data)
            loss = l1_loss(batch_data, recon) + kl_loss(mu, logvar)
            mu = mu.cpu().detach().numpy()
            latent_space_patch['patch_'+str(patch_index)] = mu
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        train_loss = running_loss / counter
        latent_space_train['batch_'+str(i)] = latent_space_patch
        subjects_train['batch_'+str(i)] = y
    return train_loss, latent_space_train, subjects_train

def validate(model, dataloader, dataset, device):
    model.eval()
    running_loss = 0.0
    counter = 0
    latent_space_val = {}
    subjects_val = {}
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
            y = data[1]
            data = data[0]
            counter += 1
            latent_space_patch = {}
            for patch_index in range(0, data.shape[1]):
                batch_data = data[:, patch_index, :, :, :]
                batch_data = batch_data.view(batch_data.shape[0], 1, patch_size, patch_size, patch_size)
                batch_data = batch_data.to(device, dtype=torch.float)
                recon, mu, logvar = model(batch_data)
                loss = l1_loss(batch_data, recon) + kl_loss(mu, logvar)
                mu = mu.cpu().detach().numpy()
                latent_space_patch['patch_'+str(patch_index)] = mu
                running_loss += loss.item()
            val_loss = running_loss / counter
            latent_space_val['batch_'+str(i)] = latent_space_patch
            subjects_val['batch_' + str(i)] = y
    return val_loss, latent_space_val, subjects_val


# Hyper-parameters
batch_size = 32
learning_rate =  0.0001
patch_size = 16
patch_step = 8
full_dataset = NiftiDataloader(path, patch_size, patch_step)

train_size = int(0.80 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_loader = data.DataLoader(val_dataset, shuffle=True, batch_size=batch_size)

dims = [1, 2, 4, 8, 16, 32, 64, 128]

for dim in dims:
    print("Running model, latent dimension:", dim)
    print("---------------------------------------------------------")
    latent_dim = dim
    model = VAE(latent_dim=latent_dim)
    model= nn.DataParallel(model)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    # initialize SaveBestModel class
    save_best_model = SaveBestModel()
    save_model_path = 'outputs1/dim_'+str(latent_dim)
    lr = 0.0001
    epochs = 50
    train_loss = []
    valid_loss = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss, latent_space_train, subjects_train = train(model, train_loader, train_dataset, device, opt)
        valid_epoch_loss, latent_space_val, subjects_val = validate(model, val_loader, val_dataset, device)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        save = save_best_model(valid_epoch_loss, epoch, model, opt, save_model_path)
        if(save):
            store_path = '/storage/DELCODE/VAE/results/'
            if (os.path.exists(store_path)):
                with open(store_path + '/' + 'train_latent_space_' + str(latent_dim)+'.pkl', 'wb+') as f:
                    pickle.dump(latent_space_train, f)
                with open(store_path + '/' + 'val_latent_space_' + str(latent_dim) + '.pkl', 'wb+') as f:
                    pickle.dump(latent_space_val, f)
                with open(store_path + '/' + 'subj_train_latent_space_' + str(latent_dim) + '.pkl', 'wb+') as f:
                    pickle.dump(subjects_train, f)
                with open(store_path + '/' + 'subj_val_latent_space_' + str(latent_dim) + '.pkl', 'wb+') as f:
                    pickle.dump(subjects_val, f)

            else:
                with open(store_path + '/' + 'train_latent_space_' + str(latent_dim) + '.pkl', 'wb+') as f:
                    pickle.dump(latent_space_train, f)
                with open(store_path + '/' + 'val_latent_space_' + str(latent_dim) + '.pkl', 'wb+') as f:
                    pickle.dump(latent_space_val, f)
                with open(store_path + '/' + 'subj_train_latent_space_' + str(latent_dim) + '.pkl', 'wb+') as f:
                    pickle.dump(subjects_train, f)
                with open(store_path + '/' + 'subj_val_latent_space_' + str(latent_dim) + '.pkl', 'wb+') as f:
                    pickle.dump(subjects_val, f)
        print('-' * 50)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Loss: {valid_epoch_loss:.4f}")

    save_model(epochs, model, opt, save_model_path)

    loss_data = pd.DataFrame()
    loss_data['training_loss'] = train_loss
    loss_data['validation_loss'] = valid_loss


    save_path = 'results1/'
    loss_data.to_csv(save_path+'loss_latent_dim_'+str(latent_dim)+'.csv', index=0)
    print("Successful, latent dimension:", dim)
    print("---------------------------------------------------------")