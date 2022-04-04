import torch
import get_data
from torch.utils import data
import nibabel as nib
import glob
import numpy as np
import os
from tqdm import tqdm
from model import VAE
import torch.nn as nn
import pandas as pd
from save_model import SaveBestModel, save_model
from loss import L1Loss, KLDivergence
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dataloder_img(data.Dataset):
    def __init__(self, path, subjects_data, target_variable):
        self.files = glob.glob(path)
        self.files = np.sort(self.files)
        self.subject_data = subjects_data
        self.target_variable = target_variable
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        image_name = self.files[idx]
        subject_id= os.path.basename(image_name).split('_')[3]
        subject_id = os.path.basename(subject_id).split('.')[0]
        img = nib.load(image_name).get_fdata()
        # change to numpy
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        img = torch.from_numpy(img)
        # change to numpy
        return img

path = '/home/aditya/Documents/DELCODE-GP_MKL/data/unmodulated_segments/2mm/wp1*'

delcode_cov = 'data/delcode_cov1079.mat'
delcode_data, hippo_data = get_data.get_data_mat(delcode_cov)


l1_loss = L1Loss()
kl_loss = KLDivergence()

mask_img = 'data/mask.nii'
mask_data = nib.load(mask_img).get_fdata()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mask_data = mask_data.reshape(1, 1, mask_data.shape[0], mask_data.shape[1], mask_data.shape[2])
mask_data = torch.tensor(mask_data)
mask_data = mask_data.to(device, dtype=torch.float)


def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="jet", origin="lower")

# Training
def train(model, dataloader, dataset, device, optimizer):
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        data = data.to(device, dtype=torch.float)
        data = data * mask_data
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        recon = recon*mask_data
        loss = l1_loss(data, recon) + kl_loss(mu, logvar)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter
    return train_loss


def validate(model, dataloader, dataset, device):
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
            counter += 1
            data = data.to(device, dtype=torch.float)
            data = data * mask_data
            recon, mu, logvar = model(data)
            recon = recon * mask_data
            loss = l1_loss(data, recon) + kl_loss(mu, logvar)
            running_loss += loss.item()
    val_loss = running_loss / counter
    return val_loss


# Hyper-parameters
batch_size = 32
learning_rate =  0.0001
full_dataset = Dataloder_img(path, delcode_data, 'age')

train_size = int(0.75 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_loader = data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size)


latent_dim = 128
model = VAE(latent_dim=latent_dim)
model= nn.DataParallel(model)
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)

# initialize SaveBestModel class
save_best_model = SaveBestModel()
save_model_path = 'outputs/dim_'+str(latent_dim)
lr = 0.001
epochs = 50
train_loss = []
valid_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(model, train_loader, train_dataset, device, opt)
    valid_epoch_loss= validate(model, val_loader, val_dataset, device)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    save_best_model(valid_epoch_loss, epoch, model, opt, save_model_path)
    print('-' * 50)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")

save_model(epochs, model, opt, save_model_path)

loss_data = pd.DataFrame()
loss_data['training_loss'] = train_loss
loss_data['validation_loss'] = valid_loss


save_path = 'results/'
loss_data.to_csv(save_path+'loss_latent_dim_'+str(latent_dim)+'.csv', index=0)
