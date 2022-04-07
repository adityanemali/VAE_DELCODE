import pandas as pd
from model import VAE
import torch.nn as nn
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from dataloader import Dataloder_img
from sim3D import ssim3D
import  get_data
from loss import L1Loss, KLDivergence
from torch.utils import data
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


path = '/storage/DELCODE/long_data/SPM_CATr1888avg_Shoot_DCTemplaterp123_1p0mm_mwmp1avg/longit_3scans/rmwmp1avg_DC_*_m00.nii'
delcode_cov = 'data/delcode_cov1079.mat'
delcode_data, hippo_data = get_data.get_data_mat(delcode_cov)


l1_loss = L1Loss()
kl_loss = KLDivergence()

batch_size = 1
learning_rate =  0.001
full_dataset = Dataloder_img(path, delcode_data, 'age')

train_size = int(0 * len(full_dataset))
val_size = len(full_dataset)
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
val_loader = data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size)


results = pd.DataFrame()
latent_dims = [1, 2, 4, 8, 16, 32, 64, 128]
for lat in latent_dims:

    latent_dim = lat # latent dimension for sampling

    model = VAE(latent_dim)
    model= nn.DataParallel(model)
    model.to(device)

    # select best model
    best_model = torch.load('outputs/dim_'+str(lat)+'/best_model.pth')
    best_model_epoch = best_model['epoch']
    print(f"Best model was saved at {best_model_epoch} epochs\n")
    model.load_state_dict(best_model['model_state_dict'])

    mask_img = 'data/icvmask_4mm.nii'
    mask_data = nib.load(mask_img).get_fdata()
    mask_dat = mask_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mask_data = mask_data.reshape(1, 1, mask_data.shape[0], mask_data.shape[1], mask_data.shape[2])
    mask_data = torch.tensor(mask_data)
    mask_data = mask_data.to(device, dtype=torch.float)

    sim_score = []
    def evaluate(model, dataloader, dataset, device):
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
                temp_score = ssim3D(data, recon)
                sim_score.append(temp_score.cpu().detach().numpy())
        val_loss = running_loss / counter
        return val_loss, recon, data

    loss_valid, recon_images, original_images = evaluate(model, val_loader, val_dataset, device)
    results['latent_dim_'+str(latent_dim)] = sim_score


results.to_csv('sim_score_m00.csv', index=0)

