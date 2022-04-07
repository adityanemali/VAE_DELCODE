from torch.utils import data
import glob
import numpy as np
import nibabel as nib
import os
import torch

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
