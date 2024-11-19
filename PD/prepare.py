import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from tqdm.auto import tqdm

import utils.preprocessing as pp


def preprocess(csv_path, img_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path, low_memory=True)

    for i in tqdm(range(len(df))):
        file_name = str(df.iloc[i]['Subject']) + '.nii.gz'
        img_name = os.path.join(img_dir, os.path.basename(file_name))
        img = nib.load(img_name).get_fdata().astype('f4')

        # img = pp.center_crop(img, (10, 180, 180))
        # img = pp.resize(img, 0.5)
        # img = pp.minmax(img)

        PD = torch.tensor(df.iloc[i]['Group'])
        age = torch.tensor(df.iloc[i]['Age'])
        sex = torch.tensor(df.iloc[i]['Sex'])
        study = torch.tensor(df.iloc[i]['Study'])
        scanner_type = torch.tensor(df.iloc[i]['Type'])
        scanner_vendor = torch.tensor(df.iloc[i]['Vendor'])

        img = transforms.ToTensor()(img)
        torch.save((np.expand_dims(img, 0), PD, age, sex, study, scanner_type, scanner_vendor),
                   os.path.join(save_dir, f'{i}'))


if __name__ == '__main__':
    img_dir = '/data/Data/PD/images'

    preprocess('/data/Data/PD/train.csv', img_dir, '/data/Data/PD/train')
    preprocess('/data/Data/PD/val.csv', img_dir, '/data/Data/PD/val')
    preprocess('/data/Data/PD/test.csv', img_dir, '/data/Data/PD/test')
