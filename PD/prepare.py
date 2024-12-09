import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


def preprocess(csv_path, img_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path, low_memory=True)

    for i in tqdm(range(len(df))):
        file_name = str(df.iloc[i]['Subject']) + '.nii.gz'
        img_name = os.path.join(img_dir, os.path.basename(file_name))
        img = nib.load(img_name).get_fdata().astype('f4')
        # print(img.shape)
        # img = pp.center_crop(img, (10, 180, 180))
        # img = pp.resize(img, 0.5)
        # img = pp.minmax(img)
        # img = np.swapaxes(img, 1,2)

        PD = torch.tensor(df.iloc[i]['Group'])
        # age = torch.tensor(df.iloc[i]['Age'])
        sex = torch.tensor(df.iloc[i]['Sex'])
        study = torch.tensor(df.iloc[i]['Study'])
        scanner_type = torch.tensor(df.iloc[i]['Type'])
        # scanner_vendor = torch.tensor(df.iloc[i]['Vendor'])

        img = torch.tensor(np.expand_dims(img, 0))
        torch.save((img, PD, sex, study, scanner_type),
                   os.path.join(save_dir, f'{i}'))


if __name__ == '__main__':
    img_dir = '/data/Data/PD/images'

    preprocess('/data/Data/PD/train.csv', img_dir, '/data/Data/PD/train')
    preprocess('/data/Data/PD/val.csv', img_dir, '/data/Data/PD/val')
    preprocess('/data/Data/PD/test.csv', img_dir, '/data/Data/PD/test')
