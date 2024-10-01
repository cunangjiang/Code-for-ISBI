from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image
from enhance.process_ref import ProcssRef



class BraTs_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(BraTs_datasets, self)
        if train:
            ori_list = sorted(os.listdir(path_Data+'train/oriT2/'))
            ref_list = sorted(os.listdir(path_Data+'train/oriT1/'))
            ref_lr_list = sorted(os.listdir(path_Data+'train/orLRbicT1/x'+str(config.upscale)+'/'))
            tar_list = sorted(os.listdir(path_Data+'train/orLRbicT2/x'+str(config.upscale)+'/'))
            self.data = []
            for i in range(len(ori_list)):
                img_name = ori_list[i]
                ori_path = path_Data+'train/oriT2/' + ori_list[i]
                ref_path = path_Data+'train/oriT1/' + ref_list[i]
                ref_lr_path = path_Data+'train/orLRbicT1/x'+str(config.upscale)+'/' + ref_lr_list[i]
                tar_path = path_Data+'train/orLRbicT2/x'+str(config.upscale)+'/' + tar_list[i]
                self.data.append([ori_path, ref_path, ref_lr_path, tar_path, img_name])
            self.transformer = config.train_transformer
        else:
            ori_list = sorted(os.listdir(path_Data+'val/oriT2/'))
            ref_list = sorted(os.listdir(path_Data+'val/oriT1/'))
            ref_lr_list = sorted(os.listdir(path_Data+'val/orLRbicT1/x'+str(config.upscale)+'/'))
            tar_list = sorted(os.listdir(path_Data+'val/orLRbicT2/x'+str(config.upscale)+'/'))
            self.data = []
            for i in range(len(ori_list)):
                img_name = ori_list[i]
                ori_path = path_Data+'val/oriT2/' + ori_list[i]
                ref_path = path_Data+'val/oriT1/' + ref_list[i]
                ref_lr_path = path_Data+'val/orLRbicT1/x'+str(config.upscale)+'/' + ref_lr_list[i]
                tar_path = path_Data+'val/orLRbicT2/x'+str(config.upscale)+'/' + tar_list[i]
                self.data.append([ori_path, ref_path, ref_lr_path, tar_path, img_name])
            self.transformer = config.test_transformer

        # Initialize the ProcssRef module
        self.processref = ProcssRef(radius=50, sigma=0, extra_sharpen_time=2)
        
    def __getitem__(self, indx):
        ori_path, ref_path, ref_lr_path, tar_path, img_name = self.data[indx]
        ori = Image.open(ori_path).convert('RGB')
        ref = Image.open(ref_path).convert('RGB')
        ref_lr = Image.open(ref_lr_path).convert('RGB')
        tar = Image.open(tar_path).convert('RGB')
        
        ori = self.transformer(ori)
        ref = self.transformer(ref)
        ref_lr = self.transformer(ref_lr)
        tar = self.transformer(tar)
        return ori, ref, ref_lr, tar, img_name

    def __len__(self):
        return len(self.data)


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
        
    
