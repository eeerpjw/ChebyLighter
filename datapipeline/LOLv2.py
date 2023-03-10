# -*- coding: utf-8 -*-
import os
from PIL import Image
import random
from glob import glob
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LOLv2Dataset(Dataset):
    def __init__(self, data_dir, mode='train', gt_unpair=False):
        """
        init a LOL dataset obj
        :param data_dir: data location of LOL dataset
        :param transform: you should know what it means
        :param mode: 'train'/'eval'
        """
        self.mode = mode
        self.gt_unpair = gt_unpair
        self.to_tensor = transforms.ToTensor()
        self.root_dir = data_dir
        print(data_dir)
        self.path_real_train_l = os.path.join(self.root_dir, 'Real_captured/Train/Low/*.*')
        self.path_syn_train_l = os.path.join(self.root_dir, 'Synthetic/Train/Low/*.*')
        if mode == "train":
            self.low_names = glob(self.path_real_train_l) + glob(self.path_syn_train_l)
            self.high_names = glob(self.path_real_train_l.replace("Low","Normal")) \
                              + glob(self.path_syn_train_l.replace("Low","Normal"))
            print('[ * ] We got %d normal images and %d low light images for TRAIN' % (len(self.high_names), len(self.low_names)))
        elif mode == "eval":
            self.low_names = glob(self.path_real_train_l.replace("Train","Test")) \
                                              + glob(self.path_syn_train_l.replace("Train","Test"))
            self.high_names = glob(self.path_real_train_l.replace("Train","Test").replace("Low","Normal")) \
                                               + glob(self.path_syn_train_l.replace("Train","Test").replace("Low","Normal"))
            print('[ * ] We got %d normal images and %d low light images for EVAL' % (len(self.high_names), len(self.low_names)))
        self.high_names.sort()
        self.low_names.sort()
        print("???",mode,"low",self.low_names[0],self.low_names[-1])
        print("???", mode, "high", self.high_names[0], self.high_names[-1])
        #random.shuffle(self.low_names)
        assert len(self.high_names) == len(self.low_names),\
            "num of input must equal to that of gt, but %d != %d." % (len(self.low_names), len(self.high_names))
        # print('We got %d normal images and %d low light images' % (len(self.high_names), len(self.low_names)))
        if self.gt_unpair:
            self.high_names_unpair = self.high_names[:]

    def __getitem__(self, idx):
        """
        :param idx: idx for dataloader
        :return: paired image low light and normal light image
        """
        img_low_name = self.low_names[idx]
        img_high_name = self.high_names[idx]
        # print("[ * ]",img_low_name,img_high_name)
        img_low = Image.open(img_low_name)
        img_high = Image.open(img_high_name)

        img_resize = 400
        img_low = img_low.resize((img_resize, img_resize), Image.ANTIALIAS)
        img_low = ((np.asarray(img_low) / 255.0)-0.5)/0.5 # 归一化了
        # img_low = (np.asarray(img_low) / 255.0)
        img_low = torch.from_numpy(img_low).float()
        img_low = img_low.permute(2, 0, 1)

        img_high = img_high.resize((img_resize, img_resize), Image.ANTIALIAS)
        img_high = (np.asarray(img_high) / 255.0)
        img_high = torch.from_numpy(img_high).float()
        img_high = img_high.permute(2, 0, 1)

        if self.gt_unpair:
            img_high_name_unpair = random.choice(self.high_names_unpair)
            img_high_unpair = Image.open(img_high_name_unpair)
            img_high_unpair = img_high_unpair.resize((img_resize, img_resize), Image.ANTIALIAS)
            img_high_unpair = (np.asarray(img_high_unpair) / 255.0)
            img_high_unpair = torch.from_numpy(img_high_unpair).float()
            img_high_unpair = img_high_unpair.permute(2, 0, 1)
            out = {"l": img_low, "h": img_high, "h_unpair":img_high_unpair}
        else:
            out =  {"l": img_low, "h": img_high}

        return out

    def __len__(self):
        return max(len(self.high_names), len(self.low_names))


if __name__=="__main__":
    path = "/Users/liam/Documents/datasets/LOL-v2"
    transform_ = None
    lol = LOLv2Dataset(path, mode="train")
    dataloader = DataLoader(dataset=lol, batch_size=4, shuffle=False, num_workers=1)
    a = next(dataloader)



