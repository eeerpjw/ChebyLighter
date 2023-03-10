import os
from PIL import Image
from glob import glob
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LOLDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        """
        init a LOL dataset obj
        :param data_dir: data location of LOL dataset
        :param transform: you should know what it means
        :param mode: 'train'/'eval'
        """
        self.mode = mode

        self.to_tensor = transforms.ToTensor()
        self.root_dir = data_dir
        self.path_real = os.path.join(self.root_dir, 'our485')
        self.path_syn = os.path.join(self.root_dir, 'syn')
        self.path_eval = os.path.join(self.root_dir, 'eval15')
        if mode == "train":
            self.high_names = glob(self.path_real + '/high/*.png') # + glob(self.path_syn + '/high/*.png')
            self.low_names = glob(self.path_real + '/low/*.png') # + glob(self.path_syn + '/low/*.png')
            print('[ * ] We got %d normal images and %d low light images for TRAIN' % (len(self.high_names), len(self.low_names)))
        elif mode == "eval":
            self.high_names = glob(self.path_eval + '/high/*.png')
            self.low_names = glob(self.path_eval + '/low/*.png')
            print('[ * ] We got %d normal images and %d low light images for EVAL' % (len(self.high_names), len(self.low_names)))
        self.high_names.sort()
        self.low_names.sort()
        #random.shuffle(self.low_names)
        assert len(self.high_names) == len(self.low_names),\
            "num of input must equal to that of gt, but %d != %d." % (len(self.low_names), len(self.high_names))
        # print('We got %d normal images and %d low light images' % (len(self.high_names), len(self.low_names)))

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
        img_low = torch.from_numpy(img_low).float()
        img_low = img_low.permute(2, 0, 1)

        img_high = img_high.resize((img_resize, img_resize), Image.ANTIALIAS)
        img_high = (np.asarray(img_high) / 255.0)
        img_high = torch.from_numpy(img_high).float()
        img_high = img_high.permute(2, 0, 1)
        
        return {"l": img_low, "h": img_high}

    def __len__(self):
        return max(len(self.high_names), len(self.low_names))

