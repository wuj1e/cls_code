from __future__ import print_function, division
import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class dataloader(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self,
                 txt_path=None,
                 transform=None
                 ):
        self.list_sample = open(txt_path, 'r').readlines()

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))
        self.transform = transform

    def __len__(self):
        return self.num_sample
        # return 64

    def __getitem__(self, index):
        img_path = self.list_sample[index].split(',')[0]
        # print(index,len(self.list_sample))
        label = int(self.list_sample[index].split(',')[1].strip())

        # print("state",state)
        # print(files)
        img_cat = []

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img_cat.append(img)
            # print("item", item)

        if len(img_cat)>16:
            img_cat = img_cat[:16]
        if len(img_cat)<16:
            sub_len = 16-len(img_cat)
            for i in range(sub_len):
                img_cat.append(img_cat[-1])
        # img_cat = img_cat[:16]
        img_cat = torch.cat(img_cat, 0)
        # img = Image.open(self.list_sample[index].strip()).convert('RGB')
        # if '3' in self.list_sample[index]: # !!!  左右有病不是全部有病
        #     lab = 1
        #S
        # else:
        #     lab = 0
        #
        # # img = self.transform(img)

        return img, label
