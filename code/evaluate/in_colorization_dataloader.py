"""Based on https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer/blob/main/data/pascal.py
"""
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from itertools import combinations,permutations

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import json



class DatasetColorization(Dataset):
    def __init__(self, datapath, image_transform, mask_transform, padding: bool = 1,
                 use_original_imgsize: bool = False, flipped_order: bool = False,
                 reverse_support_and_query: bool = False, random: bool = False,
                 shots: int = 4, use_class: bool = False, random_num: int = 0, choose_num: int =0, trn: bool = False,val_num: int = 50000,
                 dev_list = [],aug=False,sim=False):
        self.padding = padding
        self.random = random
        self.use_original_imgsize = use_original_imgsize
        self.image_transform = image_transform
        self.reverse_support_and_query = reverse_support_and_query
        self.mask_transform = mask_transform
        # self.ds = ImageFolder(os.path.join(datapath,'Imagenet', 'val'))
        # self.ds_train = ImageFolder(os.path.join(datapath,'Imagenet', 'train'))
        self.ds = ImageFolder(os.path.join(datapath,'Imagenet1K', 'val'))
        self.ds_train = ImageFolder(os.path.join(datapath,'Imagenet1K', 'train'))

        self.aug = aug
        self.sim = sim

        self.flipped_order = flipped_order

        self.val_num = val_num

        self.img_metadata = list(range(self.val_num))
        assert not (random_num > 0 and choose_num > 0)
        self.use_class = use_class
        self.choose_num = choose_num
        self.random_num = random_num

        self.img_metadata = [34767]

        demonstration = np.random.choice(range(len(self.ds_train)), shots, replace=False)
        if self.use_class:
            demonstration = []
            for key,value in self.ds_train.class_to_idx.items() :
                demonstration += np.random.choice([ index for index in range(len(self.ds_train.imgs)) if self.ds_train.imgs[index][1] == value], shots//len(self.ds_train.class_to_idx), replace=False).tolist()
        self.demonstration = demonstration
        # print("demonstration, ",demonstration)
        # print([self.ds_train.imgs[idx] for idx in demonstration])
        # assert 0

        choose_index = np.random.choice(demonstration, choose_num, replace=False)
        print("choose_index, ",choose_index)
        rest_demon = list(set(demonstration)-set(choose_index))

        if trn:
            new_img_metadata = []
            new_demonstration = []
            for now in range(choose_num):
                for now_set in combinations(choose_index, now+1):
                    new_img_metadata += rest_demon
                    new_demonstration += [list(now_set)] * len(rest_demon)
                                    
            # print(self.demonstration)
            # print(self.img_metadata)
            self.val_len = len(new_demonstration)
            for now in range(choose_num):
                for now_set in combinations(choose_index, now+1):
                    new_img_metadata += list(range(self.val_num))
                    new_demonstration += [list(now_set)] * self.val_num
            self.demonstration = new_demonstration
            self.img_metadata = new_img_metadata
            assert len(self.img_metadata) == len(self.demonstration)
        self.trn = trn

        if self.sim:
            if shots == 20:
                json_path = "../imagenet/det-similarity-trn_20-3seed.json"
            if shots == 1000:
                json_path = "../imagenet/det-similarity-trn_1000-3seed.json"
                if use_class:
                    json_path = "../imagenet/det-similarity-class_trn_1000-3seed.json"
            # with open("../imagenet/det-similarity-trn_20-3seed.json",'r', encoding='UTF-8') as f:
            with open(json_path,'r', encoding='UTF-8') as f:
                sim_json = json.load(f)
            new_demonstration = []
            train_name = [self.ds_train.imgs[now][0] for now in self.demonstration]
            for name in self.img_metadata:
                # print(name)
                name = self.ds.imgs[name][0]
                for item in sim_json[name]:
                    if item in train_name:
                        new_demonstration += [[item]]
                        break
                # new_demonstration += [[item for item in sim_json[name] if item in [self.ds_train.imgs[now][0] for now in self.demonstration]][:1]]
                # new_demonstration += [[item for item in sim_json[name]][:1]]
            now_dev = []
            for names in new_demonstration:
                now = []
                for name in names:
                    for dev in self.demonstration:
                        if self.ds_train.imgs[dev][0] == name:
                            now.append(dev)
                now_dev.append(now)
            self.demonstration = now_dev
            self.trn = True
            self.val_len = 0

        if len(dev_list) != 0:
            now_dev = []
            for name in dev_list:
                for dev in self.demonstration:
                    if "train_"+str(dev) == name:
                        now_dev.append(dev)
                        break
            assert len(now_dev) == len(dev_list)
            self.demonstration = [now_dev] * len(self.img_metadata)
            self.trn = True
            self.val_len = 0
            assert len(self.img_metadata) == len(self.demonstration)

        print("train: {} ,val_num: {}, val: {} ".format(len(self.ds_train),self.val_num,len(self.ds)))
        print("img_metadata: {} ,demonstration: {} ".format(len(self.img_metadata),len(self.demonstration)))

        # self.indices = np.random.choice(np.arange(0, len(self.ds)-1), size=1000, replace=False)


    def __len__(self):
        return len(self.img_metadata)

    def create_grid_from_images(self, support_img, support_mask, query_img, query_mask):
        if self.aug:
            canvas = []
            site_lsit = [[support_img, support_mask, query_img, query_mask],
                        [support_mask,support_img, query_mask,query_img],
                        [query_mask, query_img, support_mask,support_img],
                        [query_img, query_mask, support_img, support_mask],
                        [support_mask, query_mask, support_img, query_img],
                        [query_mask, support_mask, query_img, support_img],
                        [query_img, support_img, query_mask, support_mask],
                        [support_img, query_img, support_mask, query_mask]] # 0 = left up 1 = right up 2 = left down
            for item in site_lsit:
                now = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding, 2 * support_img.shape[2] + 2 * self.padding))
                now[:, :support_img.shape[1], :support_img.shape[2]] = item[0]
                now[:, -query_img.shape[1]:, :query_img.shape[2]] = item[2]
                now[:, :support_img.shape[1], -support_img.shape[2]:] = item[1]
                now[:, -query_img.shape[1]:, -support_img.shape[2]:] = item[3]
                canvas.append(now)
        else:
            if self.reverse_support_and_query:
                support_img, support_mask, query_img, query_mask = query_img, query_mask, support_img, support_mask
            canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding,
                                2 * support_img.shape[2] + 2 * self.padding))
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
            if self.flipped_order:
                canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
                canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
                canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
            else:
                canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
                canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
                canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas

    def __getitem__(self, idx):
        if self.random_num > 0:
            query = self.ds[self.img_metadata[idx]]
            query_img, query_mask = self.mask_transform(query[0]), self.image_transform(query[0])
            query_name = 'val_' + str(self.img_metadata[idx])
            random_index = np.random.choice(self.demonstration, self.random_num, replace=False)

            support = []
            support_name = []
            for index in random_index:
                support.append(self.ds_train[index])
                support_name.append('train_'+str(index))
        else:
            if self.trn:
                if idx >= self.val_len:
                    query = self.ds[self.img_metadata[idx]]
                    query_img, query_mask = self.mask_transform(query[0]), self.image_transform(query[0])
                    query_name = 'val_' + str(self.img_metadata[idx])
                else:
                    query = self.ds_train[self.img_metadata[idx]]
                    query_img, query_mask = self.mask_transform(query[0]), self.image_transform(query[0])
                    query_name = 'train_' + str(self.img_metadata[idx])
            
                if isinstance(self.demonstration[idx],list):
                    support = []
                    support_name = []
                    for index in self.demonstration[idx]:
                        support.append(self.ds_train[index])
                        support_name.append('train_'+str(index))
                else:
                    support = [self.demonstration[idx]]
                    support_name = ['train_'+str(self.demonstration[idx])]

        grid = []
        for support_item in support:
            support_img, support_mask = self.mask_transform(support_item[0]), self.image_transform(support_item[0])
            grid.append(self.create_grid_from_images(support_img, support_mask, query_img, query_mask))
        
        
        # batch = {'query_img': query_img, 'query_mask': query_mask, 'support_img': support_img,
        #          'support_mask': support_mask, 'grid': grid,'query_name':query_name,'support_name':support_name}
        # print(query_name,support_name)
        batch = {'grid': grid,'query_name':query_name,'support_name':support_name}

        return batch