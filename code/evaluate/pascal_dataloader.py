"""Based on https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer/blob/main/data/pascal.py
"""
import os
from PIL import Image
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset
from mae_utils import PURPLE, YELLOW
from itertools import combinations,permutations
import math
import json


def create_grid_from_images_old(canvas, support_img, support_mask, query_img, query_mask,aug=False):
    if aug:
        canvas_ori = canvas
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
            # now = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding, 2 * support_img.shape[2] + 2 * self.padding))
            now = canvas_ori.clone()
            now[:, :support_img.shape[1], :support_img.shape[2]] = item[0]
            now[:, -query_img.shape[1]:, :query_img.shape[2]] = item[2]
            now[:, :support_img.shape[1], -support_img.shape[2]:] = item[1]
            now[:, -query_img.shape[1]:, -support_img.shape[2]:] = item[3]
            canvas.append(now)
    else:
        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
        canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
        canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
        canvas[:, -query_img.shape[1]:, -query_img.shape[2]:] = query_mask
    return canvas

class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, image_transform, mask_transform, padding: bool = 1, use_original_imgsize: bool = False,
                 flipped_order: bool = False,
                 reverse_support_and_query: bool = False, random: bool = False, ensemble: bool = False, purple: bool = False,
                 shots: int = 4, use_class: bool = False, random_num: int = 0, choose_num: int =0, trn: bool = False,
                 dev_list = [],aug=False,sim=False,supsim=False):
        self.fold = fold
        self.nfolds = 4
        self.flipped_order = flipped_order
        self.nclass = 20
        self.padding = padding
        self.random = random
        self.ensemble = ensemble
        self.purple = purple
        self.use_original_imgsize = use_original_imgsize

        self.img_path = os.path.join(datapath, 'voc-2012/VOCdevkit/VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'voc-2012/SegmentationClassAug/')
        self.image_transform = image_transform
        self.reverse_support_and_query = reverse_support_and_query
        self.mask_transform = mask_transform

        self.class_ids = self.build_class_ids()
        assert not (random_num > 0 and choose_num > 0)
        self.use_class = use_class
        self.choose_num = choose_num
        self.random_num = random_num
        self.img_metadata,self.demonstration = self.build_img_metadata(shots,use_class=use_class)

        self.aug = aug
        self.sim = sim
        self.supsim = supsim

        # new_test = []
        # for name in ['2007_000676']:
        # # for name in ['2007_000042', '2007_000629', '2007_000636', '2007_000676', '2007_001175', '2007_001458', '2007_001678', '2007_001717', '2007_002119', '2007_002284', '2007_000187', '2007_000661', '2007_000804', '2007_001408', '2007_001430', '2007_001594', '2007_001763', '2007_002268', '2007_002378', '2007_002426']:
        #     for dev in self.img_metadata:
        #         if dev[0] == name:
        #             new_test.append(dev)
        #             break
        # self.img_metadata = new_test
    
        choose_index = np.random.choice(range(len(self.demonstration)), choose_num, replace=False)
        print(self.demonstration,choose_index)
        print("choose_index, ",choose_index)
        rest_demon = [self.demonstration[index] for index in range(len(self.demonstration)) if index not in choose_index]
        if trn:
            new_img_metadata = []
            new_demonstration = []
            for now in range(choose_num):
                for now_set in combinations(choose_index, now+1):
                    
                    new_img_metadata += rest_demon
                    new_demonstration += [[self.demonstration[index] for index in list(now_set)]] * len(rest_demon)
                                    
            for now in range(choose_num):
                for now_set in combinations(choose_index, now+1):
                    new_img_metadata += self.img_metadata
                    new_demonstration += [[self.demonstration[index] for index in list(now_set)]] * len(self.img_metadata)
            
            ####### one 
            # for item in self.demonstration:
            #     new_img_metadata += self.img_metadata
            #     new_demonstration += [[item]] * len(self.img_metadata)
            ####### two
            # new_img_metadata = []
            # new_demonstration = []
            # for now in range(2):
            #     for now_set in combinations(choose_index, now+1):
            #         new_img_metadata += self.img_metadata
            #         new_demonstration += [[self.demonstration[index] for index in list(now_set)]] * len(self.img_metadata)
                    
            self.demonstration = new_demonstration
            self.img_metadata = new_img_metadata
        self.trn = trn

        if len(dev_list) != 0:
            if not isinstance(dev_list[0],list):
                now_dev = []
                for name in dev_list:
                    for dev in self.demonstration:
                        if dev[0] == name:
                            now_dev.append(dev)
                            break
                assert len(now_dev) == len(dev_list)
                self.demonstration = [now_dev] * len(self.img_metadata)
                self.trn = True
            else:
                new_dev = []
                for item in range(len(dev_list)):
                    now_dev = []
                    for name in dev_list[item]:
                        for dev in self.demonstration:
                            if dev[0] == name:
                                now_dev.append(dev)
                                break
                    assert len(now_dev) == len(dev_list[item])
                    new_dev.append(now_dev)
                self.demonstration = new_dev
                self.trn = True
       
        if self.sim:
            with open("../pascal-5i_un/VOC2012/features_vit-laion2b_val/folder{}-similarity.json".format(fold),'r', encoding='UTF-8') as f:
                sim_json = json.load(f)
            new_demonstration = []
            for name in self.img_metadata:
                name = name[0]
                new_demonstration += [[item for item in sim_json[name] if item in [now[0] for now in self.demonstration]][:1]]
            now_dev = []
            for names in new_demonstration:
                now = []
                for name in names:
                    for dev in self.demonstration:
                        if dev[0] == name:
                            now.append(dev)
                now_dev.append(now)
            self.demonstration = now_dev
            self.trn = True


        # if self.sim:
        #     with open("../pascal-5i_un/VOC2012/features_vit-laion2b_val/folder{}-similarity.json".format(fold),'r', encoding='UTF-8') as f:
        #         sim_json = json.load(f)
        #     new_demonstration = []
        #     new_img_metadata = []
        #     for item in self.img_metadata:
        #         new_demonstration += self.demonstration
        #         new_img_metadata += [item] * len(self.demonstration)
        #     self.img_metadata = new_img_metadata
        #     self.demonstration = new_demonstration
        #     self.trn = False

        if self.supsim:
            with open("../pascal-5i/VOC2012/features_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder_val/folder{}-similarity.json".format(fold),'r', encoding='UTF-8') as f:
                sim_json = json.load(f)
            new_demonstration = []
            for name in self.img_metadata:
                name = name[0]
                new_demonstration += [[item for item in sim_json[name] if item in [now[0] for now in self.demonstration]][:1]]
            now_dev = []
            for names in new_demonstration:
                now = []
                for name in names:
                    for dev in self.demonstration:
                        if dev[0] == name:
                            now.append(dev)
                now_dev.append(now)
            self.demonstration = now_dev
            self.trn = True

        print("img_metadata: {} ,demonstration: {} ".format(len(self.img_metadata),len(self.demonstration)))
        assert len(self.img_metadata) == len(self.demonstration)

        # demonstration_index = np.random.choice(range(len(demonstration)), shots, replace=False)
        #     print(demonstration_index)
        #     demonstration = [demonstration[index] for index in demonstration_index]
        # if trn:
        #     new_img_metadata = []
        #     new_demonstration = []
        #     for now in range(1,shots):
        #         for now_set in permutations(list(range(shots)), now):
        #             new_img_metadata += [self.demonstration[index] for index in list(set(range(shots)) - set(now_set))]
        #             new_demonstration += [[self.demonstration[index] for index in list(now_set)]] * \
        #                                 int(shots - len(now_set))
        #     # print(self.demonstration)
        #     # print(self.img_metadata)
        #     self.val_len = len(self.demonstration)
        #     for now in range(1,shots):
        #         for now_set in permutations(list(range(shots)), now):
        #             new_img_metadata += self.img_metadata
        #             new_demonstration += [[self.demonstration[index] for index in list(now_set)]] * len(self.img_metadata)
        #     self.demonstration = new_demonstration
        #     self.img_metadata = new_img_metadata
        # self.trn = trn
        # print("img_metadata: {} ,demonstration: {} ".format(len(self.img_metadata),len(self.demonstration)))
        # assert len(self.img_metadata) == len(self.demonstration)

    def __len__(self):
        return len(self.img_metadata)

    def create_grid_from_images(self, support_img, support_mask, query_img, query_mask,flip: bool = False):
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
            canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding, 2 * support_img.shape[2] + 2 * self.padding))
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
            if flip:
                canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
                canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
                canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
            else:
                canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
                canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
                canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas

    def __getitem__(self, idx):
        query_name, support_name, class_sample_query, class_sample_support = self.sample_episode(idx)
        # print(query_name,support_name,class_sample_query, class_sample_support)
        query_img, query_cmask, support_img, support_cmask, org_qry_imsize = self.load_frame(query_name, support_name)
        # print(query_img, query_cmask, support_img, support_cmask, org_qry_imsize)

        if self.image_transform:
            query_img = self.image_transform(query_img)
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask, class_sample_query, purple=self.purple)
        if self.mask_transform:
            query_mask = self.mask_transform(query_mask)
                
        
        if isinstance(support_cmask,list):
            support_mask, support_ignore_idx = [],[]
            support_img_new,support_mask_new = [],[]
            for i in range(len(support_cmask)):
                now_mask,now_idx = self.extract_ignore_idx(support_cmask[i], class_sample_support[i],
                                                                       purple=self.purple)
                support_ignore_idx.append(now_idx)
                if self.image_transform:
                    support_img_new.append(self.image_transform(support_img[i]))
                if self.mask_transform:
                    support_mask_new.append(self.mask_transform(now_mask))
            support_img,support_mask = support_img_new,support_mask_new
        else:
            support_mask, support_ignore_idx = self.extract_ignore_idx(support_cmask, class_sample_support, purple=self.purple)
            if self.image_transform:
                support_img = self.image_transform(support_img)
            if self.mask_transform:
                support_mask = self.mask_transform(support_mask)

        
        if isinstance(support_mask, list):
            grid = []
            for i in range(len(support_mask)):
                now_grid = self.create_grid_from_images(support_img[i], support_mask[i], query_img, query_mask,
                                                    flip=self.flipped_order)
                grid.append(now_grid)
        else:
            grid = self.create_grid_from_images(support_img, support_mask, query_img, query_mask, flip=self.flipped_order)
        # if self.ensemble:
        #     grid2 = self.create_grid_from_images(support_img, support_mask, query_img, query_mask, (not self.flipped_order))
        #
        #
        #     support_purple_mask, _ = self.extract_ignore_idx(support_cmask, class_sample_support,
        #                                                                purple=True)
        #     if self.mask_transform:
        #         support_purple_mask = self.mask_transform(support_purple_mask)
        #
        #     grid3 = self.create_grid_from_images(support_img, support_purple_mask, query_img, query_mask,
        #                                         flip=self.flipped_order)
        #
        #     grid4 = self.create_grid_from_images(support_img, support_purple_mask, query_img, query_mask,
        #                                         flip=(not self.flipped_order))
        #
        #
        #     grid = grid, grid2, grid3, grid4
        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,
                 'org_query_imsize': org_qry_imsize,
                 'support_img': support_img,
                 'support_mask': support_mask,
                 'support_name': support_name,
                 'support_ignore_idx': support_ignore_idx,
                 'class_id': torch.tensor(class_sample_query),
                 'grid': grid}

        return batch

    def extract_ignore_idx(self, mask, class_id, purple):
        mask = np.array(mask)
        boundary = np.floor(mask / 255.)
        if not purple:
            mask[mask != class_id + 1] = 0
            mask[mask == class_id + 1] = 255
            return Image.fromarray(mask), boundary
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x,y] != class_id + 1:
                    color_mask[x, y] = np.array(PURPLE)
                else:
                    color_mask[x, y] = np.array(YELLOW)
        return Image.fromarray(color_mask), boundary


    def load_frame(self, query_name, support_name):
        if isinstance(support_name,list):
            support_img = [self.read_img(support) for support in support_name]
            support_mask = [self.read_mask(support) for support in support_name]
        else:
            support_img = self.read_img(support_name)
            support_mask = self.read_mask(support_name)
        
        if isinstance(query_name,list):
            assert len(query_name) == 1
            query_img = self.read_img(query_name[0])
            query_mask = self.read_mask(query_name[0])
            org_qry_imsize = query_img.size
        else:
            query_img = self.read_img(query_name)
            query_mask = self.read_mask(query_name)
            org_qry_imsize = query_img.size

        return query_img, query_mask, support_img, support_mask, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = Image.open(os.path.join(self.ann_path, img_name) + '.png')
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx):
        """Returns the index of the query, support and class."""
        query_name, class_sample = self.img_metadata[idx]
        if self.random_num > 0:
            support_name, support_class = np.random.choice(self.demonstration, self.random_num)
        else:
            if self.trn:
                support_name, support_class = [index[0] for index in self.demonstration[idx]],[index[1] for index in self.demonstration[idx]]
            else:
                support_name, support_class = self.demonstration[idx]
        return query_name, support_name, class_sample, support_class

    # def sample_episode(self, idx):
    #     """Returns the index of the query, support and class."""
    #     query_name, class_sample = self.img_metadata[idx]
    #     if not self.random:
    #         support_class = class_sample
    #     else:
    #         support_class = np.random.choice([k for k in self.img_metadata_classwise.keys() if self.img_metadata_classwise[k]], 1, replace=False)[0]
    #     while True:  # keep sampling support set if query == support
    #         support_name = np.random.choice(self.img_metadata_classwise[support_class], 1, replace=False)[0]
    #         if query_name != support_name:
    #             break
    #     return query_name, support_name, class_sample, support_class

    # def sample_episode(self, idx):
    #     """Returns the index of the query, support and class."""
    #     query_name, class_sample = self.img_metadata[idx]
    #     if not self.random:
    #         support_class = class_sample
    #     else:
    #         support_class = np.random.choice([k for k in self.img_metadata_classwise.keys() if self.img_metadata_classwise[k]], 1, replace=False)[0]
    #     while True:  # keep sampling support set if query == support
    #         support_name = np.random.choice(self.img_metadata_classwise[support_class], 1, replace=False)[0]
    #         if query_name != support_name:
    #             break
    #     return query_name, support_name, class_sample, support_class

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        return class_ids_val

    def build_img_metadata(self,shots,use_class=False):

        def read_metadata(split, fold_id):
            cwd = os.path.dirname(os.path.abspath(__file__))
            fold_n_metadata = os.path.join(cwd, 'splits/pascal/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = read_metadata('val', self.fold)
        demonstration = read_metadata('trn', self.fold)

        print('Total (val) images are : %d' % len(img_metadata))
        print('Total (trn) images are : %d' % len(demonstration))
        img_metadata_classwise = {}
        if self.use_class:
            new_demonstration = []
            for class_id in self.class_ids:
                img_metadata_classwise[class_id] = []
            for item in demonstration:
                img_name, img_class = item
                img_metadata_classwise[img_class] += [[img_name,img_class]]
            
            for class_id in self.class_ids:
                idx = np.random.choice(list(range(len(img_metadata_classwise[class_id]))), shots//len(self.class_ids), replace=False)

                new_demonstration += [img_metadata_classwise[class_id][id] for id in idx]
            while len(new_demonstration) != shots:
                now_new = np.random.choice(list(range(len(demonstration))), 1, replace=False)
                if demonstration[now_new[0]][0] not in [item[0] for item in new_demonstration]:
                    new_demonstration += [demonstration[now_new[0]]]
            demonstration = new_demonstration

        else:
            demonstration_index = np.random.choice(range(len(demonstration)), shots, replace=False)
            print(demonstration_index)
            demonstration = [demonstration[index] for index in demonstration_index]

        print('Shorts (trn) images are : %d' % len(demonstration))
        return img_metadata,demonstration

    # def build_img_metadata(self):
    #
    #     def read_metadata(split, fold_id):
    #         cwd = os.path.dirname(os.path.abspath(__file__))
    #         fold_n_metadata = os.path.join(cwd, 'splits/pascal/%s/fold%d.txt' % (split, fold_id))
    #         with open(fold_n_metadata, 'r') as f:
    #             fold_n_metadata = f.read().split('\n')[:-1]
    #         fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
    #         return fold_n_metadata
    #
    #     img_metadata = []
    #     img_metadata = read_metadata('val', self.fold)
    #
    #     print('Total (val) images are : %d' % len(img_metadata))
    #
    #     return img_metadata

    # def build_img_metadata_classwise(self):
    #     img_metadata_classwise = {}
    #     for class_id in range(self.nclass):
    #         img_metadata_classwise[class_id] = []
    #
    #     for img_name, img_class in self.img_metadata:
    #         img_metadata_classwise[img_class] += [img_name]
    #     return img_metadata_classwise

class DatasetPASCALforFinetune(Dataset):
    def __init__(self, datapath, fold, image_transform, mask_transform, num_supports=1, padding: bool = 1,
                 use_original_imgsize: bool = False, random: bool = False):
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.padding = padding
        self.random = random
        self.use_original_imgsize = use_original_imgsize

        self.img_path = os.path.join(datapath, 'VOCdevkit/VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOCdevkit/VOC2012/SegmentationClassAug/')
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.num_supports = num_supports

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample_query, class_sample_support = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name,
                                                                                             support_names)

        query_mask = self.extract_ignore_idx(query_cmask, class_sample_query)[0]

        if self.image_transform:
            query_img = self.image_transform(query_img)

        if self.mask_transform:
            query_mask = self.mask_transform(query_mask)

        if self.image_transform:
            for i in range(len(support_imgs)):
                support_imgs[i] = self.image_transform(support_imgs[i])
        support_masks = [self.extract_ignore_idx(m, class_sample_support)[0] for m in support_cmasks]
        if self.mask_transform:
            for i in range(len(support_masks)):
                support_masks[i] = self.mask_transform(support_masks[i])

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'org_query_imsize': org_qry_imsize,
                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'class_id': torch.tensor(class_sample_query),
                 }

        return batch

    def extract_ignore_idx(self, mask, class_id):
        mask = np.array(mask)
        boundary = np.floor(mask / 255.)
        mask[mask != class_id + 1] = 0.
        mask[mask == class_id + 1] = 255.
        return Image.fromarray(mask), boundary

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = []
        support_masks = []
        for support_name in support_names:
            support_imgs.append(self.read_img(support_name))
            support_masks.append(self.read_mask(support_name))

        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = Image.open(os.path.join(self.ann_path, img_name) + '.png')
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx):
        """Returns the index of the query, support and class."""
        query_name, class_sample = self.img_metadata[idx]
        if not self.random:
            support_class = class_sample
        else:
            support_class = \
            np.random.choice([k for k in self.img_metadata_classwise.keys() if self.img_metadata_classwise[k]], 1,
                             replace=False)[0]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[support_class], 1, replace=False)[0]
            if query_name != support_name and support_name not in support_names:
                support_names.append(support_name)
                if len(support_names) >= self.num_supports:
                    break
        return query_name, support_names, class_sample, support_class

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            cwd = os.path.dirname(os.path.abspath(__file__))
            fold_n_metadata = os.path.join(cwd, 'splits/pascal/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        img_metadata = read_metadata('val', self.fold)

        print('Total (val) images are : %d' % len(img_metadata))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise