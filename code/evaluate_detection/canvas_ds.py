import torch.utils.data as data
from evaluate_detection.voc_orig import VOCDetection as VOCDetectionOrig, make_transforms
import cv2
from evaluate.pascal_dataloader import create_grid_from_images_old as create_grid_from_images
from PIL import Image
from evaluate.mae_utils import *
from matplotlib import pyplot as plt
import torch
import numpy as np
import torchvision.transforms as T
from itertools import combinations,permutations
import math
import json

def box_to_img(mask, target, border_width=4):
    if mask is None:
        mask = np.zeros((112, 112, 3))
    h, w, _ = mask.shape
    for box in target['boxes']:
        x_min, y_min, x_max, y_max = list((box * (h - 1)).round().int().numpy())
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), (255, 255, 255), border_width)
    return Image.fromarray(mask.astype('uint8'))


def get_annotated_image(img, boxes, border_width=3, mode='draw', bgcolor='white', fg='image'):
    if mode == 'draw':
        image_copy = np.array(img.copy())
        for box in boxes:
            box = box.numpy().astype('int')
            cv2.rectangle(image_copy, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), border_width)
    elif mode == 'keep':
        image_copy = np.array(Image.new('RGB', (img.shape[1], img.shape[0]), color=bgcolor))

        for box in boxes:
            box = box.numpy().astype('int')
            if fg == 'image':
                image_copy[box[1]:box[3], box[0]:box[2]] = img[box[1]:box[3], box[0]:box[2]]
            elif fg == 'white':
                image_copy[box[1]:box[3], box[0]:box[2]] = 255




    return image_copy




# ids_shuffle, len_keep = generate_mask_for_evaluation_2rows()

class CanvasDataset(data.Dataset):

    def __init__(self, pascal_path='/', years=("2012",), random=False, 
                shots: int = 4, use_class: bool = False, random_num: int = 0, choose_num: int =0, trn: bool = False,
                dev_list = [],aug=False,sim=False,**kwargs):
        self.train_ds = VOCDetectionOrig(pascal_path, years, image_sets=['train'], transforms=None)
        self.val_ds = VOCDetectionOrig(pascal_path, years, image_sets=['val'], transforms=None)
        self.background_transforms = T.Compose([
            T.Resize((224, 224)),
            T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ])
        self.transforms = make_transforms('val')
        self.random = random

        assert not (random_num > 0 and choose_num > 0)
        self.use_class = use_class
        self.choose_num = choose_num
        self.random_num = random_num
        
        self.aug = aug
        self.sim = sim

        self.img_metadata = list(range(len(self.val_ds)))
        
        
        self.img_metadata = [494]

        demonstration = np.random.choice(range(len(self.train_ds)), shots, replace=False)

        self.demonstration = demonstration

        print("demonstration, ",demonstration)
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
                    new_img_metadata += list(range(len(self.val_ds)))
                    new_demonstration += [list(now_set)] * len(self.val_ds)
            self.demonstration = new_demonstration
            self.img_metadata = new_img_metadata
            assert len(self.img_metadata) == len(self.demonstration)
        self.trn = trn

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

        if self.sim:
            with open("../VOC2012/features_vit-laion2b_val/det-similarity.json",'r', encoding='UTF-8') as f:
                sim_json = json.load(f)
            new_demonstration = []
            for name in self.img_metadata:
                name = self.val_ds.images[name]
                new_demonstration += [[item for item in sim_json[name] if item in [self.train_ds.images[now] for now in self.demonstration]][:1]]
            now_dev = []
            for names in new_demonstration:
                now = []
                for name in names:
                    for dev in self.demonstration:
                        if self.train_ds.images[dev] == name:
                            now.append(dev)
                now_dev.append(now)
            self.demonstration = now_dev
            self.trn = True
            self.val_len = 0

        print("train: {} ,val: {} ".format(len(self.train_ds),len(self.val_ds)))
        print("img_metadata: {} ,demonstration: {} ".format(len(self.img_metadata),len(self.demonstration)))
        
        


    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        if self.random_num > 0:
            query_image, query_target = self.val_ds[self.img_metadata[idx]]
            query_name = 'val_' + str(self.img_metadata[idx])
            # should we run on all classes?
            label = np.random.choice(query_target['labels']).item()

            # how many supports should we use?
            indices = np.arange(len(self.train_ds))
            np.random.shuffle(indices)

            if self.use_class:
                for idx in indices:
                    support_image, support_target = self.train_ds[idx]
                    if torch.any(support_target['labels'] == label).item() or self.random:
                        break
                support_name = ["train_"+str(idx)]
            else:
                support_image, support_target = self.train_ds[indices[:self.random_num]]
        else:
            if self.trn:
                if idx >= self.val_len:
                    query_image, query_target = self.val_ds[self.img_metadata[idx]]
                    query_name = 'val_' + str(self.img_metadata[idx])
                    # should we run on all classes?
                    label = np.random.choice(query_target['labels']).item()
                    if isinstance(self.demonstration[idx],list):
                        support_image, support_target = [],[]
                        support_name = []
                        for index in self.demonstration[idx]:
                            now_image, now_target = self.train_ds[index]
                            support_image.append(now_image)
                            support_target.append(now_target)
                            support_name.append('train_'+str(index))

                    else:
                        support_image, support_target = self.train_ds[self.demonstration[idx]]
                        support_name = ['train_'+str(self.demonstration[idx])]

                else:
                    query_image, query_target = self.train_ds[self.img_metadata[idx]]
                    query_name = 'train_' + str(self.img_metadata[idx])
                    # should we run on all classes?
                    label = np.random.choice(query_target['labels']).item()
                    if isinstance(self.demonstration[idx],list):
                        support_image, support_target = [],[]
                        support_name = []
                        for index in self.demonstration[idx]:
                            now_image, now_target = self.train_ds[index]
                            support_image.append(now_image)
                            support_target.append(now_target)
                            support_name.append('train_'+str(index))

                    else:
                        support_image, support_target = self.train_ds[self.demonstration[idx]]
                        support_name = ['train_'+str(self.demonstration[idx])]


        boxes = query_target['boxes'][torch.where(query_target['labels'] == label)[0]]
        query_image_copy = get_annotated_image(np.array(query_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
        query_image_copy_pil = Image.fromarray(query_image_copy)

        query_image_ten = self.transforms(query_image, None)[0]
        query_target_ten = self.transforms(query_image_copy_pil, None)[0]
        
        background_image = Image.new('RGB', (224, 224), color='white')
        background_image = self.background_transforms(background_image)
        
        if isinstance(support_image,list):
            canvas = []
            for index in range(len(support_image)):
                if len(torch.where(support_target[index]['labels'] == label)[0]) != 0:
                    boxes = support_target[index]['boxes'][torch.where(support_target[index]['labels'] == label)[0]]
                else:
                    boxes = [support_target[index]['boxes'][0]]
                
                support_image_copy = get_annotated_image(np.array(support_image[index]), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
                support_image_copy_pil = Image.fromarray(support_image_copy)
                support_target_ten = self.transforms(support_image_copy_pil, None)[0]
                support_image_ten = self.transforms(support_image[index], None)[0]

                now = create_grid_from_images(background_image, support_image_ten, support_target_ten, query_image_ten,
                                                query_target_ten,self.aug)
                canvas.append(now)


        else:
            if len(torch.where(support_target['labels'] == label)[0]) != 0:
                boxes = support_target['boxes'][torch.where(support_target['labels'] == label)[0]]
            else:
                boxes = [support_target['boxes'][0]]
            support_image_copy = get_annotated_image(np.array(support_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
            support_image_copy_pil = Image.fromarray(support_image_copy)
            support_target_ten = self.transforms(support_image_copy_pil, None)[0]
            support_image_ten = self.transforms(support_image, None)[0]

            canvas = create_grid_from_images(background_image, support_image_ten, support_target_ten, query_image_ten,
                                            query_target_ten,self.aug)

        return {
            'grid': canvas,
            'support_name':support_name,
            'query_name':query_name
                }

