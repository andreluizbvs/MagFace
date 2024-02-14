#!/usr/bin/env python
import sys
sys.path.append("..")

import ast
import cv2
import numpy as np
import os
import pandas as pd
import torch.utils.data as data
from tqdm import tqdm

from skimage import transform as trans
from utils import cv2_trans as transforms

side_size = 112
op_path = '/home/ray/biometrics-storage/inputs/train/labels/op_data_total.csv'
arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014],  # eyes
     [56.0252, 71.7366],                      # nose
     [41.5493, 92.3655], [70.7299, 92.2041]], # mouth
    dtype=np.float32)
arcface_src = np.expand_dims(arcface_src, axis=0)


def apply_bbox(img, bbox):
    x, y, x2, y2 = map(int, bbox)
    return img[y:y2, x:x2]


def convert_coordinates(bbox, coordinates):
    x, y, _, _ = map(int, bbox)
    return [coord - np.array([x, y]) for coord in coordinates]


def trans_img_and_coords(img, coords):
    x = img.shape[0]
    y = img.shape[1]
    x_scale = side_size/x
    y_scale = side_size/y

    coords = [[coord[0] * x_scale, coord[1] * y_scale] for coord in coords]
    img = cv2.resize(img, (side_size, side_size))

    return img, coords

def estimate_norm(lmk, image_size=side_size, mode='arcface'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        if image_size == side_size:
            src = arcface_src
        else:
            src = float(image_size) / side_size * arcface_src
    else:
        src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(img, landmark, image_size=side_size, mode='arcface'):
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped
    

class MagTrainDataset(data.Dataset):
    def __init__(self, ann_file, transform=None):
        self.ann_file = ann_file
        self.transform = transform
        self.init()

    def init(self):
        self.weight = {}
        self.im_names = []
        self.targets = []
        self.coordinates = []

        csv_data = pd.read_csv(op_path)
        csv_data.dropna(subset=['face_bbox', 'eyes', 'nose', 'mouth'], inplace=True)
        
        count = 0
        num_lines = 19909712
        with open(self.ann_file) as f:
            for line in tqdm(f.readlines(), total=num_lines):
                count += 1
                if count > 1000:
                    break
                try:
                    data = line.strip().split(' ')
                    if len(data) > 2:
                        continue
                    im_name = data[0].split('/')[-1]
                    filter_mask = np.in1d(csv_data['image_name'], im_name)
                    row = csv_data.loc[filter_mask]
                    if len(row) == 0:
                        continue
                    row = row.iloc[0] # Get first row (ideally only has one)
                    self.im_names.append(data[0])
                    self.targets.append(int(data[1]))
                    bbox = ast.literal_eval(row['face_bbox'])
                    eye_left = ast.literal_eval(row['eyes'])[0]
                    eye_right = ast.literal_eval(row['eyes'])[1]
                    nose = ast.literal_eval(row['nose'])
                    mouth_left = ast.literal_eval(row['mouth'])[0]
                    mouth_right = ast.literal_eval(row['mouth'])[1]
                    self.coordinates.append(
                        (bbox,
                        convert_coordinates(
                            bbox, 
                            [eye_left, 
                             eye_right, 
                             nose, 
                             mouth_left, 
                             mouth_right]
                        ))
                    )
                except Exception as e:
                    print(e)
                    print(data)
        
        del csv_data

    def __getitem__(self, index):          
        im_name = self.im_names[index]
        target = self.targets[index]

        img = cv2.imread(im_name)

        bbox = self.coordinates[index][0]
        img = apply_bbox(img, bbox)

        coords = self.coordinates[index][1]
        img, coords = trans_img_and_coords(img, coords)
        img = norm_crop(img, np.array(self.coordinates[index][1]))

        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.im_names)


def train_loader(args):
    train_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train_dataset = MagTrainDataset(
        args.train_list,
        transform=train_trans
    )
    train_sampler = None
    train_loader = data.DataLoader(
        train_dataset,
        shuffle=(train_sampler is None),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=False)
    

    return train_loader
