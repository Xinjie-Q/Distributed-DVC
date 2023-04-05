import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
import torchvision.transforms.functional as tf
from typing import Dict
from torch import Tensor
import numpy as np
import glob
import cv2
import json
from PIL import Image
import random


class Vimeo(Dataset):
    def __init__(self, data_root, is_training, crop_size):
        self.data_root = data_root # .\vimeo_septuplet
        self.image_root = os.path.join(self.data_root, 'sequences') # .\vimeo_septuplet\sequences
        self.training = is_training
        self.crop_size = crop_size
        if self.training:
            train_fn = os.path.join(self.data_root, 'sep_trainlist.txt') # 64612
            with open(train_fn, 'r') as f:
                self.trainlist = f.read().splitlines() # ['00001/0001','00001/0002',...]
        else:
            test_fn = os.path.join(self.data_root, 'sep_testlist.txt')
            with open(test_fn, 'r') as f:
                self.testlist = f.read().splitlines()
            #self.testlist = self.testlist[:len(self.testlist)//2]

        self.transforms = transforms.Compose([transforms.ToTensor()])


    def train_transform(self, frame_list):
        # Random cropping augmentation
        h_offset = random.choice(range(256 - self.crop_size[0] + 1))
        w_offset = random.choice(range(448 - self.crop_size[1]+ 1))

        choice = [random.randint(0, 1), random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)]
        flip_code = random.randint(-1,1)  # 0 : Top-bottom | 1: Right-left | -1: both

        frame_list_ = []
        for frame in frame_list:
            frame = frame[h_offset:h_offset + self.crop_size[0], w_offset: w_offset + self.crop_size[1], :]

            # Rotation augmentation
            if self.crop_size[0] == self.crop_size[1]:
                if choice[0]:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif choice[1]:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif choice[2]:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Flip augmentation
            if choice[3]:
                frame = cv2.flip(frame, flip_code)
            
            frame = tf.to_tensor(frame) #将numpy数组或PIL.Image读的图片转换成(C,H, W)的Tensor格式且/255归一化到[0,1.0]之间
            frame_list_.append(frame)
      
        return frame_list_
        # return map(TF.to_tensor, (frame1, frame2, frame3, flow, frame_fw, frame_bw))
        #return map(tf.to_tensor, (frame1, frame2, frame3))


    def test_transform(self, frame_list):
        frame_list_ = [self.transforms(frame) for frame in frame_list]
        return frame_list_

    def __getitem__(self, index):
        if self.training:
            imgpath = os.path.join(self.image_root, self.trainlist[index])
        else:
            imgpath = os.path.join(self.image_root, self.testlist[index])
        imgpaths = [imgpath + f'/im{i}.png' for i in range(1, 8)]
        images = [cv2.imread(pth) for pth in imgpaths]
        if self.training:
            images = self.train_transform(images)
            # Random Temporal Flip
            if random.random() >= 0.5:
                images = images[::-1]
        else:
            images = self.test_transform(images)

        return images

        '''
        if random.randint(0,1):
            First_fn  = os.path.join(self.sequence_list[index], 'im1.png')
            Third_fn  = os.path.join(self.sequence_list[index], 'im3.png')
        else:
            First_fn  = os.path.join(self.sequence_list[index], 'im3.png')
            Third_fn  = os.path.join(self.sequence_list[index], 'im1.png')
        
        Second_fn = os.path.join(self.sequence_list[index], 'im2.png')

        frame1 = imread(First_fn)
        frame2 = imread(Second_fn)
        frame3 = imread(Third_fn)

        frame1, frame2, frame3 = self.transform(frame1, frame2, frame3)

        Input = torch.cat((frame1, frame3), dim=0)

        return Input, frame2
        '''

    def __len__(self):
        if self.training:
            return len(self.trainlist)
        else:
            return len(self.testlist)

def TransformImgsWithSameCrop(images, cropsize):
    (i, j, h, w) = transforms.RandomCrop.get_params(images[0], (cropsize[0], cropsize[1]))
    images_ = []
    for image in images:
        image = image.crop((j, i, j+w, i+h)) # top left corner (j,i); bottom right corner (j+w, i+h)
        image = tf.to_tensor(image)
        images_.append(image)

    images = images_

    return images



def save_checkpoint(state, is_best=False, log_dir=None, filename="ckpt.pth.tar"):
    save_file = os.path.join(log_dir, filename)
    print("save model in:", save_file)
    torch.save(state, save_file)
    if is_best:
        torch.save(state, os.path.join(log_dir, filename.replace(".pth.tar", ".best.pth.tar")))


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def get_output_folder(parent_dir, env_name, output_current_folder=False):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    if not output_current_folder: 
        experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


