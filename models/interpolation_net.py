import torch
import torch.nn as nn
import numpy as np
import itertools
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from .interpolation.refine import *
from .interpolation.warplayer import warp
from .interpolation.IFNet import *
from .interpolation.IFNet_m import *
import torch.optim as optim
import os
    
class VideoInterpolationNet:
    def __init__(self, args, local_rank=-1, arbitrary=False, finetune=False):
        if arbitrary == True:
            self.flownet = IFNet_m()
        else:
            self.flownet = IFNet()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

        self.finetune = finetune
        if self.finetune:
            self.optimG = AdamW(self.flownet.parameters(), lr=args.flow_learning_rate, weight_decay=1e-3)
            self.lr_schedulerG = optim.lr_schedulerG.ReduceLROnPlateau(self.optimG, "min", patience=10, factor=0.2)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self, device):
        self.flownet.to(device)

    def load_model(self, path, rank=0, map_location="cpu"):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
        if 'flownet.pkl' in str(path):
            checkpoint = convert(torch.load(path, map_location=map_location))
        else:
            checkpoint = torch.load(path, map_location=map_location)
        self.flownet.load_state_dict(checkpoint)
        
    def save_model(self, path, is_best):
        if is_best:
            checkpoint = os.path.join(path, "RIFEflow_best.pth.tar")
        else:
            checkpoint = os.path.join(path, "RIFEflow.pth.tar")
        torch.save(self.flownet.state_dict(), checkpoint)

    def freeze_model(self):
        for param in self.flownet.parameters():
            param.requires_grad = False 

    '''
    def inference(self, img0, img1, scale_list=[4, 2, 1], TTA=False, timestep=0.5):
        imgs = torch.cat((img0, img1), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        if TTA == False:
            return merged[2]
        else:
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2
    '''
    
    def inference(self, img0, img1, scale_list=[4, 2, 1], timestep=0.5):
        imgs = torch.cat((img0, img1), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        return merged[2]



    def coding_inference(self, img0, img1, scale_list=[4, 2, 1], TTA=False, timestep=0.5):
        h, w = img0.size(2), img0.size(3)
        p = 64  # maximum 6 strides of 2
        new_h = (h + p - 1) // p * p
        new_w = (w + p - 1) // p * p
        padding_left = (new_w - w) // 2
        padding_right = new_w - w - padding_left
        padding_top = (new_h - h) // 2
        padding_bottom = new_h - h - padding_top
        img0_padded = F.pad(
            img0,
            (padding_left, padding_right, padding_top, padding_bottom),
            mode="constant",
            value=0,
        )
        img1_padded = F.pad(
            img1,
            (padding_left, padding_right, padding_top, padding_bottom),
            mode="constant",
            value=0,
        )        
        imgs = torch.cat((img0_padded, img1_padded), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        if TTA == False:
            final_img = F.pad(merged[2], (-padding_left, -padding_right, -padding_top, -padding_bottom))
            return final_img
        else:
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            final_img = F.pad((merged[2] + merged2[2].flip(2).flip(3)) / 2, (-padding_left, -padding_right, -padding_top, -padding_bottom))
            return final_img

