

import re
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import random


def gen_split(root_dir, stackSize):
    Dataset = []
    Labels = []
    NumFrames = []
    data_mmaps=[]
    root_dir = os.path.join(root_dir, 'frames')
    for dir_user in sorted(os.listdir(root_dir)):
        class_id = 0
        dir = os.path.join(root_dir, dir_user)
        for target in sorted(os.listdir(dir)):
            dir1 = os.path.join(dir, target)
            insts = sorted(os.listdir(dir1))
            if insts != []:
                for inst in insts:
                    inst_dir = os.path.join(dir1, inst, "rgb")
                    
                    inst_dir_mmaps = os.path.join(dir1, inst, "mmaps")
                    numFrames_mmaps = len(glob.glob1(inst_dir_mmaps, '*.png'))
                    numFrames = len(glob.glob1(inst_dir, '*.png'))
                    if numFrames >= stackSize:
                        Dataset.append(inst_dir)
                        Labels.append(class_id)
                        NumFrames.append(numFrames)
                        data_mmaps.append(inst_dir_mmaps)
            class_id += 1
    return Dataset, Labels, NumFrames, data_mmaps

class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, spatial_transform_map=None,seqLen=20,
                 train=True, mulSeg=False, numSeg=1, fmt='.png'):
        self.images, self.labels, self.numFrames,self.mmaps = gen_split(root_dir, 5)
        self.spatial_transform = spatial_transform
        self.spatial_transform_map = spatial_transform_map
        self.train = train
        self.mulSeg = mulSeg
        self.numSeg = numSeg
        self.seqLen = seqLen
        self.fmt = fmt

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        label = self.labels[idx]
        mmaps_name= self.mmaps[idx]
        flag=1
        numFrame = self.numFrames[idx]
        inpSeq = []
        mmapsSeq=[]
        m=[]
        self.spatial_transform.randomize_parameters()
        if self.train==True:#applies the same transforms also at the target of the self-supervised task (eliminating random elements)
         self.spatial_transform_map.transforms[2].scale=self.spatial_transform.transforms[2].scale
         self.spatial_transform_map.transforms[2].crop_position=self.spatial_transform.transforms[2].crop_position
         self.spatial_transform_map.transforms[1].p=self.spatial_transform.transforms[1].p

        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            fl_name = vid_name + '/' + 'rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            fl_name_mmaps = vid_name + '/' + 'rgb' + str(int(np.floor(i+1))).zfill(4) + self.fmt
            if (not os.path.isfile(fl_name_mmaps)):#in case is not present the a element of the self-supervised task we put the flag=0
              fl_name_mmaps = mmaps_name + '/' + 'map' + str(int(np.floor(i))).zfill(4) + self.fmt
              flag=0
            img = Image.open(fl_name)
            img_maps = Image.open(fl_name_mmaps)

            inpSeq.append(self.spatial_transform(img.convert('RGB')))
            m.append(self.spatial_transform(img_maps.convert('RGB')))
            mmapsSeq.append(self.spatial_transform_map(img_maps ))
        inpSeq = torch.stack(inpSeq, 0)
        mmapsSeq=torch.stack(mmapsSeq, 0)
        m=torch.stack(m, 0)
        return inpSeq, label,mmapsSeq,m,flag

