import torch
from torch.utils.data import  Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import pickle
import random
from transformers import AutoProcessor
from torchvision import transforms  
from datasets import load_dataset
processor = AutoProcessor.from_pretrained("/home/wpy/wpy_workspace/work/model/clip-p32")

class fer2013_dataset(Dataset):
    def __init__(self,file_path,state):
        self.state=state
        self.file=pd.read_csv(file_path)
        self.img_path=self.file['image_path'].to_list()
        self.label=self.file['label'].to_list() 
        #数据增强
        self.train_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
        ])
    def __len__(self):
        return len(self.file)
    def __getitem__(self, index):
        cap = cv2.VideoCapture(self.image_path[index]) #读取每一帧
        frame_count = 0# 帧计数器
        img_list=[]
        while True:     # 读取视频直到视频结束
            ret, frame = cap.read() # 读取下一帧
            #output_folder = '/home/fu_xiao/test/clip_msa/data/meld/img'
            # 如果正确读取帧，ret为True，否则视频结束
            if not ret:
                break 
            if frame_count % 20 == 0 :    # 每隔5帧保存一张图片，可以根据需要调整
                # 构造图片文件名
                #filename = os.path.join(output_folder, f'{lag}image{i}_{frame_count/50}.png')
                # 保存帧为图片
                #cv2.imwrite(filename, frame)
                pil_image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if self.state=='train': 
                    pil_image=self.train_transform(pil_image) #数据增强
                img_pt=processor(images=pil_image,return_tensors="pt")
                img_list.append(img_pt)
            frame_count+=1
        label = self.label[index]
        #if i<730:
        return img_list,label