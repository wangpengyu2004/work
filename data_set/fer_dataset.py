import torch
from torch.utils.data import  Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
import pickle
import random
from transformers import AutoProcessor
from torchvision import transforms  
from datasets import load_dataset
processor = AutoProcessor.from_pretrained("/home/wpy/CLIP4emo/model_all/pretrained_model/clip_p32")

label_map = {
    0:'angry',
    1:'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

class fer2013_dataset(Dataset):
    def __init__(self,file_path,landmark_path,state):
        self.landmarks = np.load(landmark_path)
        self.state=state
        self.file=pd.read_csv(file_path)
        self.emo=self.file['label'].to_list()
        self.image=self.file['image'].to_list() 
        #数据增强
        mu, st = 0, 255
        self.test_transform = transforms.Compose([transforms.Grayscale(),
        #transforms.TenCrop(40),
        #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        #transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors]))
        ])
       
        self.train_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            #transforms.FiveCrop(40),
            #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            #transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
            #transforms.ToTensor(),
            #transforms.Lambda(lambda tensors: torch.stack([transforms.RandomErasing()(t) for t in tensors])),
            #transforms.ToPILImage()
        ])
    def __len__(self):
        return len(self.file)
    def __getitem__(self, index):
        em=self.emo[index]
        img=Image.open('/home/wpy/CLIP4emo/data/fer2013'+'/'+self.state+'/'+label_map[em]+'/'+self.image[index])
        '''
        if self.state=='train':
            img=self.train_transform(img)   #数据增强
        else:
            img=self.test_transform(img)
        '''
        img_pt=processor(images=img,return_tensors="pt")
        landmark = self.landmarks[index].reshape((1,-1))
        return em,img_pt,landmark