import torch
from torch.utils.data import  Dataset, DataLoader
import pandas as pd
from PIL import Image
import pickle
import numpy as np
import random
from transformers import AutoProcessor
from torchvision import transforms
from datasets import load_dataset
#from ..model_all.model.landmark_model import dilb_landmark
processor = AutoProcessor.from_pretrained("/home/wpy/CLIP4emo/model_all/pretrained_model/clip_p32")

class raf_dataset(Dataset):
    def __init__(self,name,file_path,landmark_file,transform=None):
        self.file=pd.read_csv(file_path)
        self.emo=self.file['label'].to_list()
        self.image=self.file['image'].to_list()
        self.state=name
        self.name=name
        #self.landmark=dilb_landmark()
        self.landmarks = np.load(landmark_file)
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
        em=self.emo[index]-1
        image_path='/home/wpy/CLIP4emo/data/rafdb/'+str(self.name)+'/'+str(self.emo[index])+'/'+self.image[index]
        img=Image.open(image_path)
        if self.name=='train':
            img=img
            #img=self.train_transform(img)   #数据增强
        else:
            img=img
        landmark=self.landmarks[index].reshape((1,-1))
        #landmarks=self.landmark.get_landmark(image_path)
        img_pt=processor(images=img,return_tensors="pt")
        return em,img_pt,landmark