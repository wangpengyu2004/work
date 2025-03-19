import torch
from torch.utils.data import  Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
import pickle
import random
from transformers import AutoProcessor
from datasets import load_dataset
processor = AutoProcessor.from_pretrained("/home/wpy/CLIP4emo/model_all/pretrained_model/clip_p32")
sentimentlabels_index={'anger':0,'contempt':1,'disgust':2,'fear':3,'happiness':4,'neutral':5,'sadness':6,'surprise':7}
class fer2013_plus_dataset(Dataset):
    def __init__(self,file_path,transform=None):
        self.file=pd.read_csv(file_path)
        self.emo=self.file['label'].to_list()
        self.image=self.file['image_path'].to_list()
    def __len__(self):
        return len(self.file)
    def __getitem__(self, index):
        em=sentimentlabels_index[self.emo[index]]
        img=Image.open(self.image[index])
        img_pt=processor(images=img,return_tensors="pt")
        return em,img_pt