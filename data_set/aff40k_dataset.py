import torch
from torch.utils.data import  Dataset, DataLoader
import pandas as pd
from PIL import Image
import pickle
import random
from transformers import AutoProcessor
from datasets import load_dataset
processor = AutoProcessor.from_pretrained("/home/wpy/CLIP4emo/model_all/pretrained_model/clip_p32")

class aff40k_dataset(Dataset):
    def __init__(self,file_path,transform=None):
        self.file=pd.read_csv(file_path)
        self.emo=self.file['label'].to_list()
        self.image_path=self.file['img_path'].to_list()
    def __len__(self):
        return len(self.file)
    def __getitem__(self, index):
        em=self.emo[index]
        img=Image.open(self.image_path[index])
        img_pt=processor(images=img,return_tensors="pt")
        return em,img_pt
