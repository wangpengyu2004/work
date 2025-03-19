from torch.utils.data import Dataset,DataLoader
import numpy as np
import pandas as pd
from transformers import AutoProcessor
from PIL import Image  
from torchvision import transforms  
processor=AutoProcessor.from_pretrained("/home/wpy/CLIP4emo/model_all/pretrained_model/clip_p32")
sentimentlabels_index={'Angry':0,'Disgust':1,'Fear':2,'Happy':3,'Neutral':4,'Sad':5,'Surprise':6}

class sfew_dataloder(Dataset):
    def __init__(self,path,state):
        super(sfew_dataloder,self).__init__()
        self.file=pd.read_csv(path)
        self.emo=self.file['label'].to_list()
        self.image_path=self.file['image_path'].to_list()
        self.state=state
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
        em=sentimentlabels_index[str(self.emo[index])]
        image=Image.open(self.image_path[index])
        image_feature=processor(images=image,return_tensors="pt")
        return em,image_feature

           