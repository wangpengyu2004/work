from model_all.model.model import clipe_msa
from trainer.train import training
from utils import set_random_seed
from dataset import return_dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import yaml
import argparse
from datasets import load_dataset
from transformers import  AutoTokenizer, CLIPModel
from model_all.clip.clip import load
import numpy as np
import os


def main(args):
    #读取配置文件
    config_path="./config/"+args.config
    with open(config_path,'r') as file:
        cfg=yaml.safe_load(file)
    sentimentlabels=cfg['experiment']['sentimentlabels']
    sentimentlabels_index=cfg['experiment']['sentimentlabels_index']
    descriptions=cfg['experiment']['descriptions']
    num_class=len(sentimentlabels)
    model_save_path=cfg['experiment']['model_save_path']
    log_path=cfg['experiment']['log_path']
    details_path=cfg['experiment']['details_path']
    tensorboard_path=cfg['experiment']['tensorboard_path']
    #随机种子
    set_random_seed(seed=cfg['training'][args.seed])
    with open(log_path,'a') as f:
        f.write("\n-----------------------------------------\n")
        f.write(f"\n      random_seed =  {cfg['training'][args.seed]}\n")
    
    #数据加载
    trian_dataset,test_dataset=return_dataset(args.data)
    train_loader=DataLoader(trian_dataset,shuffle=True,batch_size=cfg['training']['batch_size'],num_workers=args.num_workers,pin_memory=True,persistent_workers=True)
    test_loader=DataLoader(test_dataset,shuffle=True,batch_size=1,num_workers=1,pin_memory=True,persistent_workers=True)
    data_loader={'train':train_loader,'test':test_loader}

    #clip
    clip=CLIPModel.from_pretrained(args.clip_path)
    clip4prompt,_=load("ViT-B/32",device=args.device_num[0])
    tokenizer = AutoTokenizer.from_pretrained(args.clip_path)

    #model
    model=clipe_msa(cfg,clip4prompt,args.device_num)
    if args.load_model:
        load_path="./model_save/"+args.load_path
        model.load_state_dict(torch.load(load_path))
    optimizer=optim.Adam(model.parameters(),cfg['training']['lr'])
    
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, Shape: {param.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    #prompt
    learnable_prompt=model.return_learner_prompt()
    #GPU
    if args.device=='cuda':
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model, device_ids=args.device_num).cuda()
        else:
            print("your cuda is not available")
    
    #模型训练
    training(model,learnable_prompt,data_loader,optimizer,cfg['training']['scheduler_s'],cfg['training']['epoch'],
             clip,clip4prompt,tokenizer,sentimentlabels,sentimentlabels_index,descriptions,cfg['training']['batch_size'],
             model_save_path,log_path,details_path,tensorboard_path,args.device_num)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="emoclip trainning")
    parser.add_argument('--data',type=str,required=True,help='your data :aff40k fer2013 fer_plus raf sfew')
    parser.add_argument('--config',type=str,required=True,help='your config')
    parser.add_argument('--seed',type=str,default='seed1',required=True,help='your config')
    parser.add_argument('--load_model',type=bool,default=False,help='load pretrained model')
    parser.add_argument('--load_path',type=str,default= None ,help='pretrained model path')
    parser.add_argument('--clip_path',type=str,default='/home/wpy/CLIP4emo/model_all/pretrained_model/clip_p32',help='clips model path')
    parser.add_argument('--device',type=str,default='cuda',help='cuda/cpu')
    parser.add_argument('--device_num',type=tuple,default=[0],help='the numbers of cudas')
    parser.add_argument('--num_workers',type=int,default=6,help=' the number of workers')
    args = parser.parse_args()
    main(args)

"""
python main.py --data raf --config raf_cfg.yaml  --seed seed5 --load_model True --load_path raf/raf_save6.pth
python main.py --data sfew --config sfew_cfg.yaml  --seed seed1 --load_model True --load_path raf/save5.pth
python main.py --data fer2013 --config fer_cfg.yaml --seed seed1
"""