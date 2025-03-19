from model_all.model.model import tools
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from utils import show_class
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import copy
import numpy as np
import os
from torch.amp import autocast, grad_scaler

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding.to()
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        self.positional_embedding=self.positional_embedding.to(prompts.device)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x=x.type(self.dtype)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
class get_prompt():
    def __init__(self,clip_model,prompt_learner,device):
        super().__init__()
        self.device=device
        self.prompt_learner=prompt_learner
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype

    def get_prompt(self,label):
        prompts = self.prompt_learner().to(self.device)
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        return text_features[label]

def get_pred_f(img,landmarks,clip_model,sentimentlabels,tokenizer,descriptions,prompts):
    text_features_v = []
    img=img['pixel_values'][0]
    for i,em in zip(range(len(sentimentlabels)),sentimentlabels):
        emo_name=[i for i in sentimentlabels if i!=em]
        emo_str=','.join(emo_name)
        #txt=[f"this photo that evokes {em}"]#,but not evokes {emo_str}"]
        #txt=[f"this photo that evokes {em},but not {emo_str}"]
        """
        txt=[descriptions[i]]
        inputs = tokenizer(txt, padding=True, return_tensors="pt")
        text_feat_temp = (clip_model.get_text_features(**inputs)[0]).detach().numpy()
        """
        text_feat_temp=prompts.get_prompt(i).cpu().detach().numpy()
        text_features_v.append([text_feat_temp])
    img_feat_temp=(clip_model.get_image_features(img)[0]).detach().numpy()
    #img_input=np.squeeze(img_feat_temp)
    img_input=img_feat_temp
    text_input=np.asarray(text_features_v)
    img_input = np.tile(img_input, (len(text_input),1))
    landmarks=np.tile(np.asarray(landmarks), (len(text_input),1))
    img_input=torch.tensor([img_input],dtype=torch.float32).transpose(0,1)
    text_input=torch.tensor(text_input,dtype=torch.float32)
    landmarks=torch.tensor(landmarks,dtype=torch.float32).transpose(0,1)
    return text_input,img_input,landmarks

def get_input(emo,image,clip_model,tokenizer,sentimentlabels,sentimentlabels_index,descriptions,prompts):
    text_features=[]
    image_features=[]
    text=[]
    image=image['pixel_values']
    for em,img in zip(emo,image):
        emo_name=[i for i in sentimentlabels if i!=sentimentlabels_index[int(em)]]
        #emo_name=[i for i in sentimentlabels if i!=em]
        emo_str=','.join(emo_name)
        #text=[f"this photo that evokes {sentimentlabels_index[int(em)]}"]
        #text=[f"this photo that evokes {sentimentlabels_index[int(em)]}, but not {emo_str}"]
        """
        text=[descriptions[int(em)]]
        txt_inputs=tokenizer(text,padding=True, return_tensors="pt")
        text_feat_temp = (clip_model.get_text_features(**txt_inputs)[0]).detach().numpy()
        """
        text_feat_temp=prompts.get_prompt(int(em)).cpu().detach().numpy()
        text_features.append([text_feat_temp])
        img_feat_temp=(clip_model.get_image_features(img)).detach().numpy()
        image_features.append(img_feat_temp)
    text_features = np.asarray(text_features)
    #image_features = np.squeeze(image_features)
    text_input = torch.tensor(text_features, dtype=torch.float32)
    image_input = torch.tensor(image_features, dtype=torch.float32)
    return text_input,image_input

def training(model,
             learnable_prompt,
             data_loader,
             optimizer,
             scheduler_s:str,
             epoch,
             clip_model,
             clip4prompt,
             tokenizer,
             sentimentlabels,
             sentimentlabels_index,
             descriptions,
             batch_size:int,
             model_save_path:str,
             log_path:str,
             details_path,
             tensorboard_path,
             device_num):
    writer=SummaryWriter(tensorboard_path)
    prompts=get_prompt(clip4prompt,learnable_prompt,device_num[0])
    
    if scheduler_s=='steplr':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.01)
    else:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0.00001)
    # 初始化 GradScaler
    scaler = grad_scaler.GradScaler()
    best_model=copy.deepcopy(model.module.state_dict())
    best_acc=0.0
    best_epoch=0
    for i in range(epoch):
        label_rec=np.zeros((2,len(sentimentlabels)))  #记录标签
        for state in ['train','test']:
            if(state=='train'):
                model.train()
            else:
                model.eval()
            running_acc=0.0
            running_loss=0.0
            text_features=[]
            image_features=[]
            labelss=[]
            loss_writer=0.0
            totol=0
            batch_i=0      
            for (emo,image,landmarks) in tqdm(data_loader[state], desc=f'Epoch {i+1}/{epoch}', unit='batch'):#enumerate(data_loader[state]):
                if state=='train':
                    if len(emo) <batch_size:
                        continue
                    totol+=1
                    text_input,image_input=get_input(emo,image,clip_model,tokenizer,sentimentlabels,sentimentlabels_index,descriptions,prompts)
                    text_input=text_input.cuda(device=device_num[0],non_blocking=True)
                    image_input=image_input.cuda(device=device_num[0],non_blocking=True)
                    landmarks=landmarks.cuda(device=device_num[0],non_blocking=True).type(torch.float32)
                    with autocast(device_type='cuda'):
                        text_outputs,image_outputs=model(text_input,image_input,landmarks)
                        loss=model.module.compute_loss(text_outputs,image_outputs,emo)
                    """
                    text_features.append(text_outputs.detach().cpu().numpy())
                    image_features.append(image_outputs.detach().cpu().numpy())
                    labelss.append(emo)
                    show_class(text_features,labelss,f"text{i}_train")
                    show_class(image_features,labelss,f"image{i}_train")
                    """
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update() 
                    running_loss+=loss.item()
                    if totol%100==0:
                        writer.add_scalar("running_loss",(running_loss-loss_writer)/100,i*len(data_loader[state])+totol)
                        loss_writer=running_loss
                    print(f"epoch {i} batch:{batch_i} loss: {loss}")
                    batch_i+=1
                else:
                    text_input,image_input,landmarks=get_pred_f(image,landmarks,clip_model,sentimentlabels,tokenizer,descriptions,prompts)
                    with torch.no_grad():
                        text_input=text_input.cuda(device=device_num[0],non_blocking=True)
                        image_input=image_input.cuda(device=device_num[0],non_blocking=True)
                        landmarks=landmarks.cuda(device=device_num[0],non_blocking=True)
                        text_outputs,image_outputs=model(text_input,image_input,landmarks)
                    """
                    text_features.append(text_outputs.detach().cpu().numpy())
                    image_features.append(image_outputs.detach().cpu().numpy())
                    labelss.append(emo)
                    """
                    #preds=tools.similarity(text_outputs,image_outputs)
                    #print(tools.similarity_probs(text_outputs,image_outputs))
                    probs,labels=tools.similarity_probs_test(text_outputs,image_outputs)
                    probs=np.asarray((probs.cpu()).detach().numpy())
                    labels=np.asarray((labels.cpu()).detach().numpy())
                    #print(tools.similarity_probs(text_outputs,image_outputs))
                    label_rec[0][int(emo)]+=1
                    #print(f"pre:{preds},real:{emo}")
                    if labels[0]==emo :
                       label_rec[1][int(labels[0])]+=1
                       running_acc+=1
                    with open(details_path,'a') as f:
                            f.write(str(sentimentlabels_index[labels[0]])+' '+str(sentimentlabels_index[labels[1]])+' '+str(sentimentlabels_index[labels[2]])+' '+str(sentimentlabels_index[labels[3]])+' '+str(sentimentlabels_index[labels[4]])+' '+str(sentimentlabels_index[labels[5]])+' '+str(sentimentlabels_index[labels[6]]))#+' '+str(sentimentlabels_index[int(emo)]+"\n"))
                            f.write(str(probs)+'\n')
                            f.write('--------------------------------------\n')
                    totol+=1
                    print(f"epoch{i} batch:{batch_i},acc:{running_acc/totol}")
                    batch_i+=1
            if state=='train':
                scheduler.step()
                epoch_loss = running_loss / totol
                #show_class(text_features,labelss,f"text{i}_train")
                #show_class(image_features,labelss,f"image{i}_train")
                print(f"state:{state} epoch:{i} loss:{epoch_loss}")
                with open(log_path, 'a') as f:
                    f.write(f"\nmodel_fer_nodata\n")
                    f.write("=================================\n")
                    f.write("train")
                    f.write('\nepoch : ' + str(i + 1) + '\n')
                    f.write('epoch_loss :' + str(epoch_loss) + '\n')
                    f.write("=================================\n")
            else:
                #show_class(text_features,labelss,f"text{i}_test")
                #show_class(image_features,labelss,f"image{i}_test")
                epoch_acc=running_acc/totol
                if epoch_acc>best_acc:
                    best_acc=epoch_acc
                    best_epoch=i
                    best_model=copy.deepcopy(model.module.state_dict())
                writer.add_scalar("running_acc",epoch_acc,i)
                with open(log_path, 'a') as f:
                    f.write("=================================\n")
                    f.write("test")
                    f.write('\nepoch : ' + str(i + 1) + '\n')
                    f.write('epoch_acc :' + str(epoch_acc) + '\n')
                    f.write('angry'+str(label_rec[1][0]/label_rec[0][0])+'\n')
                    f.write('disgusted'+str(label_rec[1][1]/label_rec[0][1])+'\n')
                    f.write('fearful'+str(label_rec[1][2]/label_rec[0][2])+'\n')
                    f.write('happy'+str(label_rec[1][3]/label_rec[0][3])+'\n')
                    f.write('sad'+str(label_rec[1][4]/label_rec[0][4])+'\n')
                    f.write('surprise'+str(label_rec[1][5]/label_rec[0][5])+'\n')
                    f.write('neutral'+str(label_rec[1][6]/label_rec[0][6])+'\n')
                    f.write('best_epoch :' + str(best_epoch+1) + '\n')
                    f.write('best_acc :' + str(best_acc) + '\n')
                    f.write("=================================\n")
                print(f"state:{state} epoch:{i} loss:{epoch_acc}")
    writer.close()
    print(f"best_model epoch:{best_epoch}acc:{best_acc}")
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    torch.save(best_model,model_save_path)




