import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
import numpy as np
import torch.nn.functional as F
from numpy import linalg as LA
from model_all.layers.attention_layer import MultiHeadAttention
from model_all.clip import clip
from model_all.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from model_all.model.export import export_Block,MLP
from model_all.model.promptlearner import PromptLearner
from model_all.model.face_encode import FaceEncoder1,FaceEncoder2
from model_all.model.loss import AdaptiveCombinedLoss,ContrastiveLossWithHardNegatives,FocalLoss,CenterLoss

class tools():
    def similarity(img_features, list_text_features):
        # 归一化嵌入向量
        #img_features_normalized = F.normalize(img_features_normalized, p=2, dim=1)
        #list_text_features_normalized = F.normalize(list_text_features_normalized, p=2, dim=1)
        img_features_normalized = img_features / torch.norm(img_features, p=2, dim=-1, keepdim=True)
        list_text_features_normalized = list_text_features / torch.norm(list_text_features, p=2, dim=-1, keepdim=True)
        # 计算余弦相似度
        probs = torch.matmul(list_text_features_normalized,img_features_normalized.T )
        # 应用 softmax 函数
        probs = torch.softmax(probs, dim=1)
        # 获取概率最高的下标
        final_pred = torch.argmax(probs).item()  
        return final_pred
    
    def similarity_probs(img_features,list_text_features):
        img_features_normalized = img_features / torch.norm(img_features, p=2, dim=-1, keepdim=True)
        list_text_features_normalized = list_text_features / torch.norm(list_text_features, p=2, dim=-1, keepdim=True)
        # 计算余弦相似度
        probs = torch.matmul(list_text_features_normalized,img_features_normalized.T )
        # 应用 softmax 函数
        probs = torch.softmax(probs, dim=1)#.detach().numpy()
        return probs

    def similarity_probs_test(img_features,list_text_features):
        img_features_normalized = img_features / torch.norm(img_features, p=2, dim=-1, keepdim=True)
        list_text_features_normalized = list_text_features / torch.norm(list_text_features, p=2, dim=-1, keepdim=True)
        # 计算余弦相似度
        probs = torch.matmul(list_text_features_normalized,img_features_normalized.T )
        probs = torch.softmax(probs, dim=1)#.detach().numpy()
        top_probs, top_indices = torch.topk(probs, k=7, dim=1)
        return top_probs[0],top_indices[0]
    
class muti_fusion_block(nn.Module):
    def __init__(self,text_inputsize,text_outputsize,image_inputsize,image_outputsize,num_heads):
        super(muti_fusion_block,self).__init__()
         #层归一化
        self.layernorm1=nn.LayerNorm(text_outputsize)   
        self.layernorm2=nn.LayerNorm(image_outputsize)
        #交叉注意力机制
        self.crossattention_txt=MultiHeadAttention(text_outputsize,num_heads)
        self.crossattention_img=MultiHeadAttention(image_outputsize,num_heads)
    def forward(self,text_feature,image_feature):
            #交叉注意力
            txt_f=self.crossattention_txt(image_feature,text_feature,text_feature)
            img_f=self.crossattention_img(text_feature,image_feature,image_feature)
            #合并特征
            image_feature=img_f+image_feature
            text_feature=txt_f+text_feature
            #层归一化
            image_feature=self.layernorm1(image_feature)
            text_feature=self.layernorm2(text_feature)
            return text_feature,image_feature
    
class GatedFusionLayer(nn.Module):
    def __init__(self, feature_dim):
        super(GatedFusionLayer, self).__init__()
        # 生成门控向量的全连接层
        self.fc_gate = nn.Linear(feature_dim * 2, feature_dim)  # 输入大小是两个特征拼接后的大小
        self.fc_fusion = nn.Linear(feature_dim, feature_dim)

    def forward(self, image_feature, landmark_feature):
        # 将图像特征和面部特征点特征拼接起来
        #landmark_feature=landmark_feature.unsqueeze(dim=1)
        combined_features = torch.cat((image_feature, landmark_feature), dim=-1)

        # 计算门控向量，控制图像特征和面部特征点特征的加权
        gate = torch.sigmoid(self.fc_gate(combined_features))  # 生成门控值 (0到1之间)

        # 对特征进行加权融合
        fused_features = gate * image_feature + (1 - gate) * landmark_feature

        # 可以选择进一步处理融合后的特征
        fused_features = self.fc_fusion(fused_features)
        
        return fused_features

class clipe_msa(nn.Module):
    def __init__(self,cfg,clip_model,device):
        super(clipe_msa,self).__init__()
        #初始化参数
        text_inputsize=cfg['model']['text_inputsize']
        text_outputsize=cfg['model']['text_outputsize']
        image_inputsize=cfg['model']['image_inputsize']
        image_outputsize=cfg['model']['image_outputsize']
        #self.temperature=cfg['model']['temperature']
        alpha=cfg['model']['alpha']
        weight=cfg['model']['weight']
        gamma=cfg['model']['gamma']
        hard_negative_weight=cfg['model']['hard_negative_weight']
        num_heads=cfg['model']['num_heads']
        num_classes=cfg['model']['num_class']
        batch_size=cfg['model']['batch_size']
        self.a=cfg['model']['a']
        self.margin=cfg['model']['margin']
        self.lambda_spread=cfg['model']['lambda_spread']

        self.selfattention_img=MultiHeadAttention(image_outputsize,num_heads)
        self.selfattention_txt=MultiHeadAttention(text_outputsize,num_heads)
        self.muti_fusion_blocks = nn.ModuleList(
            [muti_fusion_block(text_inputsize,text_outputsize,image_inputsize,image_outputsize,num_heads) for _ in range(cfg['model']['hidesize'])]
            )
        self.layernorm1=nn.LayerNorm(text_outputsize)   
        self.layernorm2=nn.LayerNorm(image_outputsize)
        #self.landmark_encode=FaceEncoder1(output_dim=image_outputsize)
        self.landmark_encode=FaceEncoder2(in_dim=2,hidden_dim=256,out_dim=512,device=device)
        #self.exports=export_Block(dim=512,num_heads=num_heads,mlp_ratio=1,drop=0,attn_drop=0,depth=1)  #专家系统
        self.fusion_layer=GatedFusionLayer(feature_dim=image_outputsize)
        
        self.fc1=MLP(in_features=text_inputsize,hidden_features=512,out_features=text_outputsize,num_layers=4)
        self.fc2=MLP(in_features=image_inputsize,hidden_features=512,out_features=image_outputsize,num_layers=4)
        self.fc3=MLP(in_features=image_inputsize,hidden_features=512,out_features=image_outputsize,num_layers=4)
        """
        self.fc1=MLP(in_features=text_inputsize,hidden_features=512,out_features=text_outputsize)
        self.fc2=MLP(in_features=image_inputsize,hidden_features=512,out_features=image_outputsize)
        self.fc3=MLP(in_features=image_inputsize,hidden_features=512,out_features=image_outputsize)
        """
        self.liner_t=nn.Linear(text_outputsize,text_outputsize)
        self.liner_i=nn.Linear(image_outputsize,image_outputsize)
        #self.temperature = nn.Parameter(torch.full((1,len_index),fill_value=cfg['model']['temperature'],dtype=torch.float32))   #将温度参数初始化，变为可学习参
        self.temperature=nn.Parameter(torch.tensor(cfg['model']['temperature'],dtype=torch.float32))
        #loss
        self.centerloss_txt=CenterLoss(num_classes,text_outputsize,self.margin[0],self.lambda_spread[0])
        self.centerloss_img=CenterLoss(num_classes,image_outputsize,self.margin[1],self.lambda_spread[1])
        self.focalloss=FocalLoss(batch_size=batch_size,weight=weight,gamma=gamma)        #焦点损失
        #self.focalloss=FocalLoss(alpha=alpha,gamma=gamma)  
        self.ContrastiveLossWithHardNegatives=ContrastiveLossWithHardNegatives(self.temperature,hard_negative_weight=hard_negative_weight)    #对比损失
        self.AdaptiveCombinedLoss=AdaptiveCombinedLoss(self.focalloss,self.ContrastiveLossWithHardNegatives,self.centerloss_img,self.centerloss_txt,self.temperature)  
        #prompt
        self.learn_prompt=PromptLearner(cfg,cfg['experiment']['sentimentlabels'],clip_model,device[0])

    def forward(self,text_feature,image_feature,landmark):
        landmark_feature= self.landmark_encode(landmark)   #获得面部特征点编码特征
        #text_feature,image_feature,landmark_feature=self.exports(text_feature,image_feature,landmark_feature)  #专家系统
        text_feature=self.fc1(text_feature)
        image_feature=self.fc2(image_feature)
        landmark_feature=self.fc3(landmark_feature)
        _text_feature=text_feature
        _image_feature=image_feature
        image_feature=self.fusion_layer(image_feature,landmark_feature) #融合
        image_feature=self.selfattention_img(image_feature,image_feature,image_feature)
        text_feature=self.selfattention_txt(text_feature,text_feature,text_feature)
        for layer in self.muti_fusion_blocks:
            text_feature,image_feature=layer(text_feature,image_feature)
            #残差链接 
            text_feature=text_feature+_text_feature
            image_feature=image_feature+_image_feature
            #激活函数
            text_feature=F.gelu(text_feature)
            image_feature=F.gelu(image_feature)
        text_feature=text_feature+_text_feature
        image_feature=image_feature+_image_feature
        #线性层
        image_feature=self.liner_i(image_feature)
        text_feature=self.liner_t(text_feature)
        
        #曾归一化
        image_feature=self.layernorm1(image_feature)
        text_feature=self.layernorm2(text_feature)  
        text_output=text_feature.squeeze(1)
        image_output=image_feature.squeeze(1)
        return text_output,image_output
    
    def compute_loss(self,text_output,image_output,labels):
        loss=self.AdaptiveCombinedLoss(text_output,image_output,labels,self.a)
        #loss=self.contrastive_loss(text_output,image_output)
        return loss
    
    def return_learner_prompt(self):
        return self.learn_prompt