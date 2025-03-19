import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class ContrastiveLossWithHardNegatives(nn.Module):
    def __init__(self, temperature, hard_negative_weight=1.0):
        super(ContrastiveLossWithHardNegatives, self).__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def forward(self, caption_embeddings, image_embeddings):
        # 归一化嵌入向量
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        caption_embeddings = F.normalize(caption_embeddings, p=2, dim=1)

        # 计算logits，即caption和image之间的相似度
        logits = torch.matmul(caption_embeddings, image_embeddings.t()) / self.temperature
        # 计算图像间的相似度
        images_similarity = torch.matmul(image_embeddings, image_embeddings.t())
        # 计算标题间的相似度
        captions_similarity = torch.matmul(caption_embeddings, caption_embeddings.t())
        #print()
        # 计算目标张量
        targets = torch.softmax((captions_similarity + images_similarity)/ (2 * self.temperature) , dim=1)
        
        # 计算困难负样本挖掘的损失
        hard_neg_mask = (logits >= logits.diagonal().unsqueeze(1))*(~(torch.eye(len(logits),device=logits.device).bool()))# 标识困难负样本
        hard_negatives = torch.where(hard_neg_mask, targets, torch.zeros_like(targets))  # 仅保留困难负样本
        logits_masked=logits*hard_neg_mask.float()
        hard_neg_loss_captions = F.cross_entropy(logits_masked, hard_negatives, reduction='none').sum()
        hard_neg_loss_image = F.cross_entropy(logits_masked.t(), hard_negatives.t(), reduction='none').sum()
        #print(hard_neg_mask,hard_negatives,hard_neg_loss)
        
        # 使用交叉熵损失计算标题的损失
        captions_loss = F.cross_entropy(logits, targets, reduction='none').mean()
        # 使用交叉熵损失计算图像的损失
        images_loss = F.cross_entropy(logits.t(), targets.t(), reduction='none').mean()
        # 总损失 = 对比损失 + 困难负样本损失
        total_loss = (captions_loss + images_loss) / 2
        total_loss += self.hard_negative_weight * ((hard_neg_loss_captions+hard_neg_loss_image)/2)

        return total_loss,logits
    
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, margin=2.5, lambda_spread=0.5):
        """
        :param num_classes: 类别数量
        :param feature_dim: 特征维度
        :param margin: 最小类别中心距离
        :param lambda_spread: 类间分散损失的权重
        """
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.margin = margin
        self.lambda_spread = lambda_spread
        # 初始化每个类别的中心
        self.centers = nn.Parameter(F.normalize(torch.randn(num_classes, feature_dim), dim=1))

    def forward(self, features, labels):
        # 获取当前样本对应的类别中心
        centers_batch = self.centers[labels]
        #计算样本中每个类的权重
        class_counts = torch.bincount(labels, minlength=self.num_classes).float()
        weights = class_counts / class_counts.sum()
        weights=1/weights[labels]#
        weights=torch.tensor(weights/weights.sum(),device=self.centers.device)
        # 计算样本到类别中心的距离
        l_center = torch.sum(weights* torch.sum((features - centers_batch) ** 2, dim=1))
        #l_center = torch.sum(torch.sum((features - centers_batch) ** 2, dim=1))
        
        # 计算类间分散损失 (L_spread)
        """
        pairwise_dist = torch.cdist(self.centers, self.centers, p=2)  # 类别中心两两间的欧氏距离
        mask = torch.eye(self.num_classes, device=self.centers.device).bool()
        pairwise_dist = pairwise_dist.masked_fill(mask, float('inf'))  # 忽略自身到自身的距离
        print(centers_batch,'\n',pairwise_dist)
        l_spread =torch.exp(-(pairwise_dist - self.margin)).mean()
        """
        #仅计算当前批次类别中心的类间分散损失
        pairwise_dist_all = torch.cdist(self.centers, self.centers, p=2)  # 全类别的欧氏距离
        unique_labels = torch.unique(labels)    #计算掩码
        mask = torch.zeros_like(pairwise_dist_all, dtype=torch.bool)
        mask[unique_labels] = True           # 仅标记当前批次的类别中心
        pairwise_dist = pairwise_dist_all.masked_fill(~mask, float('inf'))    # 忽略未涉及的类别
        pairwise_dist = pairwise_dist.masked_fill(~mask.T, float('inf'))
        pairwise_dist = pairwise_dist.masked_fill(torch.eye(self.num_classes, device=pairwise_dist.device).bool(), float('inf'))  # 忽略自身详细解释
        l_spread = F.relu(torch.exp(-(pairwise_dist - self.margin))-1).sum()
        #l_spread = F.relu(-(pairwise_dist - self.margin)).sum()
        # 总损失
        #print(pairwise_dist_all)
        #print(l_center,l_spread)
        loss = l_center + self.lambda_spread * l_spread
        return loss
"""
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        # 计算每个标签的 BCE loss
        logits=logits.cuda(non_blocking=True)
        targets=targets.cuda(non_blocking=True)
        print(probs)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # 计算 p_t
        p_t = targets * probs + (1 - targets) * (1 - probs)
        # 焦点损失核心公式
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
""" 

class FocalLoss(nn.Module):
    def __init__(self,batch_size,weight,gamma=2):
        super(FocalLoss,self).__init__()
        self.gamma=gamma
        self.weight=weight
        self.batch_size=batch_size

    def forward(self,preds,labels):
        eps=1e-7
        device=preds.device
        labels=np.asarray(labels)
        alpha=torch.tensor(np.take(self.weight, labels, axis=0),device=device)
        target = torch.eye(self.batch_size, device=device)
        ce=-1*torch.log(preds+eps)*target
        floss=torch.pow((1-preds),self.gamma)*ce
        floss=torch.mul(floss,alpha)
        floss=torch.sum(floss,dim=1)
        return torch.sum(floss)#torch.mean(floss)


class AdaptiveCombinedLoss(nn.Module):
    def __init__(self,FocalLoss,contrastive_loss,CenterLoss_img,Centerloss_txt,temperature):
        super(AdaptiveCombinedLoss,self).__init__()
        self.focalloss=FocalLoss
        self.contrastive_loss=contrastive_loss
        self.center_img_loss=CenterLoss_img
        self.center_txt_loss=Centerloss_txt
        self.temperature=temperature
    def forward(self,text_output,image_output,labels,a):
        c_loss,logits=self.contrastive_loss(text_output,image_output)     #对比损失函数
        image_output=torch.softmax(image_output,dim=1)
        text_output=torch.softmax(text_output,dim=1)
        #targets=torch.eye(12)
        #preds=logits
        preds=torch.softmax(logits,dim=1)
        center_loss_img=self.center_img_loss(image_output,labels)      #中心损失函数
        center_loss_txt=self.center_txt_loss(text_output,labels)
        center_loss=(center_loss_txt+center_loss_img)/2
        f_loss=self.focalloss(preds,labels)                #焦点损失函数
        print(preds)
        print(c_loss,center_loss,f_loss)
        c,b=1,0.7
        all_loss=f_loss*b+c*c_loss+a*center_loss
        #print(c_loss,center_loss)
        return all_loss
    