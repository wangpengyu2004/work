import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
#attention
class attention(nn.Module):
    def __init__(self):
        super(attention,self).__init__()
    def forward(self,q,k,v,mask=None, e=-1e-20):
        batch_size,length,k_model=k.size()
        k_t=k.transpose(1,2)
        #计算分数
        score=torch.matmul(q,k_t)/math.sqrt(k_model)
        #mask
        if mask is not None:
            score=score.masked_fill(mask==0,e)
        #权重
        attention_score=torch.softmax(score,dim=-1,dtype=torch.float32)
        v_t = v.transpose(1, 2)
        output=torch.matmul(v_t,attention_score).transpose(1,2)
        
        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        # 定义线性层
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        #self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, mask=None, e=-1e-20):
        batch_size, length, _ = k.size()
        # 线性变换并拆分头部
        queries = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # 计算分数
        k_t = keys.transpose(2, 3)
        score = torch.matmul(queries, k_t) / math.sqrt(self.head_dim)
        # 应用掩码
        if mask is not None:
            score = score.masked_fill(mask == 0, e)
        attention_score = torch.softmax(score, dim=-1)
        output = torch.matmul(attention_score, values)
        # 重新排列并合并头部
        output = output.transpose(1, 2).contiguous().view(batch_size, length, -1)
        #output = self.fc_out(output)
        return output
class crossattention(nn.Module):
    def __init__(self,d_model):
        super(crossattention,self).__init__()
        self.attention=attention()
        self.qury=nn.Linear(d_model,d_model)
        self.key=nn.Linear(d_model,d_model)
        self.value=nn.Linear(d_model,d_model)
        #self.output=nn.Linear(d_model,d_model)
    
    def forward(self,q,k,v,mask=None):
        # q  /  k,v 
        qury=self.qury(q)
        key=self.key(k)
        value=self.value(v)
        # [batch_szie,length,d_model]
        #计算注意力
        score=self.attention(qury,key,value,mask=mask)
        #output=self.output(score)
        return score

class selfattention(nn.Module):
    def __init__(self,d_model):
        super(selfattention,self).__init__()
        self.attention=attention()
        self.qury=nn.Linear(d_model,d_model)
        self.key=nn.Linear(d_model,d_model)
        self.value=nn.Linear(d_model,d_model)
        #self.output=nn.Linear(d_model,d_model)
    
    def forward(self,q,k,v,mask=None):
        # q  /  k,v 
        qury=self.qury(q)
        key=self.key(k)
        value=self.value(v)
        # [batch_szie,length,d_model]
        #计算注意力
        score=self.attention(qury,key,value,mask=mask)
        #output=self.output(score)
        return score





