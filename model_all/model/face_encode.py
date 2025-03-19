import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import numpy as np
from scipy.spatial import Delaunay
from torch_geometric.nn import GATConv, GlobalAttention

# 定义使用 GAT 和 GlobalAttention 的面部编码器
class FaceEncoder2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,device):
        super(FaceEncoder2, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=4, concat=False)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=4, concat=False)
        self.relu = nn.ReLU()
        # 定义 GlobalAttention 层，用于将节点特征汇聚为全局特征
        self.gate_nn = nn.Linear(out_dim, 1)  # 用于计算注意力权重
        self.global_att = GlobalAttention(self.gate_nn)
        self.device=device
    def forward(self, landmark):
        """
        输入：
            landmark: [B, 468, 2] 的张量，B 为批次大小，每个样本有 468 个面部关键点
        输出：
            x_global: [B, out_dim] 的全局面部特征向量
        """
        # 对每个样本的关键点进行归一化
        #x = (landmark - landmark.mean(dim=1, keepdim=True)) / landmark.std(dim=1, keepdim=True)
        # 得到批处理的图：节点特征、边索引、以及 batch 向量
        landmark=landmark.reshape((-1,468,2))
        x_all, edge_index, batch_all = self.get_face_graph(landmark)
        x_all,edge_index,batch_all=x_all.to(self.device[0]),edge_index.to(self.device[0]),batch_all.to(self.device[0])
        # 图卷积层
        x_all = self.conv1(x_all, edge_index)
        x_all = self.relu(x_all)
        x_all = self.conv2(x_all, edge_index)
        x_all = self.relu(x_all)
        # 利用 GlobalAttention 将节点特征聚合为全局特征，batch_all 指明每个节点所属的样本
        x_global = self.global_att(x_all, batch_all).reshape((-1,1,512))
        #x_global=x_global.to(self.device[0]).unsqueeze(dim=1)
        return x_global
    def get_face_graph(self,landmarks):
        """
        参数：
            landmarks: [B, N, 2] 的张量，B 表示样本数，N 表示每个样本的关键点数量
        返回：
            x_all: [B*N, 2] 的节点特征张量
            edge_index_all: [2, total_edges] 的边索引张量
            batch_all: [B*N] 的批次向量，指明每个节点属于哪个样本
        """
        B, N, _ = landmarks.shape
        x_list = []
        edge_index_list = []
        batch_list = []
        for b in range(B):
            # 将第 b 个样本的关键点转换为 numpy 数组，形状为 (N, 2)
            points = landmarks[b].cpu().numpy()
            # 对该样本进行 Delaunay 三角剖分
            tri = Delaunay(points)
            edges = set()
            # 遍历所有三角形，每个三角形由三个点索引构成
            for simplex in tri.simplices:
                edges.add((simplex[0], simplex[1]))
                edges.add((simplex[1], simplex[2]))
                edges.add((simplex[2], simplex[0]))
            # 转换为 tensor，形状为 [2, num_edges]
            edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
            # 由于后续要合并所有样本的图，需要对当前样本的边索引加上偏移量
            edge_index = edge_index + b * N
            
            x_list.append(landmarks[b])
            batch_list.append(torch.full((N,), b, dtype=torch.long))
            edge_index_list.append(edge_index)
        
        # 合并所有样本的节点特征、边索引和 batch 向量
        x_all = torch.cat(x_list, dim=0)            # [B*N, 2]
        batch_all = torch.cat(batch_list, dim=0)      # [B*N]
        edge_index_all = torch.cat(edge_index_list, dim=1)  # [2, total_edges]
        return x_all, edge_index_all, batch_all

class FaceEncoder1(nn.Module):
    def __init__(self, input_dim=136, hidden_dim=256, output_dim=256):
        super().__init__()
        # 空间编码模块
        self.spatial_encoder = nn.Sequential(
            nn.Linear(2, 32),  # 单个坐标点编码
            nn.GELU(),
            nn.LayerNorm(32)
        )
        
        # 区域注意力模块
        self.region_attention = nn.MultiheadAttention(embed_dim=32, num_heads=4)
        
        # 全局聚合
        self.aggregation = nn.Sequential(
            nn.Linear(468*32, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        """
        x: [batch_size, 936] -> [batch_size, 468, 2]
        """
        # 1. 坐标归一化
        x = x.view(-1, 468, 2)
        x = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)
        # 2. 单点编码
        point_features = self.spatial_encoder(x)  # [B,68,32]
        
        # 3. 区域注意力
        region_features, _ = self.region_attention(
            point_features, point_features, point_features
        )  # [B,68,32]
        
        # 4. 全局聚合
        batch_size = x.size(0)
        fused = self.aggregation(region_features.view(batch_size,1, -1))  # [B,512]
        return fused