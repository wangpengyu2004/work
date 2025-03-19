import torch
from torch import nn
import torch.nn.functional as F
"""    
class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
"""
class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop_rate=0,
        num_layers=4,  # 新增参数：控制总层数
        layer_dims=None,  # 新增参数：自定义各层维度
        residual_connection=True,  # 新增残差连接选项
    ):
        super().__init__()
        # 参数验证
        if layer_dims is not None:
            if not isinstance(layer_dims, (list, tuple)):
                raise ValueError("layer_dims必须为列表或元组")
            if len(layer_dims) < 1:
                raise ValueError("layer_dims至少需要包含一个元素")
            num_layers = len(layer_dims) + 1  # 输入层 + 隐藏层 + 输出层
        elif hidden_features is None:
            hidden_features = in_features

        # 维度配置
        self.layer_dims = self._parse_layer_dims(
            in_features, hidden_features, out_features, num_layers, layer_dims
        )

        # 构建网络层
        self.layers = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.drops = nn.ModuleList()
        
        # 输入层
        self.layers.append(nn.Linear(self.layer_dims[0], self.layer_dims[1]))
        self.acts.append(act_layer())
        self.drops.append(nn.Dropout(drop_rate))

        # 隐藏层
        for i in range(1, len(self.layer_dims)-2):
            self.layers.append(nn.Linear(self.layer_dims[i], self.layer_dims[i+1]))
            self.acts.append(act_layer())
            self.drops.append(nn.Dropout(drop_rate))

        # 输出层
        self.fc_out = nn.Linear(self.layer_dims[-2], self.layer_dims[-1])
        
        self.residual = residual_connection

    def _parse_layer_dims(self, in_dim, hidden_dim, out_dim, num_layers, custom_dims):
        if custom_dims is not None:
            return [in_dim] + list(custom_dims) + [out_dim or custom_dims[-1]]
        
        dims = [in_dim]
        # 自动生成等差维度变化
        step = (hidden_dim - in_dim) / (num_layers-1)
        for i in range(1, num_layers):
            dims.append(int(in_dim + i*step))
        dims.append(out_dim or dims[-1])
        return dims

    def forward(self, x):
        identity = x
        for layer, act, drop in zip(self.layers, self.acts, self.drops):
            x = layer(x)
            x = act(x)
            x = drop(x)
        x = self.fc_out(x)
        
        if self.residual and identity.shape == x.shape:
            x = x + identity
        return x

class export(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            attn_drop=0,
            proj_drop=0,
            mlp_ratio=1,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.Transformer_l = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_ratio=mlp_ratio,
        )
        self.Transformer_t = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_ratio=mlp_ratio,
        )
        self.Transformer_v = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, x, cross_modality, mask_modality=None, mask=None):
        # x: [B, s, C]
        B, s, C = x.shape
        if cross_modality == 'l':
            x_a_mlp = self.Transformer_l(x, mask_modality, mask)
            return x_a_mlp
        if cross_modality == 't':
            x_t_mlp = self.Transformer_t(x, mask_modality, mask)
            return x_t_mlp
        if cross_modality == 'v':
            x_v_mlp = self.Transformer_v(x, mask_modality, mask)
            return x_v_mlp



class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            attn_drop=0.0,
            proj_drop=0.0,
            mlp_ratio=1.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.scale = head_dim ** -0.5
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=proj_drop,
        )


    def forward(self, x, mask_modality, mask=None):
        B, seq_len, C = x.shape

        q = self.q(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))  # [B, heads, s, s]

        if mask is not None:
            mask = mask.bool()
            mask = {'a':mask[:, :seq_len], 't':mask[:, seq_len:2*seq_len], 'v':mask[:, 2*seq_len:3*seq_len]}
            mask = mask[mask_modality]
            attn = self.attn_drop(attn.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(dim=-1).type_as(x))
            attn = torch.where(torch.isnan(attn), torch.full_like(attn, 0), attn)

        x_out = (attn @ v).transpose(1, 2).reshape(B, seq_len, C)
        x_out = x_out + self.mlp(x_out)

        return x_out


class export_Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            drop=0.0,
            attn_drop=0.0,
            depth=4
    ):
        super().__init__()
        self.drop = drop

        self.blocks = nn.ModuleList(
            [export(dim,num_heads=num_heads,attn_drop=attn_drop,proj_drop=drop,mlp_ratio=mlp_ratio,)for _ in range(depth)]
        )

    def forward(self,t_features,v_feartues,l_features, mask=None, modality=None):
            for block in self.blocks:
                v_feartues = t_features + block(t_features, cross_modality='t', mask_modality=modality, mask=mask)
                v_feartues = v_feartues + block(v_feartues, cross_modality='v', mask_modality=modality, mask=mask)
                l_features=  l_features + block(l_features, cross_modality='l', mask_modality=modality, mask=mask)
            return t_features, v_feartues, l_features