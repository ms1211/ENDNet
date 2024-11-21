import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from einops import rearrange, reduce
from torch import BoolTensor, Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import Linear, MessagePassing, pool, GATConv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import dense_to_sparse, softmax, subgraph, to_dense_adj

from inari.data import MyDataset, prepare_data_for_subgraph_task
from inari.loss import MMLoss
from inari.utils import fix_random_seed, metric_acc, metric_f1
from inari.model import SubCrossGMN
from inari.layers import SimpleMM, AffinityMM

a = 0#assert用の変数
#torch.set_printoptions(edgeitems=2000)#テンソルの中身を詳しく見るとき用
#f = open('alpha_d.txt', 'w')

def cosine_matrix(h_q, h_t):
    dot = torch.matmul(h_q, h_t.T)
    norm = torch.matmul(torch.norm(h_q, dim=1).unsqueeze(-1), torch.norm(h_t, dim=1).unsqueeze(0))
    mm = torch.where(norm != 0, dot / (norm + 1e-9), -1)#1e-9を足しているのは0割りを避けるため
    return mm

Device = Literal["cuda"]

class FeatureType(Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    NONE = "none"


class SimilarityType(Enum):
    T = "τ"
    DOT = "dot"
    LEAKY_RELU = "leaky_relu"


class NormType(Enum):
    SOFTMAX = "softmax"
    T = "τ"
    MEAN = "mean"


@dataclass
class LayerParam:
    is_batch_norm: bool = True
    is_norm_affine: bool = False
    drop_rate: float = 0
    has_residual: bool = True
    negative_slope_in_norm: float = 0.2
    leaky_relu_GAT_rate: int = 2
    drop_edge_p: float = 0


class ProjectLayer(nn.Module):
    def __init__(self, num_features: int, h_dim: int, feature_type: FeatureType):
        super().__init__()
        self.feature_type = feature_type

        match feature_type:
            case FeatureType.CATEGORICAL:
                self.project = nn.Embedding(num_features + 1, h_dim)

            case FeatureType.NUMERICAL:
                self.project = nn.Linear(3, h_dim)
                self.bn = nn.BatchNorm1d(h_dim)

            case FeatureType.NONE:
                pass

    def forward(self, x: Tensor):
        match self.feature_type:
            case FeatureType.CATEGORICAL:
                x = self.project(x)

            case FeatureType.NUMERICAL:
                x = self.project(x)
                x = self.bn(x)
                x = F.elu(x)

            case FeatureType.NONE:
                x = None

        return x

class Similarity(nn.Module):
    def __init__(self, h_dim: int):
        super().__init__()
        self.h_dim = h_dim
        self.h_dim_sqrt = h_dim ** 0.5

    def forward(self, h_t: Tensor, h_q: Tensor) -> Tensor:
        mm = torch.mm(h_q, h_t.T)
        mm = mm / self.h_dim_sqrt#勾配消失を防ぐため

        return mm


class MatchingMatrixNormalization(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.tau = nn.Parameter(torch.FloatTensor(1))#τ
        nn.init.constant_(self.tau, 0)

    def forward(self, matrix: Tensor, mask: BoolTensor) -> Tensor:
        matching_matrix = matrix * mask.to(torch.float32)
        matching_matrix = matching_matrix / F.sigmoid(self.tau)
        matching_matrix += -1e9 * (~mask).to(torch.float32)

        mm = F.softmax(matching_matrix, dim=1)

        return mm, matching_matrix


class MatchingMatrix(nn.Module):
    def __init__(self, h_dim: int = 128):
        super().__init__()
        self.h_dim = h_dim
        self.simlarity_layer = Similarity(h_dim)
        self.normalization_layer = MatchingMatrixNormalization(h_dim)

    def forward(self, t_emb: Tensor, q_emb: Tensor, mask: BoolTensor) -> Tensor:
        mm = cosine_matrix(q_emb, t_emb)
        mm, matching_matrix = self.normalization_layer(mm, mask)
        return mm, matching_matrix


class MLP(nn.Module):
    def __init__(self, input_dim: int, hid_dim: int, out_dim: int):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        self.L1 = nn.Linear(input_dim, hid_dim)
        self.L2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.L1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.L2(x)
        x = self.bn2(x)
        x = F.elu(x)
        return x

class ENDGATConv2(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope

        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.W = torch.nn.Parameter(torch.Tensor(self.heads, self.in_channels, self.out_channels)).to(device)
        nn.init.xavier_uniform_(self.W)

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
    
    def forward(self, x: Tensor, edge_index: Adj):
        H, C = self.heads, self.out_channels

        x1 = torch.matmul(x, self.W)
        x2 = rearrange(x1, "h n c -> h c n")
        
        alpha = torch.matmul(x1, x2)#内積
        norm = torch.matmul(torch.norm(x, dim=1).unsqueeze(-1), torch.norm(x, dim=1).unsqueeze(0))#cos
        alpha = torch.mean(input=alpha, dim=0)
        alpha = torch.squeeze(alpha)
        all_edge = to_dense_adj(edge_index, max_num_nodes=len(x))
        all_edge = torch.squeeze(all_edge)
        alpha = torch.where(all_edge == 1, alpha, 0)#内積
        alpha = torch.where(norm != 0, alpha, 0)

        alpha = F.sigmoid(alpha)
        alpha= torch.squeeze(alpha)

        alpha = alpha[torch.where(all_edge == 1)]
        
        x1 = torch.mean(input=x1, dim=0)
        out = self.propagate(edge_index, x=x1, alpha=alpha)
        #out = out + self.bias

        return out

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j


fix_random_seed(42)

device = torch.device("cuda")

Tensors = tuple[Tensor, ...]


@dataclass
class ENDResult:
    mm: Tensor
    h_t_next: Tensor
    h_q_next: Tensor
    list_negative_DE: List[Tensor]
    list_mm: List[Tensor]


class ENDGAT_layer(nn.Module):
    def __init__(self, h_dim: int, heads: int = 8):
        super().__init__()
        self.h_dim = h_dim
        self.heads = heads
        self.matchingmatrix = MatchingMatrix(h_dim)
        self.gat = ENDGATConv2(h_dim, h_dim, heads=8)
        self.mlp1 = MLP(h_dim, 2 * h_dim, h_dim)#ENDGATConv2用
        self.beta = nn.Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.beta, 0)
        self.ganma = nn.Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.ganma, 0)

    def forward(self, h_t, h_q, target, query, mask, mm, matching_matrix):
        
        if mm is not None:
            matching_matrix = torch.max(matching_matrix, dim=0).values
            matching_matrix = torch.unsqueeze(matching_matrix, dim = 1)
            matching_matrix = matching_matrix.repeat(1, self.h_dim)
            h_t = torch.where(matching_matrix > F.tanh(self.beta), h_t, 0)
            n = torch.mm(mm, h_t)
        else:
            n = h_q

        h_t_gat = self.gat(h_t, target.edge_index)
        h_q_gat = self.gat(n, query.edge_index)
        
        gm = F.sigmoid(self.ganma)
        h_t = torch.where(h_t != 0, gm * self.mlp1(h_t_gat) + (1 - gm) * h_t, h_t)
        #h_t = gm * self.mlp1(h_t_gat) + (1 - gm) * h_t
        h_q = gm * self.mlp1(h_q_gat) + (1 - gm) * h_q
        
        
        mm, matching_matrix = self.matchingmatrix(h_t, h_q, mask)

        return h_t, h_q, mm, matching_matrix, gm


class ENDNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_layers: int = 7,
        h_dim: int = 64,
        heads: int = 8,
        feature_type: FeatureType = FeatureType.CATEGORICAL,
        #feature_type: FeatureType = FeatureType.NUMERICAL,
    ):
        super().__init__()
        self.heads = heads
        self.h_dim = h_dim

        self.projection = ProjectLayer(num_features, h_dim, feature_type)
        self.layers = torch.nn.ModuleList([ENDGAT_layer(h_dim) for _ in range(num_layers)])
        self.matchingmatrix = MatchingMatrix(h_dim)

    def forward(self, target, query, mask):
        h_t = self.projection(target.x)
        h_q = self.projection(query.x)
        h_t = torch.squeeze(h_t)
        h_q = torch.squeeze(h_q)

        matching_matrices = []
        alpha_t = []
        alpha_q = []
        f_t = []
        #mm = None
        mm, matching_matrix = self.matchingmatrix(h_t, h_q, mask)

        for layer in self.layers:
            h_t, h_q, mm, matching_matrix, gm = layer(h_t, h_q, target, query, mask, mm, matching_matrix)
            matching_matrices.append(mm)
            f_t.append(h_t)

        return matching_matrices, f_t, gm

#特徴量lossGAT版
class ENDLoss(nn.Module):
    def __init__(self, criterion=nn.MSELoss().to(device)):
        super().__init__()
        self.criterion = criterion
        self.lamda_mm = 0.5

    def loss(self, matching_matrices: Tensor, mm: Tensor, mask: BoolTensor, feature: Tensor) -> Tensor:
        
        #マッチングloss
        m_t = matching_matrices[0:-1] * (mm == 1).to(torch.float32) * mask.to(torch.float32)#最終層以外でマッチするもの
        m_f = matching_matrices[-1] * (mm == 1).to(torch.float32) * mask.to(torch.float32)#最終層でマッチするもの
        nm_t = matching_matrices[0:-1] * (mm == 0).to(torch.float32) * mask.to(torch.float32)#最終層以外でマッチしないもの
        nm_f = matching_matrices[-1] * (mm == 0).to(torch.float32) * mask.to(torch.float32)#最終層でマッチしないもの
        m_t = torch.sum(m_t, dim=2).reshape(len(matching_matrices[0:-1]) * len(mm), 1)
        m_f = torch.sum(m_f, dim=1).reshape(len(mm), 1)
        nm_t = torch.sum(nm_t, dim=2).reshape(len(matching_matrices[0:-1]) * len(mm), 1)
        nm_f = torch.sum(nm_f, dim=1).reshape(len(mm), 1)
        U_t = 1 - (m_t - nm_t)
        U_f = 1 - (m_f - nm_f)
        zeros_m = torch.zeros(len(matching_matrices[0:-1]) * len(mm), 1, dtype=torch.float32).to(device)
        loss_mm_t = self.criterion(U_t, zeros_m)#最終層以外のマッチング行列のloss
        loss_mm_f = self.criterion(U_f, torch.zeros(len(mm), 1, dtype=torch.float32).to(device))#最終層のマッチング行列のloss
        
        loss_mm = self.lamda_mm * loss_mm_t + (1 - self.lamda_mm) * loss_mm_f
        loss_total = loss_mm
        return loss_total

    def forward(self, list_mm: Tensor, mm: Tensor, mask: BoolTensor, f_t: Tensor) -> Tensor:
        loss = self.loss(torch.stack(list_mm), mm, mask, f_t)

        return loss

@dataclass
class LossParams:
    λ_t: float = 0.8
    λ_t_de: float = 0.5
    λ_de: float = 0.5

num_features = 35
dataset = MyDataset("data/cox2.pt", num_features)
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=prepare_data_for_subgraph_task, num_workers=1, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=prepare_data_for_subgraph_task, num_workers=1, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=prepare_data_for_subgraph_task, num_workers=1, pin_memory=True)

model = ENDNet(num_features=num_features).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=25)

criterion = ENDLoss().to(device)#END用loss(特徴量lossGAT版)

best_score = 0
max_acc = 0
max_ep = 0

pbar = tqdm.tqdm(range(1000))

for epoch in pbar:
    pbar.set_description(f"Epoch {epoch}")

    model.train()
    count = 0
    train_loss = 0


    for target, query, mm, mask in tqdm.tqdm(train_loader, leave=False, desc="Train"):
        target = target.to(device)
        query = query.to(device)
        mm = mm.to(device)
        mask = mask.to(device)
        
        matching_matrices, f_t, gm = model(target, query, mask)#END
        
        loss = criterion(matching_matrices, mm, mask, f_t)#END
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    val_loss = 0
    val_acc = 0
    for target, query, mm, mask in tqdm.tqdm(val_loader, leave=False, desc="Val"):
        with torch.no_grad():
            target = target.to(device)
            query = query.to(device)
            mask = mask.to(device)
            mm = mm.to(device)

            matching_matrices, f_t, gm = model(target, query, mask)#END
            
            loss = criterion(matching_matrices, mm, mask, f_t)#END
            val_loss += loss.detach().item()

    scheduler.step(val_loss)

    test_acc = 0
    test_f1 = 0
    for target, query, mm, mask in tqdm.tqdm(test_loader, leave=False, desc="Test"):
        with torch.no_grad():
            target = target.to(device)
            query = query.to(device)
            mask = mask.to(device)
            mm = mm.to(device)

            matching_matrices, f_t, gm = model(target, query, mask)#END
            test_acc += metric_acc(matching_matrices[-1].cpu(), mm)
    test_acc /= len(test_loader)
    if test_acc > max_acc:
        max_acc = test_acc
        max_ep = epoch
    #f.write(f'test_acc={test_acc}\n')
    #f.write(f'\n')

    pbar.set_postfix(lr=optimizer.param_groups[0]["lr"], val_loss=val_loss, test_acc=test_acc, max_acc=max_acc, max_ep=max_ep, gm=float(gm))
#f.close()
