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
from torch_geometric.nn import Linear, MessagePassing, pool
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import dense_to_sparse, softmax, subgraph, to_dense_adj

from inari.data import MyDataset, prepare_data_for_subgraph_task
from inari.loss import MMLoss
from inari.utils import fix_random_seed, metric_acc
from inari.model import SubCrossGMN
from inari.layers import SimpleMM, AffinityMM

a = 0#assert用の変数
device = torch.device("cuda:0")

def get_d_nd_index(t, mm):
    """
    match_nodes: ターゲットグラフでクエリーグラフとマッチしているノードの番号
    positive_edge: マッチしているノードからマッチしてるノードへ出ているエッジ
    delete_edge: positive_edge以外のエッジ
    """

    x = torch.any(mm != 0, dim=0)
    match_nodes = torch.masked_select(torch.arange(0, x.size(0), dtype=torch.long).to(device), x)

    positive_edge, _ = subgraph(match_nodes, t.edge_index)

    p_index = torch.isin(t.edge_index.T, positive_edge.T).all(-1).nonzero().squeeze()
    d_index = (~torch.isin(t.edge_index.T, positive_edge.T).all(-1)).nonzero().squeeze()
    

    return p_index, d_index


Device = Literal["cuda:0"]


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


        matching_matrix = F.softmax(matching_matrix, dim=1)

        return matching_matrix


class MatchingMatrix(nn.Module):
    def __init__(self, h_dim: int = 128):
        super().__init__()
        self.h_dim = h_dim
        self.simlarity_layer = Similarity(h_dim)
        self.normalization_layer = MatchingMatrixNormalization(h_dim)

    def forward(self, t_emb: Tensor, q_emb: Tensor, mask: BoolTensor) -> Tensor:
        mm = self.simlarity_layer(t_emb, q_emb)
        mm = self.normalization_layer(mm, mask)
        return mm


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

class ATTpolling_for_3Dtensor(torch.nn.Module):
    def __init__(self, h_dim, heads):
        super(ATTpolling_for_3Dtensor, self).__init__()
        self.h_dim = h_dim
        self.heads = heads
        self.batch = 32
        self.pooling1 = AttentionalAggregation(gate_nn=nn.Linear(h_dim, 1))
        self.pooling2 = AttentionalAggregation(gate_nn=nn.Linear(h_dim, 1))
        self.mlp1 = MLP(h_dim, 2 * h_dim, self.batch * h_dim)
        self.mlp2 = MLP(h_dim, 2 * h_dim, self.batch * h_dim)
        self.w_q = nn.Parameter(torch.Tensor(1, h_dim))
        self.lin = Linear(h_dim, heads * h_dim)
        nn.init.xavier_normal_(self.w_q)
        self.lin.reset_parameters()
    
    def forward(self, graph, M1, M2):
        k_graph_level_M1 = self.pooling1(x=M1, ptr=graph.ptr)
        k_graph_level_M1 = self.mlp1(k_graph_level_M1)
        k_graph_level_M1 = k_graph_level_M1.view(-1, self.h_dim)

        k_graph_level_M2 = self.pooling2(x=M2, ptr=graph.ptr)
        k_graph_level_M2 = self.mlp2(k_graph_level_M2)
        k_graph_level_M2 = k_graph_level_M2.view(-1, self.h_dim)

        s1 = (self.w_q * k_graph_level_M1).sum(dim=-1).unsqueeze(-1)
        s2 = (self.w_q * k_graph_level_M2).sum(dim=-1).unsqueeze(-1)
        att = torch.cat([s1, s2], dim=-1)
        att = F.softmax(att, dim=-1)
        att = att[graph.batch]
        end = M1 * att[:, 0].unsqueeze(-1) + M2 * att[:, 1].unsqueeze(-1)
        end = self.lin(end)
        end = end.view(-1, self.heads, self.h_dim)
        return end

class AEDGATConv(MessagePassing):
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

        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, att: Tensor):
        H, C = self.heads, self.out_channels

        x1 = x2 = self.lin(x).view(-1, H, C)

        att = torch.split(att, [C, C], dim=2)
        alpha1 = (x1 * att[0]).sum(dim=-1)
        alpha2 = (x2 * att[1]).sum(dim=-1)
        alpha = (alpha1, alpha2)

        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=None)

        out = self.propagate(edge_index, x=x1, alpha=alpha)

        out = out.view(-1, self.heads * self.out_channels)
        out = out + self.bias

        return out, alpha

    def edge_update(self, alpha_j: Tensor, index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        alpha = F.leaky_relu(alpha_j, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, " f"{self.out_channels}, heads={self.heads})"

fix_random_seed(42)


Tensors = tuple[Tensor, ...]


@dataclass
class AEDResult:
    mm: Tensor
    h_t_next: Tensor
    h_q_next: Tensor
    list_negative_DE: List[Tensor]
    list_mm: List[Tensor]


class AEDGAT_layer(nn.Module):
    def __init__(self, h_dim: int, heads: int = 8):
        super().__init__()
        self.h_dim = h_dim
        self.heads = heads
        self.matchingmatrix = MatchingMatrix(h_dim)
        self.pooling_q = AttentionalAggregation(gate_nn=nn.Linear(h_dim, 1))
        self.mlp0 = MLP(h_dim, 2 * h_dim, 2 * heads * h_dim)
        self.gat = AEDGATConv(h_dim, h_dim, heads=8)
        self.mlp1 = MLP(heads * h_dim, 2 * h_dim, h_dim)

    def forward(self, h_t, h_q, target, query, mask, mm,  h_t0, h_q0):
        n = torch.mm(mm, h_t)
        
        q = self.pooling_q(x=h_q, ptr=query.ptr)
        q = self.mlp0(q)
        q = q.view(-1, self.heads, 2 * self.h_dim)

        h_t_gat, a_t = self.gat(h_t, target.edge_index, q[target.batch])
        h_q_gat, a_q = self.gat(n, query.edge_index, q[query.batch])

        h_t = self.mlp1(h_t_gat) + h_t
        h_q = self.mlp1(h_q_gat) + h_q

        mm = self.matchingmatrix(h_t, h_q, mask)

        return h_t, h_q, a_t, a_q, mm


class AEDNet(nn.Module):
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
        self.layers = torch.nn.ModuleList([AEDGAT_layer(h_dim) for _ in range(num_layers)])
        self.matchingmatrix = MatchingMatrix(h_dim)

    def forward(self, target, query, mask):
        h_t = self.projection(target.x)
        h_t = torch.squeeze(h_t)
        h_t0 = h_t
        h_q = self.projection(query.x)
        h_q = torch.squeeze(h_q)
        h_q0 = h_q

        matching_matrices = []
        alpha_t = []
        alpha_q = []
        mm = self.matchingmatrix(h_t, h_q, mask)

        for layer in self.layers:
            h_t, h_q, a_t, a_q, mm = layer(h_t, h_q, target, query, mask, mm, h_t0, h_q0)
            matching_matrices.append(mm)
            alpha_t.append(a_t)
            alpha_q.append(a_q)

        return matching_matrices, alpha_t, alpha_q

class AEDLoss(nn.Module):
    def __init__(self, criterion=nn.MSELoss().to(device)):
        super().__init__()
        self.criterion = criterion
        self.lamda_t = 0.5#最終層以外のエッジ削除のlossの割合
        self.lamda_f = 0.5#最終層のエッジ削除のlossの割合
        self.lamda_all = 0.2#最終層以外のlossの割合

    def loss(self, matching_matrices: Tensor, mm: Tensor, mask: BoolTensor, alpha: Tensor, p_index: Tensor, d_index: Tensor) -> Tensor:
        #エッジ削除loss
        alpha_p = torch.stack(list(map(lambda a: a[p_index], alpha)))
        alpha_d = torch.stack(list(map(lambda a: a[d_index], alpha)))
        
        alpha_p = torch.sum(alpha_p, dim=1)
        alpha_d = torch.sum(alpha_d, dim=1)
        V = (alpha_p - alpha_d) / len(mm)
        V = torch.mean(V, dim=1).reshape(len(alpha_p), 1)
        V = 1 - V
        V_t = V[0:-1]#最終層以外のエッジ削除
        V_f = V[-1]#最終層のエッジ削除
        V_f = V_f.reshape(1, 1)#warningが出るためreshape
        zeros = torch.zeros(len(V[0:-1]), 1, dtype=torch.float32).to(device)
        zero = torch.zeros(1, 1, dtype=torch.float32).to(device)
        loss_de_t = self.criterion(V_t, zeros)#最終層以外のエッジ削除のloss
        loss_de_f = self.criterion(V_f, zero)#最終層のエッジ削除のloss
        
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
        
        loss_t = self.lamda_t * loss_de_t + (1 - self.lamda_t) * loss_mm_t
        loss_f = self.lamda_f * loss_de_f + (1 - self.lamda_f) * loss_mm_f
        loss_total = self.lamda_all * loss_t + (1 - self.lamda_all) * loss_f
        return loss_total

    def forward(self, list_mm: Tensor, mm: Tensor, alpha_t: Tensor, p_index: Tensor, d_index: Tensor, mask: BoolTensor) -> Tensor:
        loss = self.loss(torch.stack(list_mm), mm, mask, alpha_t, p_index, d_index)

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


model = AEDNet(num_features=num_features).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=25)

criterion = AEDLoss().to(device)

best_score = 0

pbar = tqdm.tqdm(range(1000))
max_acc = 0
max_ep = 0

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
        
        p_index, d_index = get_d_nd_index(target, mm)
        matching_matrices, alpha_t, alpha_q = model(target, query, mask)

        loss = criterion(matching_matrices, mm, alpha_t, p_index, d_index, mask)
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

            p_index, d_index = get_d_nd_index(target, mm)

            matching_matrices, alpha_t, alpha_q = model(target, query, mask)

            loss = criterion(matching_matrices, mm, alpha_t, p_index, d_index, mask)
            val_loss += loss.detach().item()

    scheduler.step(val_loss)

    test_acc = 0
    for target, query, mm, mask in tqdm.tqdm(test_loader, leave=False, desc="Test"):
        with torch.no_grad():
            target = target.to(device)
            query = query.to(device)
            mask = mask.to(device)
            mm = mm.to(device)

            matching_matrices, alpha_t, alpha_q = model(target, query, mask)
            test_acc += metric_acc(matching_matrices[-1].cpu(), mm)
    test_acc /= len(test_loader)
    if test_acc > max_acc:
        max_acc = test_acc
        max_ep = epoch

    pbar.set_postfix(lr=optimizer.param_groups[0]["lr"], val_loss=val_loss, test_acc=test_acc, max_acc=max_acc, max_ep=max_ep)
