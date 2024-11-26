import os

import torch
import tqdm
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import subgraph
from torch_geometric.data import Data

from inari.utils import k_subgraph, mk_matching_matrix_vf3, feature_trans_categorical, fix_random_seed, cal_matching_matrix
import random
import math

fix_random_seed(42)

name = 'DD'
a = 0
dataset = TUDataset(root="raw", name=f'{name}')

targets = []
queries = []
mms = []

count = 0

for t in tqdm.tqdm(dataset):
    if t.has_isolated_nodes():
        continue
    #count += 1
    t = feature_trans_categorical(t)
    
    if t.num_nodes <= 300:
        t2 = t
        sub = math.floor(0.3 * t2.num_nodes)
        q = k_subgraph(t2, sub)
    else:
        t2 = k_subgraph(t, 300)
        t2["y"] = t.y
        sub = math.floor(0.3 * t2.num_nodes)
        q = k_subgraph(t2, sub)
    
    #mm = mk_matching_matrix_vf3(t2, q, timeout=300)
    mm = cal_matching_matrix(t2, q)

    if mm is None:
        continue
    elif mm.size(0) == 0:
        continue

    targets.append(t2)
    queries.append(q)
    mms.append(mm)

os.makedirs("data", exist_ok=True)
torch.save([targets, queries, mms], f"data/{name.lower()}.pt")
print("success!")
