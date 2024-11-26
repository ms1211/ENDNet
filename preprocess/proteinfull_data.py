import os

import torch
import tqdm
from torch_geometric.datasets import TUDataset

from inari.utils import k_subgraph, mk_matching_matrix_vf3, feature_trans_numerical, cal_matching_matrix

name = "PROTEINS_full"
a = 0
dataset = TUDataset(root="raw", name=f"{name}")

targets = []
queries = []
mms = []

for i, t in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
    if t.has_isolated_nodes():
        continue
    print(f"t.y={t.y}")
    print(f"t={t}")
    assert a == 1

    q = k_subgraph(t, int(0.3 * t.num_nodes))

    t, feat_map = feature_trans_numerical(t)
    q["numeric_x"] = torch.unsqueeze(torch.IntTensor(list(map(lambda x: feat_map[x], map(str, q.x.tolist())))), 1)

    mm = mk_matching_matrix_vf3(t, q, feature='numeric_x', timeout=60)
    #mm = cal_matching_matrix(t, q, feature='numeric_x')

    if mm is None or mm.size(0) == 0:
        print(i)
        continue

    targets.append(t)
    queries.append(q)
    mms.append(mm)
    
os.makedirs("data", exist_ok=True)
torch.save([targets, queries, mms], f"data/{name.lower()}.pt")
