import os

import torch
import tqdm
from torch_geometric.datasets import TUDataset

from inari.utils import k_subgraph, mk_matching_matrix, feature_trans_categorical

name = 'COX2'

dataset = TUDataset(root="raw", name=f'{name}')

targets = []
queries = []
mms = []

for t in tqdm.tqdm(dataset):
    if t.has_isolated_nodes():
        continue

    t = feature_trans_categorical(t)

    q = k_subgraph(t, int(0.3 * t.num_nodes))
    mm = mk_matching_matrix(t, q)

    if mm is None:
        continue

    targets.append(t)
    queries.append(q)
    mms.append(mm)

os.makedirs("data", exist_ok=True)
torch.save([targets, queries, mms], f"data/{name.lower()}.pt")
