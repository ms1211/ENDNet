import os

import torch
import tqdm
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import subgraph, to_networkx
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx

from inari.utils import k_subgraph, mk_matching_matrix_vf3, feature_trans_categorical, fix_random_seed, cal_matching_matrix
import random

fix_random_seed(42)

def add_onepoint(select, data):
    selectable = set()
    edge_index = data.edge_index
    for i in select:
        pred = set(edge_index[1, edge_index[0] == i])
        suc = set(edge_index[0, edge_index[1] == i])
        neigh = pred | suc
        selectable = selectable | neigh
    selectable = selectable - set(select)
    selectable = list(selectable)
    node = random.choice(list(selectable))
    return node

def gene_sg(seed, graph, size):
    col, row = graph.edge_index
    select = [int(seed)]
    while len(select) < size:
        node = add_onepoint(select=select, data=graph)
        select.append(int(node))
        select = set(select)
        select = list(select)
        select = sorted(select)
    if len(select) != size:
        print("wrong!!")
    select = torch.tensor(select)
    q_index = subgraph(select, graph.edge_index)
    node_idx = row.new_full((graph.num_nodes,), -1)
    node_idx[select] = torch.arange(select.size(0), device=row.device)
    edge_index = node_idx[q_index[0]]
    return Data(x=graph.x[select], edge_index=edge_index)    



name = 'SYNTHETIC'

dataset = TUDataset(root="raw", name=f'{name}')

targets = []
queries = []
mms = []

for t in tqdm.tqdm(dataset):
    if t.has_isolated_nodes():
        continue

    t = feature_trans_categorical(t)
    G = to_networkx(t, node_attrs=["x"])
    nx.draw_networkx(G)
    plt.show()

    seed = random.randint(0, t.num_nodes - 1)
    q = gene_sg(seed, t, int(0.3 * t.num_nodes))
    #mm = mk_matching_matrix_vf3(t, q, timeout=600)
    mm = cal_matching_matrix(t, q)

    if mm is None:
        continue
    elif mm.size(0) == 0:
        continue
    
    targets.append(t)
    queries.append(q)
    mms.append(mm)

os.makedirs("data", exist_ok=True)
torch.save([targets, queries, mms], f"data/{name.lower()}_aed.pt")
