"""Dataset loading: Cora / Citeseer (Planetoid public split) and ogbn-products (OGB split)."""
import os.path as osp
from dataclasses import dataclass

import torch
from torch_geometric.utils import remove_self_loops, to_undirected

DATA_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', 'data'))


@dataclass
class Graph:
    name: str
    x: torch.Tensor           # [N, F] node features
    y: torch.Tensor           # [N] labels
    edge_index: torch.Tensor  # [2, E] undirected, no self loops
    split: dict               # train / valid / test node indices
    num_classes: int

    @property
    def num_nodes(self):
        return self.x.size(0)


def random_split(n, seed=0, frac=(0.6, 0.2)):
    """Random 60/20/20 split of all nodes (fixed seed), used to test the small graphs with many labels."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return {'train': perm[:a].sort()[0], 'valid': perm[a:b].sort()[0], 'test': perm[b:].sort()[0]}


def load_graph(name, split='public'):
    """split: 'public' (Planetoid / OGB standard split) or 'random60' (60/20/20, Cora and Citeseer only)."""
    name = name.lower()
    if name in ('cora', 'citeseer'):
        from torch_geometric.datasets import Planetoid
        ds = Planetoid(osp.join(DATA_ROOT, 'planetoid'),
                       {'cora': 'Cora', 'citeseer': 'CiteSeer'}[name], split='public')
        d = ds[0]
        if split == 'random60':
            split = random_split(d.num_nodes)
        else:
            split = {'train': d.train_mask.nonzero().view(-1),
                     'valid': d.val_mask.nonzero().view(-1),
                     'test': d.test_mask.nonzero().view(-1)}
        # Row-normalised bag-of-words features, as in Kipf & Welling (2017).
        x = d.x / d.x.sum(1, keepdim=True).clamp(min=1)
        y, num_classes = d.y, ds.num_classes
    elif name in ('products', 'ogbn-products'):
        from ogb.nodeproppred import PygNodePropPredDataset
        ds = PygNodePropPredDataset('ogbn-products', root=osp.join(DATA_ROOT, 'ogb'))
        d = ds[0]
        split = ds.get_idx_split()
        x, y, num_classes = d.x, d.y.view(-1), ds.num_classes
        name = 'ogbn-products'
    else:
        raise ValueError(name)
    edge_index = remove_self_loops(d.edge_index)[0]
    edge_index = to_undirected(edge_index, num_nodes=x.size(0))
    return Graph(name, x, y, edge_index, split, num_classes)


def accuracy(graph, pred, idx):
    """Accuracy on node subset idx; ogbn-products goes through the official OGB evaluator."""
    idx = idx.cpu()
    y_true, y_pred = graph.y[idx].view(-1, 1), pred.cpu()[idx].view(-1, 1)
    if graph.name == 'ogbn-products':
        from ogb.nodeproppred import Evaluator
        return Evaluator('ogbn-products').eval({'y_true': y_true, 'y_pred': y_pred})['acc']
    return (y_true == y_pred).float().mean().item()
