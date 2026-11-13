"""Node classifiers. Every model exposes forward(idx) -> logits of the nodes in idx; the graph
tensors (features, sparse adjacency, precomputed hop features) are held as plain attributes so
they are not copied into state_dicts."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(dim, bn):
    return nn.BatchNorm1d(dim) if bn else nn.Identity()


class MLP(nn.Module):
    def __init__(self, x, in_dim, hidden, out_dim, layers=2, dropout=0.5, bn=False, input_drop=0.0):
        super().__init__()
        self.x = x
        self.body = MLPHead(in_dim, hidden, out_dim, layers, dropout, bn, input_drop)

    def forward(self, idx):
        return self.body(self.x[idx])


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, layers=2, dropout=0.5, bn=False, input_drop=0.0):
        super().__init__()
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList(_norm(d, bn) for d in dims[1:-1])
        self.dropout, self.input_drop = dropout, input_drop

    def forward(self, h):
        h = F.dropout(h, self.input_drop, self.training)
        for i, lin in enumerate(self.lins):
            if i > 0:
                h = F.dropout(h, self.dropout, self.training)
            h = lin(h)
            if i < len(self.lins) - 1:
                h = F.relu(self.norms[i](h))
        return h


class GCN(nn.Module):
    """Kipf & Welling (2017), full-batch with a sparse normalised adjacency."""
    def __init__(self, x, adj, in_dim, hidden, out_dim, layers=2, dropout=0.5, bn=False):
        super().__init__()
        self.x, self.adj = x, adj
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList(_norm(d, bn) for d in dims[1:-1])
        self.dropout = dropout

    def forward(self, idx):
        h = self.x
        for i, lin in enumerate(self.lins):
            h = F.dropout(h, self.dropout, self.training)
            h = torch.sparse.mm(self.adj, lin(h))
            if i < len(self.lins) - 1:
                h = F.relu(self.norms[i](h))
        return h[idx]


class SAGE(nn.Module):
    """GraphSAGE-mean (Hamilton et al., 2017), full-batch."""
    def __init__(self, x, adj_mean, in_dim, hidden, out_dim, layers=3, dropout=0.5, bn=True):
        super().__init__()
        self.x, self.adj = x, adj_mean
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        self.self_lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.nb_lins = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList(_norm(d, bn) for d in dims[1:-1])
        self.dropout = dropout

    def forward(self, idx):
        h = self.x
        for i in range(len(self.self_lins)):
            h = F.dropout(h, self.dropout, self.training)
            h = self.self_lins[i](h) + torch.sparse.mm(self.adj, self.nb_lins[i](h))
            if i < len(self.self_lins) - 1:
                h = F.relu(self.norms[i](h))
        return h[idx]


class GAT(nn.Module):
    """Velickovic et al. (2018); edge-level attention, used on the small graphs only."""
    def __init__(self, x, edge_index, in_dim, hidden, out_dim, heads=8, dropout=0.6):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.x, self.edge_index = x, edge_index
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, out_dim, heads=1, dropout=dropout)
        self.dropout = dropout

    def forward(self, idx):
        h = F.dropout(self.x, self.dropout, self.training)
        h = F.elu(self.conv1(h, self.edge_index))
        h = F.dropout(h, self.dropout, self.training)
        return self.conv2(h, self.edge_index)[idx]


class APPNP(nn.Module):
    """Klicpera et al. (2019): MLP predictions followed by K steps of personalised PageRank."""
    def __init__(self, x, adj, in_dim, hidden, out_dim, layers=2, dropout=0.5, k=10, teleport=0.1):
        super().__init__()
        self.x, self.adj, self.k, self.teleport = x, adj, k, teleport
        self.mlp = MLPHead(in_dim, hidden, out_dim, layers, dropout)

    def forward(self, idx):
        h0 = self.mlp(F.dropout(self.x, self.mlp.dropout, self.training))
        h = h0
        for _ in range(self.k):
            h = self.teleport * h0 + (1 - self.teleport) * torch.sparse.mm(self.adj, h)
        return h[idx]


class SIGN(nn.Module):
    """Rossi et al. (2020): per-hop linear encoders on precomputed A^k X, concatenated into an MLP.
    Only the rows in idx are touched, so it trains with mini-batches on ogbn-products."""
    def __init__(self, feats, in_dim, hidden, out_dim, layers=2, dropout=0.5, input_drop=0.0, bn=True):
        super().__init__()
        self.feats = feats
        self.hop_lins = nn.ModuleList(nn.Linear(in_dim, hidden) for _ in feats)
        self.hop_norms = nn.ModuleList(_norm(hidden, bn) for _ in feats)
        self.act = nn.PReLU()
        self.head = MLPHead(hidden * len(feats), hidden, out_dim, layers, dropout, bn)
        self.dropout, self.input_drop = dropout, input_drop

    def forward(self, idx):
        hs = []
        for f, lin, norm in zip(self.feats, self.hop_lins, self.hop_norms):
            h = F.dropout(f[idx], self.input_drop, self.training)
            hs.append(norm(lin(h)))
        h = F.dropout(self.act(torch.cat(hs, 1)), self.dropout, self.training)
        return self.head(h)


FULL_GRAPH_MODELS = {'gcn', 'sage', 'gat', 'appnp'}  # forward computes all nodes at once


def build_model(name, ctx, cfg, num_classes):
    """ctx holds the device tensors: x, adj (gcn-normalised), adj_mean, edge_index, feats."""
    in_dim = ctx['x'].size(1)
    h, d = cfg['hidden'], cfg['dropout']
    if name == 'mlp':
        return MLP(ctx['x'], in_dim, h, num_classes, cfg.get('layers', 2), d, cfg.get('bn', False),
                   cfg.get('input_drop', 0.0))
    if name == 'gcn':
        return GCN(ctx['x'], ctx['adj'], in_dim, h, num_classes, cfg.get('layers', 2), d, cfg.get('bn', False))
    if name == 'sage':
        return SAGE(ctx['x'], ctx['adj_mean'], in_dim, h, num_classes, cfg.get('layers', 2), d, cfg.get('bn', False))
    if name == 'gat':
        return GAT(ctx['x'], ctx['edge_index'], in_dim, h, num_classes, cfg.get('heads', 8), d)
    if name == 'appnp':
        return APPNP(ctx['x'], ctx['adj'], in_dim, h, num_classes, cfg.get('layers', 2), d,
                     cfg.get('k', 10), cfg.get('teleport', 0.1))
    if name == 'sign':
        return SIGN(ctx['feats'], in_dim, h, num_classes, cfg.get('layers', 2), d,
                    cfg.get('input_drop', 0.0), cfg.get('bn', True))
    raise ValueError(name)
