"""Sparse graph operators, label propagation and Correct & Smooth (all via sparse matmul,
so they scale to ogbn-products without materialising per-edge messages)."""
import torch


def _sparse(row, col, val, n):
    return torch.sparse_coo_tensor(torch.stack([row, col]), val, (n, n)).coalesce()


def gcn_adj(edge_index, n, self_loops=True):
    """D^-1/2 (A + I) D^-1/2."""
    row, col = edge_index
    if self_loops:
        loop = torch.arange(n, device=row.device)
        row, col = torch.cat([row, loop]), torch.cat([col, loop])
    deg = torch.zeros(n, device=row.device).scatter_add_(0, row, torch.ones(row.numel(), device=row.device))
    dinv = deg.pow(-0.5)
    dinv[torch.isinf(dinv)] = 0
    return _sparse(row, col, dinv[row] * dinv[col], n)


def mean_adj(edge_index, n):
    """D^-1 A (neighbour mean, used by GraphSAGE)."""
    row, col = edge_index
    deg = torch.zeros(n, device=row.device).scatter_add_(0, row, torch.ones(row.numel(), device=row.device))
    return _sparse(row, col, 1.0 / deg.clamp(min=1)[row], n)


@torch.no_grad()
def hop_features(adj, x, hops):
    """[X, AX, A^2 X, ..., A^K X] for SIGN-style decoupled models."""
    out = [x]
    for _ in range(hops):
        out.append(torch.sparse.mm(adj, out[-1]))
    return out


@torch.no_grad()
def propagate(adj, y0, alpha, iters, clamp=False):
    """Personalised-PageRank style propagation  Y <- (1 - alpha) Y0 + alpha A Y."""
    out = y0.clone()
    for _ in range(iters):
        out = (1 - alpha) * y0 + alpha * torch.sparse.mm(adj, out)
        if clamp:
            out.clamp_(0, 1)
    return out


def one_hot(y, idx, n, num_classes):
    out = torch.zeros(n, num_classes, device=y.device)
    out[idx, y[idx]] = 1
    return out


@torch.no_grad()
def label_propagation(adj, y, train_idx, num_classes, alpha=0.9, iters=50):
    """Class scores obtained by diffusing the training labels over the graph only."""
    out = propagate(adj, one_hot(y, train_idx, y.numel(), num_classes), alpha, iters)
    return out / out.sum(1, keepdim=True).clamp(min=1e-12)


@torch.no_grad()
def correct_and_smooth(adj, probs, y, train_idx, alpha1, alpha2, iters=50):
    """Correct & Smooth (Huang et al., ICLR 2021) with autoscale; uses training labels only."""
    n, c = probs.shape
    y_train = one_hot(y, train_idx, n, c)[train_idx]
    # Correct: spread training residuals, rescaled to the average training error size.
    err0 = torch.zeros_like(probs)
    err0[train_idx] = y_train - probs[train_idx]
    err = propagate(adj, err0, alpha1, iters)
    sigma = err0[train_idx].abs().sum(1).mean()
    scale = sigma / err.abs().sum(1, keepdim=True).clamp(min=1e-12)
    corrected = probs + scale * err
    corrected[train_idx] = y_train
    # Smooth: label-propagate the corrected predictions.
    return propagate(adj, corrected, alpha2, iters)
