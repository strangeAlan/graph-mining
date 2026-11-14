"""Cooperation of Experts (CoE, Wang et al., ICML 2025) adapted to semi-supervised node classification.

Experts are heterogeneous node classifiers (feature-only MLP, label-only LP, and GNNs with different
propagation schemes). Their class distributions are combined by a learned cooperation matrix
    z_i = sum_j Theta_j p_{j,i},        Theta_j in R^{C x C},
trained with cross-entropy plus the CoE large-margin term on the predicted distribution.

Adaptation to the transductive setting: with a few labelled nodes every expert fits its own training
labels (LP even returns them verbatim), so Theta cannot be fitted on in-sample predictions as in the
original code. Theta is fitted on out-of-fold predictions instead (cross-fitting over the training
nodes), and applied to experts trained on the full training set.
"""
import itertools

import torch
import torch.nn.functional as F

from .prop import label_propagation
from .train import fit


def stratified_folds(y, idx, k, seed):
    """Fold id for every node in idx, balanced per class."""
    g = torch.Generator().manual_seed(seed)
    fold = torch.empty(idx.numel(), dtype=torch.long)
    labels = y[idx].cpu()
    for c in labels.unique():
        pos = (labels == c).nonzero().view(-1)
        pos = pos[torch.randperm(pos.numel(), generator=g)]
        fold[pos] = torch.arange(pos.numel()) % k
    return fold.to(idx.device)


def expert_probs(graph, ctx, expert, cfg, idx, labels, seed, n_true=None):
    """Class distribution of all nodes from one expert trained on (idx, labels)."""
    if expert == 'lp':
        y = ctx['y'].clone()
        y[idx] = labels  # LP diffuses exactly the given (true or pseudo) labels
        return label_propagation(ctx['adj'], y, idx, graph.num_classes, cfg['lp']['alpha'], cfg['lp']['iters'])[0]
    return fit(graph, ctx, expert, cfg[expert], idx, labels, seed, n_true=n_true,
               pseudo_weight=cfg['st']['weight'])[0]


def train_experts(graph, ctx, experts, cfg, seed, folds, pseudo_idx=None, pseudo_labels=None):
    """Returns (oof [K, n_train, C] out-of-fold distributions on training nodes,
                full [K, N, C] distributions from experts trained on all training labels)."""
    y, train = ctx['y'], ctx['split']['train']
    if pseudo_idx is None:
        pseudo_idx = train.new_empty(0)
        pseudo_labels = y.new_empty(0)
    fold = stratified_folds(y, train, folds, seed)
    k, n_tr, c = len(experts), train.numel(), graph.num_classes
    oof = torch.zeros(k, n_tr, c, device=y.device)
    full = []
    for j, e in enumerate(experts):
        for f in range(folds):
            tr = train[fold != f]
            idx = torch.cat([tr, pseudo_idx])
            lab = torch.cat([y[tr], pseudo_labels])
            p = expert_probs(graph, ctx, e, cfg, idx, lab, seed + 100 * (f + 1), n_true=tr.numel())
            oof[j, fold == f] = p[train[fold == f]]
        idx = torch.cat([train, pseudo_idx])
        lab = torch.cat([y[train], pseudo_labels])
        full.append(expert_probs(graph, ctx, e, cfg, idx, lab, seed, n_true=train.numel()))
    return oof, torch.stack(full)


def coe_loss(logits, y, gamma, alpha=100.0):
    """Cross-entropy + CoE large margin: true-class probability minus a smooth maximum
    (log-sum-exp with temperature 1/alpha) over the other classes' probabilities."""
    prob = F.softmax(logits, 1)
    true = prob.gather(1, y.view(-1, 1)).view(-1)
    others = prob.scatter(1, y.view(-1, 1), float('-inf'))
    smooth_max = torch.logsumexp(alpha * others, 1) / alpha
    return F.cross_entropy(logits, y) - gamma * (true - smooth_max).mean()


def cooperate(P, Theta):
    """P: [K, n, C] expert distributions, Theta: [K, C, C] -> logits [n, C]."""
    return torch.einsum('knc,kdc->nd', P, Theta)


def fit_theta(oof, y_tr, acc, gamma, reg, epochs=500, lr=0.01, scale=10.0):
    """Theta initialised as accuracy-weighted soft voting (Theta_j = scale * acc_j * I), then trained on
    out-of-fold predictions; reg pulls Theta back towards that initialisation."""
    k, _, c = oof.shape
    theta0 = scale * acc.view(k, 1, 1) * torch.eye(c, device=oof.device).expand(k, c, c)
    theta = theta0.clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = coe_loss(cooperate(oof, theta), y_tr, gamma) + reg * (theta - theta0).pow(2).sum()
        loss.backward()
        opt.step()
    return theta.detach(), theta0


def combine(graph, ctx, oof, full, grid):
    """Fit Theta for each (gamma, reg) in the grid, pick on validation accuracy.
    Also returns the soft-voting references."""
    y, split = ctx['y'], ctx['split']
    y_tr = y[split['train']]
    acc = (oof.argmax(2) == y_tr.view(1, -1)).float().mean(1)  # out-of-fold accuracy per expert
    val_acc = lambda logits: (logits.argmax(1)[split['valid']] == y[split['valid']]).float().mean().item()
    res = {'expert_oof_acc': acc.tolist()}
    res['mean_vote'] = full.mean(0)
    res['acc_vote'] = (acc.view(-1, 1, 1) * full).sum(0)
    best = None
    for gamma, reg in itertools.product(grid['gamma'], grid['reg']):
        theta, _ = fit_theta(oof, y_tr, acc, gamma, reg, grid.get('epochs', 500), grid.get('lr', 0.01))
        logits = cooperate(full, theta)
        v = val_acc(logits)
        if best is None or v > best[0]:
            best = (v, logits, gamma, reg)
    res['coe'] = F.softmax(best[1], 1)
    res['coe_gamma'], res['coe_reg'] = best[2], best[3]
    # Same grid with gamma = 0 only: plain learned stacking, isolates the large-margin term.
    best0 = None
    for reg in grid['reg']:
        theta, _ = fit_theta(oof, y_tr, acc, 0.0, reg, grid.get('epochs', 500), grid.get('lr', 0.01))
        logits = cooperate(full, theta)
        v = val_acc(logits)
        if best0 is None or v > best0[0]:
            best0 = (v, logits)
    res['stack'] = F.softmax(best0[1], 1)
    return res
