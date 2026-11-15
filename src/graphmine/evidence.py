"""Evidence-conditioned fusion of frozen node classifiers.

For diffusion weights h_ij from labelled sources j, the effective source count is
    n_eff(i) = (sum_j h_ij)^2 / sum_j h_ij^2.
It measures concentration of *source influence*, not statistical independence of
labels. A one-source component has n_eff=1 even if LP entropy is zero. We compute
the denominator exactly on citation graphs, or estimate it with seeded Rademacher
probes on large graphs. No dense N-by-N operator is constructed.

All fusion training inputs must be out-of-fold (including the diffusion evidence).
Validation labels select checkpoints; test labels are never read by this module.
"""
import copy
import math

import torch
import torch.nn.functional as F

from .prop import propagate


@torch.no_grad()
def diffusion_evidence(adj, y, sources, num_classes, alpha=0.9, iters=50,
                       probes=32, seed=0, exact_limit=256):
    n, count = y.numel(), sources.numel()
    exact = count <= exact_limit
    width = count if exact else probes
    initial = torch.zeros(n, num_classes + 1 + width, device=y.device)
    initial[sources, y[sources]] = 1
    # Total possible diffusion mass provides a degree-normalised coverage ratio.
    initial[:, num_classes] = 1
    if exact:
        initial[sources, num_classes + 1 + torch.arange(count, device=y.device)] = 1
    else:
        gen = torch.Generator(device=y.device).manual_seed(seed)
        signs = torch.randint(2, (count, width), generator=gen, device=y.device)
        initial[sources, num_classes + 1:] = signs.float() * 2 - 1
    out = propagate(adj, initial, alpha, iters)
    raw = out[:, :num_classes]
    mass = raw.sum(1)
    signal = mass > 1e-12
    q = torch.where(signal[:, None], raw / mass[:, None].clamp_min(1e-12),
                    torch.full_like(raw, 1 / num_classes))
    sketch = out[:, num_classes + 1:]
    squared = sketch.square().sum(1) if exact else sketch.square().mean(1)
    ess = (mass.square() / squared.clamp_min(1e-24)).clamp(1, max(count, 1))
    ess = torch.where(signal, ess, torch.zeros_like(ess))
    coverage = mass / out[:, num_classes].clamp_min(1e-12)
    # Raw statistics are retained for interpretable, label-free stratification.
    stats = torch.stack([coverage, ess, signal.float()], 1)
    return q, stats


@torch.no_grad()
def fusion_features(p, stats, degree, lp_index, evidence=True):
    """p [K,N,C], stats [N,3]. Features have shape [N,K,D]."""
    p = p.permute(1, 0, 2).clamp_min(1e-8)
    top = p.topk(2, dim=2).values
    entropy = -(p * p.log()).sum(2) / math.log(p.size(2))
    mean = p.mean(1, keepdim=True)
    mid = (p + mean) / 2
    js = 0.5 * ((p * (p / mid).log()).sum(2) +
                (mean * (mean / mid).log()).sum(2))
    pred = p.argmax(2)
    agreement = (pred[:, :, None] == pred[:, None, :]).float().mean(2)
    cols = [top[:, :, 0], top[:, :, 0] - top[:, :, 1], entropy, js, agreement]
    if evidence:
        lp = p[:, lp_index, :]
        lp_at_pred = lp.gather(1, pred)
        lp_pred = lp.argmax(1)
        at_lp = p.gather(2, lp_pred[:, None, None].expand(-1, p.size(1), 1)).squeeze(2)
        cols.extend([lp_at_pred, at_lp])
        context = torch.stack([torch.log1p(100 * stats[:, 0]), torch.log1p(stats[:, 1]),
                               stats[:, 2], torch.log1p(degree)], 1)
        cols.extend([v[:, None].expand(-1, p.size(1)) for v in context.unbind(1)])
    return torch.stack(cols, 2)


class FusionGate(torch.nn.Module):
    """Low-capacity per-expert linear router, shrunk towards a global mixture."""
    def __init__(self, experts, features, initial_bias, lp_index):
        super().__init__()
        self.bias = torch.nn.Parameter(initial_bias.clone())
        self.weight = torch.nn.Parameter(torch.zeros(experts, features, device=initial_bias.device))
        self.lp_index = lp_index

    def forward(self, p, x, signal):
        score = self.bias[None] + (x * self.weight[None]).sum(2)
        # Abstention is shared by the global and adaptive controls.
        score = score.clone()
        score[:, self.lp_index] = score[:, self.lp_index].masked_fill(~signal, -1e9)
        weights = score.softmax(1)
        probs = (p.permute(1, 0, 2) * weights[:, :, None]).sum(1)
        return probs, weights


def fit_gate(oof, y_train, train_stats, train_degree, val_p, y_valid,
             val_stats, val_degree, lp_index, mode, regs=(0.01, 0.1, 1.0),
             epochs=400, lr=0.03, batch_size=32768, seed=0):
    """Select only on validation accuracy (NLL breaks ties). No test inputs."""
    torch.manual_seed(seed)
    evidence = mode == 'evidence'
    x = fusion_features(oof, train_stats, train_degree, lp_index, evidence)
    xv = fusion_features(val_p, val_stats, val_degree, lp_index, evidence)
    if mode == 'global':
        x, xv = x[:, :, :0], xv[:, :, :0]
    center = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(0.1)
    x, xv = ((x - center) / scale).clamp(-5, 5), ((xv - center) / scale).clamp(-5, 5)
    acc = (oof.argmax(2) == y_train[None]).float().mean(1)
    initial = acc.clamp_min(1e-4).log()
    best = None
    for reg in regs:
        gate = FusionGate(oof.size(0), x.size(2), initial, lp_index)
        opt = torch.optim.Adam(gate.parameters(), lr=lr)
        # Reset shuffling for paired hyperparameter comparisons.
        gen = torch.Generator(device=oof.device).manual_seed(seed)
        for epoch in range(epochs + 1):
            if epoch % 20 == 0 or epoch == epochs:
                with torch.no_grad():
                    pv, _ = gate(val_p, xv, val_stats[:, 2].bool())
                    va = (pv.argmax(1) == y_valid).float().mean().item()
                    nll = F.nll_loss(pv.clamp_min(1e-8).log(), y_valid).item()
                    rank = (va, -nll)
                    if best is None or rank > best['rank']:
                        best = {'rank': rank, 'reg': reg, 'epoch': epoch,
                                'state': copy.deepcopy(gate.state_dict())}
            if epoch == epochs:
                break
            order = torch.randperm(y_train.numel(), generator=gen, device=oof.device)
            for ids in order.split(batch_size):
                opt.zero_grad()
                pr, _ = gate(oof[:, ids], x[ids], train_stats[ids, 2].bool())
                loss = F.nll_loss(pr.clamp_min(1e-8).log(), y_train[ids])
                # Global weights get only a weak pull; local departures are regularised.
                loss = loss + reg * gate.weight.square().sum() + 1e-3 * (gate.bias - initial).square().sum()
                loss.backward()
                opt.step()
    gate.load_state_dict(best.pop('state'))
    return gate.eval(), center, scale, best


@torch.no_grad()
def apply_gate(gate, full, stats, degree, center, scale, mode, device, chunk=100000):
    """Chunked inference; full expert tensors may stay on CPU on large graphs."""
    probs, weights = [], []
    for start in range(0, full.size(1), chunk):
        sl = slice(start, start + chunk)
        p, s, d = full[:, sl].to(device), stats[sl].to(device), degree[sl].to(device)
        x = fusion_features(p, s, d, gate.lp_index, mode == 'evidence')
        if mode == 'global':
            x = x[:, :, :0]
        x = ((x - center) / scale).clamp(-5, 5)
        pr, w = gate(p, x, s[:, 2].bool())
        probs.append(pr.cpu())
        weights.append(w.cpu())
    return torch.cat(probs), torch.cat(weights)
