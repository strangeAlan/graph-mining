"""Training loop, self-training with label-propagation agreement, and C&S post-processing."""
import copy
import itertools
import time

import torch
import torch.nn.functional as F

from .data import accuracy
from .models import FULL_GRAPH_MODELS, build_model
from .prop import correct_and_smooth


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _acc(pred, y, idx):
    return (pred[idx] == y[idx]).float().mean().item()


@torch.no_grad()
def predict(model, model_name, n, chunk=200_000):
    model.eval()
    all_idx = torch.arange(n, device=next(model.parameters()).device)
    if model_name in FULL_GRAPH_MODELS:
        return F.softmax(model(all_idx), 1)
    return torch.cat([F.softmax(model(all_idx[i:i + chunk]), 1) for i in range(0, n, chunk)])


def fit(graph, ctx, model_name, cfg, idx, labels, seed, n_true=None, pseudo_weight=1.0):
    """Train a fresh model on (idx, labels); early-stop on validation accuracy. The first n_true
    entries are ground-truth labels, the rest pseudo-labels: the two losses are averaged separately,
    loss = CE_true + pseudo_weight * CE_pseudo, so the pseudo-label count does not swamp the true labels.
    Returns softmax probabilities of all nodes from the best-validation epoch."""
    n_true = idx.numel() if n_true is None else n_true
    set_seed(seed)
    y, valid = ctx['y'], ctx['split']['valid']
    model = build_model(model_name, ctx, cfg, graph.num_classes).to(y.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    bs = cfg.get('batch_size') if model_name not in FULL_GRAPH_MODELS else None
    best_val, best_probs, bad = -1.0, None, 0
    for epoch in range(cfg['epochs']):
        model.train()
        order = torch.randperm(idx.numel(), device=idx.device)
        for b in ([order] if bs is None else order.split(bs)):
            opt.zero_grad()
            ce = F.cross_entropy(model(idx[b]), labels[b], reduction='none')
            true = b < n_true
            loss = ce[true].mean() if true.any() else ce.new_zeros(())
            if (~true).any():
                loss = loss + pseudo_weight * ce[~true].mean()
            loss.backward()
            opt.step()
        if (epoch + 1) % cfg.get('eval_every', 1):
            continue
        probs = predict(model, model_name, graph.num_nodes)
        val = _acc(probs.argmax(1), y, valid)
        if val > best_val:
            best_val, best_probs, bad = val, probs, 0
        else:
            bad += 1
            if bad >= cfg['patience']:
                break
    return best_probs, best_val


def cs_search(graph, ctx, probs, grid):
    """Pick the C&S propagation strengths on validation accuracy (training labels only are propagated)."""
    y, split = ctx['y'], ctx['split']
    best = None
    for a1, a2 in itertools.product(grid['alpha1'], grid['alpha2']):
        out = correct_and_smooth(ctx['adj'], probs, y, split['train'], a1, a2, grid.get('iters', 50))
        pred = out.argmax(1)
        val = _acc(pred, y, split['valid'])
        if best is None or val > best[0]:
            best = (val, pred, a1, a2)
    val, pred, a1, a2 = best
    return {'cs_val': val, 'cs_test': accuracy(graph, pred, split['test']), 'cs_alpha1': a1, 'cs_alpha2': a2}


def run_seed(graph, ctx, model_name, method, cfg, seed):
    """One seed of: base model, self-training with confidence only ('st'), or self-training with
    confidence + agreement with label propagation ('lpst'). C&S is reported on top of each."""
    t0 = time.time()
    y, split = ctx['y'], ctx['split']
    train = split['train']
    probs, val = fit(graph, ctx, model_name, cfg, train, y[train], seed)
    stages = [{'stage': 0, 'val': val, 'test': accuracy(graph, probs.argmax(1), split['test']), 'n_pseudo': 0}]
    best_probs, best_val = probs, val

    if method in ('st', 'lpst'):
        st = cfg['st']
        candidate = torch.ones(graph.num_nodes, dtype=torch.bool, device=y.device)
        candidate[train] = False
        candidate[split['valid']] = False  # validation stays clean for model selection
        lp_pred = ctx['lp'].argmax(1)
        lp_signal = ctx['lp_signal']
        for s in range(1, st['stages'] + 1):
            conf, pred = probs.max(1)
            sel = candidate & (conf >= st['tau'])
            if method == 'lpst':
                sel &= (pred == lp_pred) | ~lp_signal  # LP abstains where no label mass arrives
            pidx = sel.nonzero().view(-1)
            idx = torch.cat([train, pidx])
            labels = torch.cat([y[train], pred[pidx]])
            probs, val = fit(graph, ctx, model_name, cfg, idx, labels, seed + 1000 * s,
                             n_true=train.numel(), pseudo_weight=st['weight'])
            stages.append({'stage': s, 'val': val, 'test': accuracy(graph, probs.argmax(1), split['test']),
                           'n_pseudo': pidx.numel(),
                           # diagnostic only: pseudo-label precision, never used for any decision
                           'pseudo_precision': (pred[pidx] == y[pidx]).float().mean().item() if pidx.numel() else None})
            if val > best_val:
                best_probs, best_val = probs, val

    chosen = max(stages, key=lambda r: r['val'])
    out = {'seed': seed, 'val': chosen['val'], 'test': chosen['test'], 'chosen_stage': chosen['stage'],
           'stages': stages}
    out.update(cs_search(graph, ctx, best_probs, cfg['cs_grid']))
    out['time_s'] = time.time() - t0
    return out
