import torch
import math
import random
import numpy as np
import torch.nn as nn
from scipy.spatial.distance import cdist
from model.dataset import gep_collate_fn
from model.model import MAGEPModel
from torch.utils.data import DataLoader, random_split, Subset
from model.args import get_args

def coral_loss(source_feat, target_feat):
    d = source_feat.size(1)
    if source_feat.shape[0] < 2:
        source_cov = torch.eye(d, device=source_feat.device) * 1e-4
    else:
        source_cov = torch.cov(source_feat.t()) + torch.eye(d, device=source_feat.device) * 1e-4

    if target_feat.shape[0] < 2:
        target_cov = torch.eye(d, device=target_feat.device) * 1e-4
    else:
        target_cov = torch.cov(target_feat.t()) + torch.eye(d, device=target_feat.device) * 1e-4

    loss = torch.norm(source_cov - target_cov, p='fro') ** 2 / (4 * d * d)
    return loss


def mmd_loss(source_feat, target_feat):
    if source_feat.size(0) == 0 or target_feat.size(0) == 0:
        return torch.tensor(0.0, device=source_feat.device)

    # Compute squared Euclidean distances
    xx = torch.mm(source_feat, source_feat.t())
    xy = torch.mm(source_feat, target_feat.t())
    yy = torch.mm(target_feat, target_feat.t())

    x_sqnorms = torch.diagonal(xx).unsqueeze(1)
    y_sqnorms = torch.diagonal(yy).unsqueeze(1)

    # Median heuristic for bandwidth
    pairwise_dist = torch.cdist(source_feat, target_feat)
    if pairwise_dist.numel() == 0:
        return torch.tensor(0.0, device=source_feat.device)
    median_dist = torch.median(pairwise_dist[pairwise_dist > 0]) if (pairwise_dist > 0).any() else torch.tensor(1.0, device=source_feat.device)
    gamma = 1.0 / (2 * median_dist ** 2 + 1e-8)

    # RBF kernel
    k_xx = torch.exp(-gamma * (-2 * xx + x_sqnorms + x_sqnorms.t()))
    k_xy = torch.exp(-gamma * (-2 * xy + x_sqnorms + y_sqnorms.t()))
    k_yy = torch.exp(-gamma * (-2 * yy + y_sqnorms + y_sqnorms.t()))

    mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    mmd = torch.clamp(mmd, min=0.0)
    return torch.sqrt(mmd + 1e-8)

def domain_acc(model, source_loader, target_loader, device, lambda_adv=1.0):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in source_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            _, _, d_pred, _ = model(batch['G_seq'], batch['E_stats'], batch['E_seq'], adv=True, lambda_adv=lambda_adv)
            pred = (torch.sigmoid(d_pred) > 0.5).float()
            correct += (pred == 0).sum().item()
            total += d_pred.size(0)

        for batch in target_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            _, _, d_pred, _ = model(batch['G_seq'], batch['E_stats'], batch['E_seq'], adv=True, lambda_adv=lambda_adv)
            pred = (torch.sigmoid(d_pred) > 0.5).float()
            correct += (pred == 1).sum().item()
            total += d_pred.size(0)

    acc = correct / total
    return acc

def build_task_loaders(dataset, locations, indices_in_dataset, batch_size, support_ratio=0.2):
    index_set = set(indices_in_dataset)
    task_loaders = []

    for loc in locations:
        loc_indices = [idx for idx, s in enumerate(dataset.samples)
                       if s['location'] == loc and idx in index_set]

        n_support = max(1, int(len(loc_indices) * support_ratio))
        n_query = len(loc_indices) - n_support

        if n_query < 1:
            support_idx = loc_indices
            query_idx = loc_indices
        else:
            rng = np.random.RandomState(get_args().seed)
            shuffled = loc_indices.copy()
            rng.shuffle(shuffled)
            support_idx = shuffled[:n_support]
            query_idx = shuffled[n_support:]

        support_subset = Subset(dataset, support_idx)
        query_subset = Subset(dataset, query_idx)

        support_loader = DataLoader(support_subset, batch_size=batch_size, shuffle=True,
                                    collate_fn=gep_collate_fn)
        query_loader = DataLoader(query_subset, batch_size=batch_size, shuffle=False,
                                  collate_fn=gep_collate_fn)

        task_loaders.append({
            'support': support_loader,
            'query': query_loader,
            'location': loc,
            'n_support': len(support_idx),
            'n_query': len(query_idx)
        })
    return task_loaders


def reptile_episode(meta_model, task_loaders, k_tasks, inner_lr, inner_updates,
                    meta_lr, env_stats_dim, mse_fn, use_query_for_outer=True):
    selected = random.sample(task_loaders, min(k_tasks, len(task_loaders)))
    task_deltas = []

    for task in selected:
        temp_model = MAGEPModel(env_stats_dim=env_stats_dim).to(get_args().device)
        temp_model.load_state_dict(meta_model.state_dict())
        temp_model.train()

        inner_opt = torch.optim.Adam(temp_model.parameters(), lr=inner_lr)
        support_iter = iter(task['support'])

        for _ in range(inner_updates):
            try:
                batch = next(support_iter)
            except StopIteration:
                support_iter = iter(task['support'])
                batch = next(support_iter)

            batch = {k: v.to(get_args().device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            inner_opt.zero_grad()
            h_pred, y_pred, _, _ = temp_model(batch['G_seq'], batch['E_stats'],
                                              batch['E_seq'], adv=False)
            loss = mse_fn(h_pred, batch['y_norm'][:, 0]) + mse_fn(y_pred, batch['y_norm'][:, 1])
            loss.backward()
            inner_opt.step()

        if use_query_for_outer:
            query_iter = iter(task['query'])
            for _ in range(max(1, inner_updates // 3)):
                try:
                    batch = next(query_iter)
                except StopIteration:
                    query_iter = iter(task['query'])
                    batch = next(query_iter)

                batch = {k: v.to(get_args().device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                inner_opt.zero_grad()
                h_pred, y_pred, _, _ = temp_model(batch['G_seq'], batch['E_stats'],
                                                  batch['E_seq'], adv=False)
                loss = mse_fn(h_pred, batch['y_norm'][:, 0]) + mse_fn(y_pred, batch['y_norm'][:, 1])
                loss.backward()
                inner_opt.step()

        delta = {}
        for name, param in meta_model.named_parameters():
            delta[name] = temp_model.state_dict()[name].data - param.data
        task_deltas.append(delta)

    with torch.no_grad():
        for name, param in meta_model.named_parameters():
            avg_update = torch.stack([d[name] for d in task_deltas]).mean(dim=0)
            param.data.add_(avg_update, alpha=meta_lr)