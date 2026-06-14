import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import random
import copy
import matplotlib.pyplot as plt
from model.utils import *
from model.dataset import GEPDataset, gep_collate_fn
from model.model import MAGEPModel
from model.evalution import *
from model.args import init_args, apply_seed

args = init_args()
apply_seed()
DEVICE = torch.device(args.device)

if __name__ == '__main__':
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.pic_dir, exist_ok=True)

    source_locations = args.source_locations.split(',')
    target_locations = args.target_locations.split(',')

    source_full_dataset = GEPDataset(geno_dir=args.geno_dir, env_dir=args.env_dir, pheno_dir=args.pheno_dir,
                                     locations=source_locations,
                                     domain_label=0, is_train=True)
    y_scaler = source_full_dataset.y_scaler
    env_seq_scaler = source_full_dataset.env_seq_scaler
    env_stats_scaler = source_full_dataset.env_stats_scaler
    env_stats_names = source_full_dataset.env_stats_names

    target_full_dataset = GEPDataset(geno_dir=args.geno_dir, env_dir=args.env_dir, pheno_dir=args.pheno_dir,
                                     locations=target_locations,
                                     domain_label=1, y_scaler=y_scaler,
                                     env_seq_scaler=env_seq_scaler,
                                     env_stats_scaler=env_stats_scaler, is_train=False)

    target_support_size = int(args.support_ratio * len(target_full_dataset))
    target_query_size = len(target_full_dataset) - target_support_size
    target_support_dataset, target_query_dataset = random_split(
        target_full_dataset, [target_support_size, target_query_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    target_query_loader = DataLoader(target_query_dataset, batch_size=args.batch_size, shuffle=False,
                                     collate_fn=gep_collate_fn)

    sample = source_full_dataset[0]
    num_markers = sample['G_seq'].shape[0]
    env_stats_dim = sample['E_stats'].shape[0]

    print(f"\nSource domain samples：{len(source_full_dataset)}")
    print(f"Target domain support set ({int(args.support_ratio*100)}%): {len(target_support_dataset)}")
    print(f"Target domain query set ({int((1-args.support_ratio)*100)}%): {len(target_query_dataset)}")

    print("\n=== source domain meta learning task  ===")
    source_all_indices = list(range(len(source_full_dataset)))
    source_task_loaders = build_task_loaders(
        dataset=source_full_dataset,
        locations=source_locations,
        indices_in_dataset=source_all_indices,
        batch_size=args.batch_size,
        support_ratio=args.support_ratio,
    )
    print(f" Total of {len(source_task_loaders)} source domain meta learning task")

    # ## Source domain Reptile-based Meta-Training
    # print("\n" + "=" * 60)
    # print("Source Reptile Meta-Training")
    # print("=" * 60)
    #
    #
    # source_reptile_model = MAGEPModel(env_stats_dim=env_stats_dim).to(DEVICE)
    # mse = nn.MSELoss()
    #
    # print(f"source domain task：{len(source_task_loaders)} ")
    # pbar = tqdm(range(args.episodes), desc="Stage1 Source Reptile-based meta-learning", ncols=100)
    # for episode in pbar:
    #     reptile_episode(
    #         meta_model=source_reptile_model,
    #         task_loaders=source_task_loaders,
    #         k_tasks=args.k_tasks,
    #         inner_lr=args.inner_lr,
    #         inner_updates=args.inner_updates,
    #         meta_lr=args.meta_lr,
    #         env_stats_dim=env_stats_dim,
    #         mse_fn=mse,
    #         use_query_for_outer=True
    #     )
    #     if (episode + 1) % 20 == 0:
    #         pbar.set_postfix_str(f"Ep {episode + 1}/{args.episodes}")
    #
    # torch.save(source_reptile_model.state_dict(), f'{args.save_dir}/source_reptile_pretrain.pth')
    # print("Source domain Reptile pre training parameters saved\n")
    #
    #
    # ## Fine-tuning the target domain support set from pre-trained parameters in the source domain
    # print("\n" + "=" * 60)
    # print("Fine-tuning the target domain support set ")
    # print("=" * 60)
    #
    # target_reptile_model = MAGEPModel(env_stats_dim=env_stats_dim).to(DEVICE)
    # target_reptile_model.load_state_dict(
    #     torch.load(f'{args.save_dir}/source_reptile_pretrain.pth', map_location=DEVICE, weights_only=True)
    # )
    #
    # target_support_task_loaders = build_task_loaders(
    #     dataset=target_full_dataset,
    #     locations=target_locations,
    #     indices_in_dataset=target_support_dataset.indices,
    #     batch_size=args.batch_size,
    #     support_ratio=0.2,
    # )
    #
    # mse = nn.MSELoss()
    # pbar = tqdm(range(200), desc="Stage2 Target domain", ncols=100)
    # for episode in pbar:
    #     reptile_episode(
    #         meta_model=target_reptile_model,
    #         task_loaders=target_support_task_loaders,
    #         k_tasks=4,
    #         inner_lr=5e-4,
    #         inner_updates=20,
    #         meta_lr=0.0005,
    #         env_stats_dim=env_stats_dim,
    #         mse_fn=mse,
    #         use_query_for_outer=True
    #     )
    #     if (episode + 1) % 20 == 0:
    #         pbar.set_postfix_str(f"Ep {episode + 1}/{200}")
    #
    # source_da_loader = DataLoader(
    #     source_full_dataset, batch_size=args.batch_size, shuffle=True,
    #     collate_fn=gep_collate_fn
    # )
    #
    # target_support_indices = []
    # target_query_indices = []
    # for task in target_support_task_loaders:
    #     target_support_indices.extend(task['support'].dataset.indices)
    #     target_query_indices.extend(task['query'].dataset.indices)
    #
    # support_set = set(target_support_indices)
    # target_query_indices = [idx for idx in target_query_indices
    #                         if idx not in support_set]
    #
    # target_support_subset = Subset(target_full_dataset, target_support_indices)
    # target_query_subset = Subset(target_full_dataset, target_query_indices)
    #
    # da_target_train_loader = DataLoader(target_support_subset, batch_size=args.batch_size, shuffle=True,
    #                                     collate_fn=gep_collate_fn)
    # da_val_loader = DataLoader(target_query_subset, batch_size=args.batch_size, shuffle=False,
    #                            collate_fn=gep_collate_fn)
    #
    # bce_fn = nn.BCEWithLogitsLoss()
    #
    # da_model = MAGEPModel(env_stats_dim=env_stats_dim).to(DEVICE)
    # da_model.load_state_dict(target_reptile_model.state_dict())
    #
    # da_optimizer = torch.optim.AdamW(da_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # da_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     da_optimizer, 'min', factor=args.scheduler_factor, patience=args.scheduler_patience
    # )
    #
    # best_val_loss = float('inf')
    # patience_counter = 0
    # mse = nn.MSELoss()
    # steps_per_epoch = len(da_target_train_loader)
    #
    # for epoch in range(1, args.epochs + 1):
    #     p = epoch / args.epochs
    #     lambda_adv = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
    #
    #     da_model.train()
    #     source_iter = iter(source_da_loader)
    #     target_iter = iter(da_target_train_loader)
    #
    #     epoch_total_loss = 0.0
    #     epoch_task_loss = 0.0
    #     epoch_domain_loss = 0.0
    #     epoch_coral_loss = 0.0
    #     n_batches = 0
    #
    #     for step in range(steps_per_epoch):
    #         try:
    #             batch_s = next(source_iter)
    #         except StopIteration:
    #             source_iter = iter(source_da_loader)
    #             batch_s = next(source_iter)
    #
    #         try:
    #             batch_t = next(target_iter)
    #         except StopIteration:
    #             target_iter = iter(da_target_train_loader)
    #             batch_t = next(target_iter)
    #
    #         batch_s = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
    #                    for k, v in batch_s.items()}
    #         batch_t = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
    #                    for k, v in batch_t.items()}
    #
    #         da_optimizer.zero_grad()
    #
    #         h_pred_s, y_pred_s, domain_pred_s, fused_s = da_model(
    #             batch_s['G_seq'], batch_s['E_stats'], batch_s['E_seq'],
    #             adv=True, lambda_adv=lambda_adv
    #         )
    #         loss_task_s = (mse(h_pred_s, batch_s['y_norm'][:, 0]) +
    #                        mse(y_pred_s, batch_s['y_norm'][:, 1]))
    #         domain_label_s = torch.zeros(fused_s.size(0), 1, device=DEVICE)
    #         loss_domain_s = bce_fn(domain_pred_s.unsqueeze(-1), domain_label_s)
    #
    #         h_pred_t, y_pred_t, domain_pred_t, fused_t = da_model(
    #             batch_t['G_seq'], batch_t['E_stats'], batch_t['E_seq'],
    #             adv=True, lambda_adv=lambda_adv
    #         )
    #         loss_task_t = (mse(h_pred_t, batch_t['y_norm'][:, 0]) +
    #                        mse(y_pred_t, batch_t['y_norm'][:, 1]))
    #         domain_label_t = torch.ones(fused_t.size(0), 1, device=DEVICE)
    #         loss_domain_t = bce_fn(domain_pred_t.unsqueeze(-1), domain_label_t)
    #
    #         loss_coral_val = coral_loss(fused_s, fused_t)
    #
    #         total_loss = (loss_task_s + loss_task_t + (loss_domain_s + loss_domain_t) + 5.0 * loss_coral_val)
    #
    #         total_loss.backward()
    #         torch.nn.utils.clip_grad_norm_(da_model.parameters(), max_norm=5.0)
    #         da_optimizer.step()
    #
    #         epoch_total_loss += total_loss.item()
    #         epoch_task_loss += (loss_task_s + loss_task_t).item()
    #         epoch_domain_loss += (loss_domain_s + loss_domain_t).item()
    #         epoch_coral_loss += loss_coral_val.item()
    #         n_batches += 1
    #
    #     da_model.eval()
    #     val_loss = 0.0
    #     val_samples = 0
    #     with torch.no_grad():
    #         for batch in da_val_loader:
    #             batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
    #                      for k, v in batch.items()}
    #             h_pred, y_pred, _, _ = da_model(batch['G_seq'], batch['E_stats'],
    #                                             batch['E_seq'], adv=False)
    #             val_loss += (mse(h_pred, batch['y_norm'][:, 0]) +
    #                          mse(y_pred, batch['y_norm'][:, 1])).item() * batch['G_seq'].size(0)
    #             val_samples += batch['G_seq'].size(0)
    #
    #     avg_val_loss = val_loss / val_samples if val_samples > 0 else 0.0
    #     da_scheduler.step(avg_val_loss)
    #
    #     avg_total = epoch_total_loss / max(n_batches, 1)
    #     avg_task = epoch_task_loss / max(n_batches, 1)
    #     avg_dom = epoch_domain_loss / max(n_batches, 1)
    #     avg_coral = epoch_coral_loss / max(n_batches, 1)
    #
    #     print(f"  Ep {epoch:3d}/{args.epochs} | λ_adv={lambda_adv:.3f} | "
    #           f"Task {avg_task:.4f} | Domain {avg_dom:.4f} | "
    #           f"MMD {avg_coral:.4f} | Val {avg_val_loss:.4f}")
    #
    #     if avg_val_loss < best_val_loss:
    #         best_val_loss = avg_val_loss
    #         torch.save(da_model.state_dict(), f'{args.save_dir}/best_da_model.pth')
    #         patience_counter = 0
    #     else:
    #         patience_counter += 1
    #         if patience_counter >= args.early_stop_patience:
    #             print(f"  >>> Early stopping at epoch {epoch}")
    #             break
    #
    # da_model.load_state_dict(torch.load(f'{args.save_dir}/best_da_model.pth', weights_only=True))
    # print(f"Save the best fine-tuning model（val={best_val_loss:.4f}）\n")

    # ===================================================
    #  Final evaluation on target domain query set (80%)
    # ===================================================
    print(f"target domain query set: {len(target_query_dataset)} samples")
    da_model = MAGEPModel(env_stats_dim=env_stats_dim).to(DEVICE)
    da_model.load_state_dict(
        torch.load(f'{args.save_dir}/best_da_model.pth', map_location=DEVICE, weights_only=True)
    )
    evaluate(da_model, target_query_loader, "Result",
             f"{args.pic_dir}", y_scaler, env_stats_names,
             num_markers, env_stats_dim)
