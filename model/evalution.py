# visualization.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from umap import UMAP


DISTINCT_COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
DISTINCT_MARKERS = ['o','v','^','<','>','s']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

def plot_geno_importance(imp_h, imp_y, num_markers, name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    global_max = max(imp_h.max(), imp_y.max()) + 1e-8
    for imp, trait in zip([imp_h, imp_y], ['height', 'yield']):
        norm = imp / global_max
        plt.figure(figsize=(12,2))
        plt.imshow(norm[None,:], aspect='auto', cmap='hot')
        plt.title(f'{name} - Geno Importance ({trait.capitalize()})')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{name}_geno_{trait}.png'), dpi=150)
        plt.close()


def plot_env_stats_importance(imp_h, imp_y, names, name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for imp, trait in zip([imp_h, imp_y], ['height', 'yield']):
        norm = imp / (imp.max() + 1e-8)
        plt.figure(figsize=(12,6))
        plt.bar(range(len(names)), norm)
        plt.xticks(range(len(names)), names, rotation=90)
        plt.title(f'{name} - Env Stats Importance ({trait.capitalize()})')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{name}_envstats_{trait}.png'))
        plt.close()


def plot_env_seq_importance(imp_h, imp_y, channel_names, name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for imp, trait in zip([imp_h, imp_y], ['height', 'yield']):
        plt.figure(figsize=(12,6))
        for c in range(imp.shape[1]):
            norm = imp[:,c] / (imp[:,c].max() + 1e-8)
            plt.plot(norm, label=channel_names[c])
        plt.legend()
        plt.title(f'{name} - Env Seq Importance ({trait.capitalize()})')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{name}_envseq_{trait}.png'))
        plt.close()

def collect_fused_features(model, loaders, device):
    model.eval()
    feats = []
    domain_ids = []
    locations = []

    def hook(module, input, output):
        feats.append(output.detach().cpu().numpy())

    handle = model.fusion.register_forward_hook(hook)

    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                model(batch['G_seq'], batch['E_stats'], batch['E_seq'], adv=False)
                domain_ids.extend(batch['domain_id'].cpu().numpy().tolist())
                locations.extend(batch['location'])

    handle.remove()

    all_feats = np.concatenate(feats, axis=0) if feats else np.array([])
    domain_ids = np.array(domain_ids)
    locations = np.array(locations)

    if len(all_feats) > 2000:
        idx = np.random.choice(len(all_feats), 2000, replace=False)
        all_feats = all_feats[idx]
        domain_ids = domain_ids[idx]
        locations = np.array(locations)[idx]

    return all_feats, domain_ids, locations


def plot_interactive_tsne_3d(features, labels, title):
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    tsne = TSNE(n_components=3, random_state=SEED, perplexity=30, n_iter=1000)
    features_3d = tsne.fit_transform(features)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))

    for label in unique_labels:
        mask = (labels == label)
        ax.scatter(
            features_3d[mask, 0],
            features_3d[mask, 1],
            features_3d[mask, 2],
            label=str(label),
            color=label_to_color[label],
            alpha=0.7,
            s=20
        )

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
    ax.set_zlabel('t-SNE Dimension 3', fontsize=10)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.show()

def evaluate(model, loader, name, save_dir, y_scaler, env_stats_names, num_markers, env_stats_dim):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    preds = []
    true_h, pred_h = [], []
    true_y, pred_y = [], []
    g_imp_h = np.zeros(num_markers)
    g_imp_y = np.zeros(num_markers)
    stats_imp_h = np.zeros(env_stats_dim)
    stats_imp_y = np.zeros(env_stats_dim)
    seq_imp_h = np.zeros((1800, 4))
    seq_imp_y = np.zeros((1800, 4))
    total_n = 0
    channel_names = ['Temperature', 'RelativeHumidity', 'SolarRadiation', 'Rainfall']

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            h_pred, y_pred, _, _ = model(batch['G_seq'], batch['E_stats'], batch['E_seq'], adv=False)
            pred_norm = torch.stack([h_pred, y_pred], dim=1).cpu().numpy()
            pred_real = y_scaler.inverse_transform(pred_norm)
            y_raw = batch['y_raw'].cpu().numpy()

            for i in range(batch['G_seq'].size(0)):
                preds.append({
                    'Location': batch['location'][i],
                    'Sample': batch['sample'][i],
                    'Height_True': y_raw[i][0],
                    'Height_Pred': pred_real[i][0],
                    'Yield_True': y_raw[i][1],
                    'Yield_Pred': pred_real[i][1]
                })
                true_h.append(y_raw[i][0])
                pred_h.append(pred_real[i][0])
                true_y.append(y_raw[i][1])
                pred_y.append(pred_real[i][1])

    pd.DataFrame(preds).to_csv(os.path.join(save_dir, f'{name}_preds.csv'), index=False)

    metrics = []
    for trait, true, pred, unit in zip(['Height', 'Yield'], [true_h, true_y], [pred_h, pred_y], ['cm', 'bu/A']):
        rmse = np.sqrt(mean_squared_error(true, pred))
        mae = mean_absolute_error(true, pred)
        pcc = pearsonr(true, pred)[0] if len(true) > 1 else np.nan
        print(f"{trait} ({name}): RMSE {rmse:.2f} | MAE {mae:.2f} | PCC {pcc:.3f}")
        metrics.append({'Trait': f'{trait} [{unit}]', 'RMSE': rmse, 'MAE': mae, 'PCC': pcc})

    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)

    was_training = model.training
    model.train()
    for batch in loader:
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        G = batch['G_seq']
        E_stats = batch['E_stats']
        E_seq = batch['E_seq']

        G.requires_grad_(True)
        h_pred, y_pred, _, _ = model(G, E_stats, E_seq, adv=False)
        grad_h_g = torch.autograd.grad(h_pred.sum(), G, retain_graph=True)[0]
        grad_y_g = torch.autograd.grad(y_pred.sum(), G)[0]
        g_imp_h += grad_h_g.abs().mean(dim=0).cpu().numpy()
        g_imp_y += grad_y_g.abs().mean(dim=0).cpu().numpy()

        E_stats.requires_grad_(True)
        h_pred, y_pred, _, _ = model(G, E_stats, E_seq, adv=False)
        grad_h_s = torch.autograd.grad(h_pred.sum(), E_stats, retain_graph=True,
                                       allow_unused=True)[0]
        grad_y_s = torch.autograd.grad(y_pred.sum(), E_stats,
                                       allow_unused=True)[0]
        if grad_h_s is not None:
            stats_imp_h += grad_h_s.abs().mean(dim=0).cpu().numpy()
        if grad_y_s is not None:
            stats_imp_y += grad_y_s.abs().mean(dim=0).cpu().numpy()

        E_seq.requires_grad_(True)
        h_pred, y_pred, _, _ = model(G, E_stats, E_seq, adv=False)
        grad_h_seq = torch.autograd.grad(h_pred.sum(), E_seq, retain_graph=True,
                                         allow_unused=True)[0]
        grad_y_seq = torch.autograd.grad(y_pred.sum(), E_seq,
                                         allow_unused=True)[0]
        if grad_h_seq is not None:
            seq_imp_h += grad_h_seq.abs().mean(dim=0).cpu().numpy()
        if grad_y_seq is not None:
            seq_imp_y += grad_y_seq.abs().mean(dim=0).cpu().numpy()

        total_n += G.size(0)

    if not was_training:
        model.eval()

    g_imp_h /= total_n
    g_imp_y /= total_n
    stats_imp_h /= total_n
    stats_imp_y /= total_n
    seq_imp_h /= total_n
    seq_imp_y /= total_n

def plot_shap_dependence(model, loader, name, save_dir, y_scaler, env_stats_names):
    import shap
    from scipy.interpolate import UnivariateSpline
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    model = model.float()

    background_G = []
    background_E_stats = []
    background_E_seq = []
    sample_limit = 80
    collected = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            bs = batch['G_seq'].shape[0]
            take = min(bs, sample_limit - collected)
            if take > 0:
                background_G.append(batch['G_seq'][:take].cpu())
                background_E_stats.append(batch['E_stats'][:take].cpu())
                background_E_seq.append(batch['E_seq'][:take].cpu())
                collected += take
            if collected >= sample_limit:
                break

    bg_G = torch.cat(background_G, dim=0).to(DEVICE)
    bg_E_stats = torch.cat(background_E_stats, dim=0).to(DEVICE)
    bg_E_seq = torch.cat(background_E_seq, dim=0).to(DEVICE)

    feat_idx_height = 21
    factor_name_height = env_stats_names[feat_idx_height] if feat_idx_height < len(env_stats_names) else 'max_Rainfall'
    feat_idx_yield = 21
    factor_name_yield = env_stats_names[feat_idx_yield] if feat_idx_yield < len(env_stats_names) else 'mean_RelativeHumidity'


    class HeightWrapper(torch.nn.Module):
        def forward(self, G_seq, E_stats, E_seq):
            h_pred, _, _, _ = model(G_seq, E_stats, E_seq, adv=False)
            return h_pred.unsqueeze(1)

    class YieldWrapper(torch.nn.Module):
        def forward(self, G_seq, E_stats, E_seq):
            _, y_pred, _, _ = model(G_seq, E_stats, E_seq, adv=False)
            return y_pred.unsqueeze(1)


    explainer_h = shap.GradientExplainer(HeightWrapper().to(DEVICE), [bg_G, bg_E_stats, bg_E_seq])
    shap_values_h = explainer_h.shap_values([bg_G, bg_E_stats, bg_E_seq])

    explainer_y = shap.GradientExplainer(YieldWrapper().to(DEVICE), [bg_G, bg_E_stats, bg_E_seq])
    shap_values_y = explainer_y.shap_values([bg_G, bg_E_stats, bg_E_seq])


    shap_h = shap_values_h[1][:, :, 0]          # (samples, features)
    shap_y = shap_values_y[1][:, :, 0]
    X = bg_E_stats.cpu().numpy()                # 归一化后的特征值


    feat_val_h = X[:, feat_idx_height]
    shap_val_h = shap_h[:, feat_idx_height]

    plt.figure(figsize=(6, 6))
    plt.scatter(feat_val_h, shap_val_h, color='#1976D2', alpha=0.65, s=12, edgecolors='none', label='SHAP values')

    if len(feat_val_h) > 10:
        sort_idx = np.argsort(feat_val_h)
        x_s = feat_val_h[sort_idx]
        y_s = shap_val_h[sort_idx]
        spline_h = UnivariateSpline(x_s, y_s, s=0.5, k=3)
        x_smooth = np.linspace(feat_val_h.min(), feat_val_h.max(), 200)
        y_smooth = spline_h(x_smooth)
        plt.plot(x_smooth, y_smooth, color='#D32F2F', linewidth=3.5, label='GAM Spline Fit')

    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.6)
    plt.xlabel(f'Normalized {factor_name_height}', fontsize=13)
    plt.ylabel('SHAP Value (cm)', fontsize=13)
    plt.title(f'{name} - SHAP Dependence + GAM Fit\n{factor_name_height} on Height', fontsize=14, pad=15)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{name}_shap_height_max_rainfall_1.png'), dpi=200)
    plt.close()

    feat_val_y = X[:, feat_idx_yield]
    shap_val_y = shap_y[:, feat_idx_yield]

    plt.figure(figsize=(6, 6))
    plt.scatter(feat_val_y, shap_val_y, color='#D32F2F', alpha=0.65, s=12, edgecolors='none', label='SHAP values')

    if len(feat_val_y) > 10:
        sort_idx = np.argsort(feat_val_y)
        x_s = feat_val_y[sort_idx]
        y_s = shap_val_y[sort_idx]
        spline_y = UnivariateSpline(x_s, y_s, s=0.5, k=3)
        x_smooth = np.linspace(feat_val_y.min(), feat_val_y.max(), 200)
        y_smooth = spline_y(x_smooth)
        plt.plot(x_smooth, y_smooth, color='#1976D2', linewidth=3.5, label='GAM Spline Fit')

    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.6)
    plt.xlabel(f'Normalized {factor_name_yield}', fontsize=13)
    plt.ylabel('SHAP Value (bu/A)', fontsize=13)
    plt.title(f'{name} - SHAP Dependence + GAM Fit\n{factor_name_yield} on Yield', fontsize=14, pad=15)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{name}_shap_yield_mean_rh_1.png'), dpi=200)
    plt.close()

    print(f"SHAP Dependence plot: {save_dir}")