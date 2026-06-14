import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

def parse_location(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    for suffix in ['_gene', '_env', '_pheno']:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name

def extract_env_features(E_numeric, time_series, target_len=1800):
    T, C = E_numeric.shape
    features = []
    col_names = ['Temperature', 'RelativeHumidity', 'SolarRadiation', 'Rainfall']

    names = []
    for col in col_names:
        for agg in ['mean', 'std', 'min', 'max', 'median']:
            names.append(f'{agg}_{col}')
        for p in ['5th', '25th', '75th', '95th']:
            names.append(f'{p}_percentile_{col}')

    for func in [np.mean, np.std, np.min, np.max, np.median]:
        features.extend(func(E_numeric, axis=0))

    for q in [0.05, 0.25, 0.75, 0.95]:
        features.extend(np.quantile(E_numeric, q, axis=0))

    T_col = E_numeric[:, 0]
    RH_col = E_numeric[:, 1]
    SR_col = E_numeric[:, 2]
    Rain_col = E_numeric[:, 3]

    if C >= 2:
        es = 0.611 * np.exp(17.27 * T_col / (T_col + 237.15))
        ea = es * (RH_col / 100)
        VPD = es - ea
        features.append(np.mean(VPD))
        features.append(np.sum(VPD))
        names += ['mean_VPD', 'cumulative_VPD']

        time_dt = pd.to_datetime(time_series)
        hours = time_dt.hour
        day_mask = (hours >= 6) & (hours < 18)
        night_mask = ~day_mask
        day_rh_mean = np.mean(RH_col[day_mask]) if day_mask.sum() > 0 else 0.0
        night_rh_mean = np.mean(RH_col[night_mask]) if night_mask.sum() > 0 else 0.0
        features.append(day_rh_mean - night_rh_mean)
        names.append('day_night_RH_difference')

    if C >= 3:
        effective_light = np.sum(SR_col > 10)
        features.append(effective_light)
        names.append('effective_light_hours')

        if 'day_mask' in locals():
            day_effective_light = np.sum(SR_col[day_mask] > 10)
            features.append(day_effective_light)
            names.append('daytime_effective_light_hours')

    if C >= 4:
        cumulative_rain = np.sum(Rain_col)
        features.append(cumulative_rain)
        names.append('cumulative_rainfall')

    if C > 0:
        gdd = np.sum(np.maximum(T_col - 10, 0))
        features.append(gdd)
        names.append('growing_degree_days')

    features = np.array(features, dtype=np.float32)

    if T > target_len:
        indices = np.linspace(0, T-1, target_len, dtype=int)
        seq = E_numeric[indices]
    else:
        seq = np.pad(E_numeric, ((0, target_len - T), (0, 0)), mode='constant')
    seq = seq.astype(np.float32)

    return features, seq, names

class GEPDataset(Dataset):
    def __init__(self, geno_dir, env_dir, pheno_dir, locations, domain_label, y_scaler=None,
                 env_seq_scaler=None, env_stats_scaler=None, is_train=False):
        self.samples = []
        self.domain_label = domain_label  # 0 for source, 1 for target
        self.y_scaler = y_scaler
        self.env_seq_scaler = env_seq_scaler
        self.env_stats_scaler = env_stats_scaler
        y_raw_list = []

        for loc in locations:
            geno_file = next((f for f in os.listdir(geno_dir) if parse_location(f) == loc and '_gene.csv' in f), None)
            env_file = next((f for f in os.listdir(env_dir) if parse_location(f) == loc and '_env.csv' in f), None)
            pheno_file = next((f for f in os.listdir(pheno_dir) if parse_location(f) == loc and '_pheno.csv' in f), None)
            if not all([geno_file, env_file, pheno_file]):
                continue

            print(f"Location: {loc} (domain: {'source' if domain_label == 0 else 'target'})")

            G_df = pd.read_csv(os.path.join(geno_dir, geno_file), index_col=0)
            G_values = G_df.values.astype(np.float32)

            self.snp_names = G_df.columns.tolist()
            print(f"{geno_file} has {len(self.snp_names)} SNPs")


            E_df = pd.read_csv(os.path.join(env_dir, env_file))
            time_series = E_df.iloc[:, 1].values
            E_numeric = E_df.iloc[:, 2:].values.astype(np.float32)
            E_stats, E_seq, env_stats_names = extract_env_features(E_numeric, time_series)

            if not hasattr(self, 'env_stats_names'):
                self.env_stats_names = env_stats_names
            else:
                assert self.env_stats_names == env_stats_names

            P_df = pd.read_csv(os.path.join(pheno_dir, pheno_file))
            sample_col = P_df.columns[1]
            P_df = P_df.set_index(sample_col)
            P_df.index = P_df.index.astype(str)

            height_col = next((c for c in P_df.columns if 'height' in c.lower() or 'cm' in c.lower()), None)
            yield_col = next((c for c in P_df.columns if 'yield' in c.lower() or 'bu/a' in c.lower()), None)
            if not height_col or not yield_col:
                continue

            common = G_df.index.intersection(P_df.index)
            if len(common) == 0:
                continue

            print(f"  {loc}: {len(common)} samples")

            for i, sample in enumerate(common):
                g_seq = G_values[G_df.index.get_loc(sample)]
                try:
                    h = float(P_df.loc[sample, height_col])
                    y = float(P_df.loc[sample, yield_col])
                except:
                    continue
                if np.isnan(h) or np.isnan(y):
                    continue

                if is_train:
                    y_raw_list.append([h, y])

                self.samples.append({
                    'G_seq': torch.from_numpy(g_seq).float(),
                    'E_stats_raw': torch.from_numpy(E_stats).float(),
                    'E_seq_raw': torch.from_numpy(E_seq).float(),
                    'y_raw': torch.tensor([h, y], dtype=torch.float32),
                    'domain_id': torch.tensor(domain_label, dtype=torch.long),
                    'location': loc,
                    'sample': sample
                })

        if is_train and y_raw_list:
            y_arr = np.stack(y_raw_list)
            self.y_scaler = StandardScaler()
            self.y_scaler.fit(y_arr)

            all_seq = np.concatenate([s['E_seq_raw'].numpy().reshape(-1,4) for s in self.samples], axis=0)
            self.env_seq_scaler = StandardScaler()
            self.env_seq_scaler.fit(all_seq)

            all_stats = np.stack([s['E_stats_raw'].numpy() for s in self.samples])
            self.env_stats_scaler = StandardScaler()
            self.env_stats_scaler.fit(all_stats)

        for s in self.samples:
            seq_flat = s['E_seq_raw'].numpy().reshape(-1, 4)
            seq_scaled = self.env_seq_scaler.transform(seq_flat).reshape(s['E_seq_raw'].shape)
            s['E_seq'] = torch.from_numpy(seq_scaled).float()

            stats_scaled = self.env_stats_scaler.transform(s['E_stats_raw'].numpy().reshape(1, -1))[0]
            s['E_stats'] = torch.from_numpy(stats_scaled).float()

            if self.y_scaler is not None:
                y_norm = self.y_scaler.transform(s['y_raw'].numpy().reshape(1, -1))[0]
                s['y_norm'] = torch.tensor(y_norm, dtype=torch.float32)
            else:
                s['y_norm'] = None

        print(f"loading: {len(self.samples)} sample (domain {'source' if domain_label == 0 else 'target'})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            'G_seq': item['G_seq'],
            'E_stats': item['E_stats'],
            'E_seq': item['E_seq'],
            'y_norm': item['y_norm'],
            'y_raw': item['y_raw'],
            'domain_id': item['domain_id'],
            'location': item['location'],
            'sample': item['sample']
        }

def gep_collate_fn(batch):
    y_norm = [b['y_norm'] for b in batch]
    if y_norm[0] is not None:
        y_norm = torch.stack(y_norm)
    else:
        y_norm = None

    return {
        'G_seq': torch.stack([b['G_seq'] for b in batch]),
        'E_stats': torch.stack([b['E_stats'] for b in batch]),
        'E_seq': torch.stack([b['E_seq'] for b in batch]),
        'y_norm': y_norm,
        'y_raw': torch.stack([b['y_raw'] for b in batch]),
        'domain_id': torch.stack([b['domain_id'] for b in batch]),
        'location': [b['location'] for b in batch],
        'sample': [b['sample'] for b in batch]
    }