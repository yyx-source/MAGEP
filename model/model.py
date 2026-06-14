import torch
import torch.nn as nn

class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_adv=1.0):
        ctx.lambda_adv = lambda_adv
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output * (-ctx.lambda_adv)
        return grad_input, None


class GRL(nn.Module):
    def __init__(self, lambda_adv=1.0):
        super().__init__()
        self.lambda_adv = lambda_adv

    def forward(self, x):
        return GradientReverseLayer.apply(x, self.lambda_adv)

class GeneExtract(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, high_dim=256, output_dim=128):
        super().__init__()
        self.dilation_branches = nn.ModuleList()
        dilation_rates = [1, 2]
        kernel_size = 3
        for dilation in dilation_rates:
            pad = (dilation * (kernel_size - 1)) // 2
            branch = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size,
                          padding=pad, dilation=dilation),
                nn.GELU(),
                nn.BatchNorm1d(hidden_dim)
            )
            self.dilation_branches.append(branch)

        self.branch_fusion = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim * 2)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, high_dim),
            nn.GELU(),
            nn.LayerNorm(high_dim),
            nn.Linear(high_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim)
        )

        self.shortcut = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()

    def forward(self, x):
        x = x.unsqueeze(1)

        branch_feats = []
        for branch in self.dilation_branches:
            branch_feat = branch(x)
            branch_feats.append(branch_feat)

        fused_branch_feat = sum(branch_feats)
        fused_branch_feat = self.branch_fusion(fused_branch_feat)

        pool_feat = self.pool(fused_branch_feat).squeeze(-1)

        proj_feat = self.proj(pool_feat)

        basic_feat = self.pool(self.dilation_branches[0](x)).squeeze(-1)
        residual_feat = self.shortcut(basic_feat)

        return proj_feat + residual_feat

class EnvExtract(nn.Module):
    def __init__(self, seq_input_dim=4, stats_input_dim=16, hidden_dim=64, high_dim=256, output_dim=128):
        super().__init__()
        self.seq_emb = nn.Linear(seq_input_dim, hidden_dim)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=2,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)

        self.seq_proj = nn.Sequential(
            nn.Linear(hidden_dim, high_dim),
            nn.GELU(),
            nn.LayerNorm(high_dim),
            nn.Linear(high_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim)
        )

        self.stats_proj = nn.Sequential(
            nn.Linear(stats_input_dim, high_dim),
            nn.GELU(),
            nn.LayerNorm(high_dim),
            nn.Linear(high_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim)
        )

        self.env_fusion = nn.Sequential(
            nn.Linear(output_dim * 2, high_dim),
            nn.GELU(),
            nn.Linear(high_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, seq, stats):
        seq_emb = self.seq_emb(seq)
        trans_feat = self.transformer(seq_emb)
        seq_feat = trans_feat.mean(dim=1)
        seq_feat = self.seq_proj(seq_feat)
        stats_feat = self.stats_proj(stats)
        concat_env = torch.cat([seq_feat, stats_feat], dim=-1)
        fused_env_feat = self.env_fusion(concat_env)
        return fused_env_feat + (seq_feat + stats_feat) * 0.5


class GEFusion(nn.Module):
    def __init__(self, g_dim=128, e_dim=128, high_dim=256, hidden_dim=128, output_dim=128):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(g_dim + e_dim + g_dim, high_dim),
            nn.GELU(),
            nn.LayerNorm(high_dim),
            nn.Linear(high_dim, high_dim),
            nn.GELU(),
            nn.LayerNorm(high_dim),
            nn.Linear(high_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, g_feat, e_feat):
        interaction = g_feat * e_feat
        concat_feat = torch.cat([g_feat, e_feat, interaction], dim=-1)
        return self.fusion(concat_feat)


class PhenoHead(nn.Module):
    def __init__(self, input_dim=128, high_dim=256):
        super().__init__()
        self.shared_high_proj = nn.Sequential(
            nn.Linear(input_dim, high_dim),
            nn.GELU(),
            nn.LayerNorm(high_dim)
        )

        self.height_head = nn.Sequential(
            nn.Linear(high_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        self.yield_head = nn.Sequential(
            nn.Linear(high_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        high_feat = self.shared_high_proj(x)
        h = self.height_head(high_feat).squeeze(-1)
        y = self.yield_head(high_feat).squeeze(-1)
        return h, y


class MAGEPModel(nn.Module):
    def __init__(self, env_stats_dim=16, env_seq_dim=4, gene_hidden=32, env_hidden=64, fusion_hidden=128, high_dim=256):
        super().__init__()
        self.gene_encoder = GeneExtract(
            hidden_dim=gene_hidden,
            high_dim=high_dim,
            output_dim=fusion_hidden
        )

        self.env_encoder = EnvExtract(
            seq_input_dim=env_seq_dim,
            stats_input_dim=env_stats_dim,
            hidden_dim=env_hidden,
            high_dim=high_dim,
            output_dim=fusion_hidden
        )

        self.ge_fusion = GEFusion(
            g_dim=fusion_hidden,
            e_dim=fusion_hidden,
            high_dim=high_dim,
            output_dim=fusion_hidden
        )

        self.pheno_head = PhenoHead(
            input_dim=fusion_hidden,
            high_dim=high_dim
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(fusion_hidden, high_dim),
            nn.GELU(),
            nn.Linear(high_dim, high_dim // 2),
            nn.GELU(),
            nn.Linear(high_dim // 2, 1)
        )

        self.fusion = self.ge_fusion

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, G_seq, E_stats, E_seq, adv=False, lambda_adv=1.0):
        g_feat = self.gene_encoder(G_seq)
        e_feat = self.env_encoder(E_seq, E_stats)
        fused = self.ge_fusion(g_feat, e_feat)
        h_pred, y_pred = self.pheno_head(fused)

        domain_pred = None
        if adv:
            rev_feat = GradientReverseLayer.apply(fused, lambda_adv)
            domain_pred = self.domain_classifier(rev_feat).squeeze(-1)

        return h_pred, y_pred, domain_pred, fused