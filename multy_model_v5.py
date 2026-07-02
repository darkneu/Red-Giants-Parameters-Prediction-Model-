"""
两阶段融合模型训练脚本 (CNN + Transformer + 自适应多目标加权)

阶段 1: 仅功率谱 → SpectrumModel  (CNN + 轻量 Transformer)
阶段 2: 仅恒星参数 → StellarModel  (MLP)
阶段 3: 冻结 1&2, 训练 FusionHead  (融合两路预测)

亮点:
  - AdaptiveTargetHuberLoss : 按 EMA 损失动态平衡多目标 (借鉴 PatchTST 风格脚本)
  - OneCycleLR              : warmup + 余弦退火, 训练更稳
  - asinh_robust            : 抗离群值的逐样本归一化
  - 变长序列 + 动态 padding  : 训练数据无需等长
  - EMA 模型权重 (可选)      : 验证更稳
  - AMP 混合精度             : 加速 / 省显存
  - 详细 per-target 日志     : 每轮打印 loss / 权重 / pred_std
"""

import os
import math
import json
import argparse
import copy
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler


# ================================================================
# 1. 模型定义
# ================================================================

class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        r = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, r), nn.GELU(),
            nn.Linear(r, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        s = x.mean(dim=-1)
        s = self.fc(s).unsqueeze(-1)
        return x * s


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, stride=1, dropout=0.1, use_se=True):
        super().__init__()
        pad = kernel // 2
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride, pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, 1, pad, bias=False)
        self.drop = nn.Dropout1d(dropout)
        self.se = SEBlock1D(out_ch) if use_se else nn.Identity()
        self.short = (nn.Conv1d(in_ch, out_ch, 1, stride, bias=False)
                      if (stride != 1 or in_ch != out_ch) else nn.Identity())

    def forward(self, x):
        residual = self.short(x)
        out = self.conv1(F.gelu(self.bn1(x)))
        out = self.drop(out)
        out = self.conv2(F.gelu(self.bn2(out)))
        out = self.se(out)
        return out + residual


class AttentionPool1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):  # (B, L, D)
        w = torch.softmax(self.attn(x).squeeze(-1), dim=-1)
        return (x * w.unsqueeze(-1)).sum(dim=1)


class SpectrumModel(nn.Module):
    """CNN(ResNet1D + SE) + 轻量 Transformer"""
    def __init__(self, num_targets=1, base_ch=32, dropout=0.15,
                 use_transformer=True, n_trans_layers=2, n_heads=4):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 4

        self.stem = nn.Sequential(
            nn.Conv1d(1, c1, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(c1), nn.GELU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(
            ResBlock1D(c1, c1, dropout=dropout),
            ResBlock1D(c1, c2, stride=2, dropout=dropout),
        )
        self.stage2 = nn.Sequential(
            ResBlock1D(c2, c2, dropout=dropout),
            ResBlock1D(c2, c3, stride=2, dropout=dropout),
        )
        self.stage3 = nn.Sequential(
            ResBlock1D(c3, c3, dropout=dropout),
            ResBlock1D(c3, c4, stride=2, dropout=dropout),
        )
        d = c4

        self.use_trans = use_transformer
        if use_transformer:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=n_heads, dim_feedforward=d * 2,
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_trans_layers)
            self.pool = AttentionPool1D(d)
        else:
            self.pool = None

        self.norm = nn.LayerNorm(d)
        self.feat_dim = 64
        self.shared = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, self.feat_dim), nn.GELU(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(self.feat_dim, 1) for _ in range(num_targets)]
        )

    def encode(self, x, lengths=None):
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        if self.use_trans:
            x = x.transpose(1, 2)
            x = self.transformer(x)
            x = self.pool(x)
        else:
            x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.norm(x)
        return self.shared(x)

    def forward(self, x, lengths=None, return_feat=False):
        h = self.encode(x, lengths)
        pred = torch.cat([head(h) for head in self.heads], dim=1)
        return (pred, h) if return_feat else pred


class StellarModel(nn.Module):
    def __init__(self, input_dim, num_targets=1, hidden=128, dropout=0.1):
        super().__init__()
        assert input_dim >= 1
        self.feat_dim = 64
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),    nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, self.feat_dim), nn.GELU(),
        )
        self.heads = nn.ModuleList([nn.Linear(self.feat_dim, 1)
                                    for _ in range(num_targets)])

    def encode(self, x):
        return self.backbone(x)

    def forward(self, x, return_feat=False):
        h = self.encode(x)
        pred = torch.cat([head(h) for head in self.heads], dim=1)
        return (pred, h) if return_feat else pred


class FusionHead(nn.Module):
    def __init__(self, num_targets, mode="mlp_pred",
                 feat_dim_spec=64, feat_dim_stellar=64,
                 hidden=128, dropout=0.1):
        super().__init__()
        self.mode = mode
        self.num_targets = num_targets

        if mode == "avg":
            pass
        elif mode == "weighted":
            self.alpha_logit = nn.Parameter(torch.zeros(num_targets))
        elif mode == "mlp_pred":
            in_dim = num_targets * 2
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, num_targets),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        elif mode == "mlp_feat":
            in_dim = feat_dim_spec + feat_dim_stellar
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, num_targets),
            )
        else:
            raise ValueError(f"未知 fusion_mode: {mode}")

    def forward(self, pred_spec, pred_stellar, feat_spec=None, feat_stellar=None):
        if self.mode == "avg":
            return 0.5 * (pred_spec + pred_stellar)
        if self.mode == "weighted":
            w = torch.sigmoid(self.alpha_logit)
            return w * pred_spec + (1.0 - w) * pred_stellar
        if self.mode == "mlp_pred":
            x = torch.cat([pred_spec, pred_stellar], dim=-1)
            return 0.5 * (pred_spec + pred_stellar) + self.net(x)
        if self.mode == "mlp_feat":
            x = torch.cat([feat_spec, feat_stellar], dim=-1)
            return self.net(x)


# ================================================================
# 2. 自适应多目标加权 Huber 损失 (借鉴自 train_num.py)
# ================================================================

class AdaptiveTargetHuberLoss(nn.Module):
    """
    EMA 跟踪每个目标的近期损失, 难学的目标自动获得更高权重.
    weight_i = clip( sqrt(EMA_i / mean(EMA)), [min_w, max_w] )
    """
    def __init__(self, num_targets, delta=1.0, momentum=0.9,
                 min_weight=0.5, max_weight=3.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction="none")
        self.momentum = momentum
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.register_buffer("ema_losses", torch.ones(num_targets))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, preds, targets, update_state=True):
        loss_matrix = self.huber(preds, targets)        # (B, T)
        target_losses = loss_matrix.mean(dim=0)         # (T,)

        if update_state:
            d = target_losses.detach()
            if not bool(self.initialized.item()):
                self.ema_losses.copy_(d)
                self.initialized.fill_(True)
            else:
                self.ema_losses.mul_(self.momentum).add_((1.0 - self.momentum) * d)

        ref = self.ema_losses.mean().clamp_min(1e-6)
        weights = torch.sqrt(self.ema_losses / ref).clamp(self.min_weight, self.max_weight)
        weighted_loss = (target_losses * weights).mean()
        return weighted_loss, target_losses.detach(), weights.detach()


# ================================================================
# 3. EMA 权重平均 (新增)
# ================================================================

class ModelEMA:
    """权重指数滑动平均, 提升验证稳定性"""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ep, p in zip(self.ema.parameters(), model.parameters()):
            ep.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for eb, b in zip(self.ema.buffers(), model.buffers()):
            eb.copy_(b)


# ================================================================
# 4. 数据预处理
# ================================================================

def pad_and_align_sequences(sequences, patch_size=1, max_seq_len=None):
    if isinstance(sequences, np.ndarray) and sequences.ndim == 2:
        N, actual_len = sequences.shape
        data_max_len = actual_len
    else:
        raw_lengths = np.array([len(s) for s in sequences], dtype=np.int64)
        data_max_len = int(raw_lengths.max())

    effective_max = min(data_max_len, max_seq_len) if max_seq_len else data_max_len
    aligned_len = (math.ceil(effective_max / patch_size) * patch_size
                   if patch_size > 1 else effective_max)

    N = len(sequences) if not isinstance(sequences, np.ndarray) else sequences.shape[0]
    padded = np.zeros((N, aligned_len), dtype=np.float32)
    lengths = np.zeros(N, dtype=np.int64)

    if isinstance(sequences, np.ndarray) and sequences.ndim == 2:
        copy_len = min(sequences.shape[1], aligned_len)
        padded[:, :copy_len] = sequences[:, :copy_len]
        lengths[:] = copy_len
    else:
        for i, seq in enumerate(sequences):
            actual_len = min(len(seq), aligned_len)
            padded[i, :actual_len] = seq[:actual_len]
            lengths[i] = actual_len

    return padded, lengths, aligned_len


def normalize_per_spectrum(sequences, lengths, mode="asinh_robust",
                            eps=1e-10, clip_sigma=6.0):
    out = np.zeros_like(sequences)
    for i in range(len(sequences)):
        L = int(lengths[i])
        if L == 0:
            continue
        seg = sequences[i, :L].astype(np.float64)

        if mode == "raw_div2_16":
            out[i, :L] = (seg / float(2 ** 16)).astype(np.float32)
            continue

        seg = np.clip(seg, 0.0, None)

        if mode == "asinh_robust":
            scale = np.percentile(seg, 75) + 1e-6
            t = np.arcsinh(seg / scale)
            med = np.median(t)
            mad = np.median(np.abs(t - med)) + 1e-8
            normed = (t - med) / (1.4826 * mad)
            if clip_sigma:
                normed = np.clip(normed, -clip_sigma, clip_sigma)
            out[i, :L] = normed.astype(np.float32)

        elif mode == "log_robust":
            seg_log = np.log1p(seg)
            med = np.median(seg_log)
            mad = np.median(np.abs(seg_log - med))
            normed = (seg_log - med) / (1.4826 * mad + 1e-8)
            if clip_sigma:
                normed = np.clip(normed, -clip_sigma, clip_sigma)
            out[i, :L] = normed.astype(np.float32)
        else:
            raise ValueError(f"未知 mode: {mode}")
    return out


# ================================================================
# 5. 数据集
# ================================================================

class FusionDataset(Dataset):
    def __init__(self, sequences, stellar, targets, lengths):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.stellar = torch.tensor(stellar, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (self.sequences[idx], self.stellar[idx],
                self.targets[idx], self.lengths[idx])


# ================================================================
# 6. 训练循环 (带 AMP)
# ================================================================

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device,
                    branch="spec", scaler_amp=None, ema=None,
                    grad_clip=1.0):
    """branch: 'spec' / 'stellar' / 'fusion'"""
    model.train()
    criterion.train()
    total_loss = 0.0
    total_target_losses = None
    total_weights = None
    n_batches = 0

    for batch in loader:
        seqs, stellar, targets, lengths = [b.to(device, non_blocking=True) for b in batch]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=(scaler_amp is not None)):
            if branch == "spec":
                preds = model(seqs, lengths=lengths)
            elif branch == "stellar":
                preds = model(stellar)
            else:
                raise ValueError(branch)
            loss, target_losses, weights = criterion(preds, targets, update_state=True)

        if scaler_amp is not None:
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        if total_target_losses is None:
            total_target_losses = target_losses.clone()
            total_weights = weights.clone()
        else:
            total_target_losses += target_losses
            total_weights += weights
        n_batches += 1

    return (total_loss / n_batches,
            (total_target_losses / n_batches).cpu().numpy(),
            (total_weights / n_batches).cpu().numpy())


@torch.no_grad()
def evaluate(model, loader, criterion, device, scaler, branch="spec"):
    model.eval()
    criterion.eval()
    total_loss = 0.0
    total_target_losses = None
    all_preds, all_targets = [], []
    n_batches = 0

    for batch in loader:
        seqs, stellar, targets, lengths = [b.to(device, non_blocking=True) for b in batch]
        if branch == "spec":
            preds = model(seqs, lengths=lengths)
        elif branch == "stellar":
            preds = model(stellar)
        else:
            raise ValueError(branch)
        loss, target_losses, _ = criterion(preds, targets, update_state=False)
        total_loss += loss.item()
        if total_target_losses is None:
            total_target_losses = target_losses.clone()
        else:
            total_target_losses += target_losses
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        n_batches += 1

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    preds_real = scaler.inverse_transform(all_preds)
    targets_real = scaler.inverse_transform(all_targets)
    mae = np.mean(np.abs(preds_real - targets_real), axis=0)
    pred_std = np.std(preds_real, axis=0)

    return (total_loss / n_batches, mae, pred_std,
            (total_target_losses / n_batches).cpu().numpy())


# ================================================================
# 7. Fusion 阶段专用 (需要前两路同时跑)
# ================================================================

def train_fusion_epoch(spec_model, stellar_model, fusion, loader,
                      optimizer, scheduler, criterion, device,
                      ema=None, grad_clip=1.0):
    spec_model.eval()
    stellar_model.eval()
    fusion.train()
    criterion.train()

    total_loss = 0.0
    total_target_losses = None
    total_weights = None
    n_batches = 0

    for batch in loader:
        seqs, stellar, targets, lengths = [b.to(device, non_blocking=True) for b in batch]

        with torch.no_grad():
            ps, fs = spec_model(seqs, lengths=lengths, return_feat=True)
            pp, fp = stellar_model(stellar, return_feat=True)

        optimizer.zero_grad(set_to_none=True)
        fused = fusion(ps, pp, feat_spec=fs, feat_stellar=fp)
        loss, target_losses, weights = criterion(fused, targets, update_state=True)
        loss.backward()
        nn.utils.clip_grad_norm_(fusion.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(fusion)

        total_loss += loss.item()
        if total_target_losses is None:
            total_target_losses = target_losses.clone()
            total_weights = weights.clone()
        else:
            total_target_losses += target_losses
            total_weights += weights
        n_batches += 1

    return (total_loss / n_batches,
            (total_target_losses / n_batches).cpu().numpy(),
            (total_weights / n_batches).cpu().numpy())


@torch.no_grad()
def evaluate_fusion(spec_model, stellar_model, fusion, loader,
                    criterion, device, scaler):
    spec_model.eval()
    stellar_model.eval()
    fusion.eval()
    criterion.eval()

    total_loss = 0.0
    total_target_losses = None
    all_preds, all_targets = [], []
    n_batches = 0

    for batch in loader:
        seqs, stellar, targets, lengths = [b.to(device, non_blocking=True) for b in batch]
        ps, fs = spec_model(seqs, lengths=lengths, return_feat=True)
        pp, fp = stellar_model(stellar, return_feat=True)
        fused = fusion(ps, pp, feat_spec=fs, feat_stellar=fp)
        loss, target_losses, _ = criterion(fused, targets, update_state=False)
        total_loss += loss.item()
        if total_target_losses is None:
            total_target_losses = target_losses.clone()
        else:
            total_target_losses += target_losses
        all_preds.append(fused.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        n_batches += 1

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    preds_real = scaler.inverse_transform(all_preds)
    targets_real = scaler.inverse_transform(all_targets)
    mae = np.mean(np.abs(preds_real - targets_real), axis=0)
    pred_std = np.std(preds_real, axis=0)

    return (total_loss / n_batches, mae, pred_std,
            (total_target_losses / n_batches).cpu().numpy())


# ================================================================
# 8. 通用日志打印
# ================================================================

def fmt_arr(arr, prefix=""):
    return prefix + " ".join([f"{v:.4f}" for v in arr])


def log_epoch(epoch, total, train_loss, val_loss, mae, target_cols,
              train_tloss, train_w, val_tloss, pred_std, lr=None):
    lr_str = f"  lr={lr:.2e}" if lr is not None else ""
    print(f"Epoch {epoch:3d}/{total} | Train: {train_loss:.4f} | "
          f"Val: {val_loss:.4f}{lr_str}")
    print(f"  MAE         : {fmt_arr(mae)}  ({', '.join(target_cols)})")
    print(f"  Train T-loss: {fmt_arr(train_tloss)}")
    print(f"  Train weight: {fmt_arr(train_w)}")
    print(f"  Val   T-loss: {fmt_arr(val_tloss)}")
    print(f"  Pred  std   : {fmt_arr(pred_std)}")


# ================================================================
# 9. 主流程
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    # 数据
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--target_cols", type=str, default="mass,radius,age,logg")
    parser.add_argument("--log_target_cols", type=str, default="age",
                        help="逗号分隔, 训练前对其取 log10")
    parser.add_argument("--stellar_cols", type=str, default="",
                        help="恒星参数列, 逗号分隔")
    parser.add_argument("--log_stellar_cols", type=str, default="")
    parser.add_argument("--spectrum_prefix", type=str, default="spec_",
                        help="频谱列前缀, 不指定时排除其他列后剩下的全是频谱")
    parser.add_argument("--id_col", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--seq_norm", type=str, default="asinh_robust",
                        choices=["asinh_robust", "log_robust", "raw_div2_16"])
    parser.add_argument("--seq_clip_sigma", type=float, default=6.0)
    parser.add_argument("--patch_size", type=int, default=1,
                        help="数据对齐用 (CNN 通常无需对齐, 设 1 即可)")

    # 训练
    parser.add_argument("--epochs_spec",    type=int, default=120)
    parser.add_argument("--epochs_stellar", type=int, default=80)
    parser.add_argument("--epochs_fusion",  type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_spec",    type=float, default=5e-4)
    parser.add_argument("--lr_stellar", type=float, default=1e-3)
    parser.add_argument("--lr_fusion",  type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # 损失
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--target_weight_momentum", type=float, default=0.9)
    parser.add_argument("--target_weight_min", type=float, default=0.5)
    parser.add_argument("--target_weight_max", type=float, default=3.0)

    # 模型
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--use_transformer", action="store_true", default=True)
    parser.add_argument("--n_trans_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--stellar_hidden", type=int, default=128)
    parser.add_argument("--fusion_mode", type=str, default="mlp_pred",
                        choices=["avg", "weighted", "mlp_pred", "mlp_feat"])
    parser.add_argument("--fusion_hidden", type=int, default=128)

    # 输出
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_spec",    action="store_true")
    parser.add_argument("--skip_stellar", action="store_true")
    parser.add_argument("--skip_fusion",  action="store_true")
    args = parser.parse_args()

    # ---- 设置 ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"设备: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    target_cols      = [c.strip() for c in args.target_cols.split(",") if c.strip()]
    log_target_cols  = [c.strip() for c in args.log_target_cols.split(",") if c.strip()]
    stellar_cols     = [c.strip() for c in args.stellar_cols.split(",") if c.strip()]
    log_stellar_cols = [c.strip() for c in args.log_stellar_cols.split(",") if c.strip()]
    log_target_indices = [target_cols.index(c) for c in log_target_cols if c in target_cols]
    log_sp_indices     = [stellar_cols.index(c) for c in log_stellar_cols if c in stellar_cols]
    num_targets = len(target_cols)

    print(f"目标列: {target_cols}")
    print(f"  log10 目标: {log_target_cols} → 索引 {log_target_indices}")
    print(f"恒星参数列: {stellar_cols}")
    print(f"  log10 恒星: {log_stellar_cols} → 索引 {log_sp_indices}")

    # ---- 读数据 ----
    df = pd.read_csv(args.data)
    print(f"原始数据形状: {df.shape}")

    # 新增：过滤 age > 15 的样本
    if "age" in target_cols:
        age_col = "age"
        mask = df[age_col] <= 15.0
        drop_cnt = (~mask).sum()
        if drop_cnt > 0:
            df = df[mask].reset_index(drop=True)
            print(f"过滤掉 age > 15 的样本数量：{drop_cnt}，过滤后数据形状：{df.shape}")
        else:
            print("所有样本 age ≤ 15，无需过滤")
    else:
        print("目标列不含 age，跳过 age 过滤")

    # 频谱列
    if args.spectrum_prefix:
        spec_cols = [c for c in df.columns if c.startswith(args.spectrum_prefix)]
    else:
        excl = set(stellar_cols) | set(target_cols)
        for k in ("obsid", "id", "name", "ID", "KIC", "Kepler_ID"):
            if k in df.columns:
                excl.add(k)
        if args.id_col and args.id_col in df.columns:
            excl.add(args.id_col)
        spec_cols = [c for c in df.columns if c not in excl]
    print(f"频谱列数: {len(spec_cols)}")

    sequences_raw = df[spec_cols].to_numpy(dtype=np.float32)
    print(f"原始频谱形状: {sequences_raw.shape}")

    # 对齐 + 归一化
    padded_seq, lengths, aligned_len = pad_and_align_sequences(
        sequences_raw, patch_size=args.patch_size, max_seq_len=args.max_seq_len)
    print(f"对齐长度: {aligned_len}, 长度统计: "
          f"min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.0f}")

    clip = args.seq_clip_sigma if args.seq_clip_sigma > 0 else None
    padded_seq = normalize_per_spectrum(
        padded_seq, lengths, args.seq_norm, clip_sigma=clip)
    print(f"频谱归一化: {args.seq_norm}, clip_sigma={clip}")

    # 目标
    targets_raw = df[target_cols].to_numpy(dtype=np.float64)
    targets_proc = targets_raw.copy()
    for idx in log_target_indices:
        col = targets_proc[:, idx]
        if np.any(col <= 0):
            raise ValueError(f"目标列 '{target_cols[idx]}' 含非正值, 无法 log10")
        targets_proc[:, idx] = np.log10(col)
    target_scaler = StandardScaler()
    targets_scaled = target_scaler.fit_transform(targets_proc).astype(np.float32)

    # 恒星参数
    if stellar_cols:
        miss = [c for c in stellar_cols if c not in df.columns]
        if miss:
            raise ValueError(f"缺少恒星参数列: {miss}")
        stellar_raw = df[stellar_cols].to_numpy(dtype=np.float64)
        for idx in log_sp_indices:
            col = stellar_raw[:, idx]
            if np.any(col <= 0):
                raise ValueError(f"恒星参数 '{stellar_cols[idx]}' 含非正值, 无法 log10")
            stellar_raw[:, idx] = np.log10(col)
        sp_mean = stellar_raw.mean(axis=0)
        sp_std = stellar_raw.std(axis=0) + 1e-8
        stellar_scaled = ((stellar_raw - sp_mean) / sp_std).astype(np.float32)
    else:
        sp_mean = None
        sp_std = None
        stellar_scaled = np.zeros((len(padded_seq), 0), dtype=np.float32)

    print(f"目标原始 std: {targets_raw.std(axis=0)}")

    # ---- 划分 ----
    full_dataset = FusionDataset(padded_seq, stellar_scaled, targets_scaled, lengths)
    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed))
    n_workers = min(os.cpu_count() or 1, 4)
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=n_workers, pin_memory=pin)
    print(f"训练 {train_size} | 验证 {val_size}")

    # ---- 通用 meta (所有 ckpt 共享) ----
    common_meta = {
        "target_cols":         target_cols,
        "log_target_cols":     log_target_cols,
        "log_target_indices":  log_target_indices,
        "target_scaler_mean":  target_scaler.mean_,
        "target_scaler_scale": target_scaler.scale_,
        "stellar_cols":        stellar_cols,
        "log_stellar_cols":    log_stellar_cols,
        "log_sp_indices":      log_sp_indices,
        "sp_mean":             sp_mean,
        "sp_std":              sp_std,
        "seq_norm":            args.seq_norm,
        "seq_clip_sigma":      args.seq_clip_sigma,
        "aligned_len":         aligned_len,
        "patch_size":          args.patch_size,
        "args":                vars(args),
    }

    # =========================================================
    # 阶段 1: SpectrumModel
    # =========================================================
    spec_ckpt_path = os.path.join(args.save_dir, "best_spec.pt")
    if not args.skip_spec:
        print("\n" + "=" * 60)
        print("阶段 1: 训练 SpectrumModel (CNN + Transformer)")
        print("=" * 60)

        spec_model = SpectrumModel(
            num_targets=num_targets, base_ch=args.base_ch,
            dropout=args.dropout, use_transformer=args.use_transformer,
            n_trans_layers=args.n_trans_layers, n_heads=args.n_heads,
        ).to(device)
        print(f"参数量: {sum(p.numel() for p in spec_model.parameters()):,}")

        criterion = AdaptiveTargetHuberLoss(
            num_targets=num_targets, delta=args.huber_delta,
            momentum=args.target_weight_momentum,
            min_weight=args.target_weight_min, max_weight=args.target_weight_max,
        ).to(device)
        optimizer = torch.optim.AdamW(spec_model.parameters(),
                                      lr=args.lr_spec,
                                      weight_decay=args.weight_decay)
        total_steps = max(1, len(train_loader) * args.epochs_spec)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr_spec, total_steps=total_steps,
            pct_start=0.1, anneal_strategy="cos")
        scaler_amp = torch.amp.GradScaler() if (args.use_amp and device.type == "cuda") else None
        ema = ModelEMA(spec_model, args.ema_decay) if args.use_ema else None

        best_val = float("inf")
        patience = 0
        for epoch in range(1, args.epochs_spec + 1):
            tr_loss, tr_tl, tr_w = train_one_epoch(
                spec_model, train_loader, optimizer, scheduler, criterion,
                device, branch="spec", scaler_amp=scaler_amp, ema=ema,
                grad_clip=args.grad_clip)
            eval_model = ema.ema if ema is not None else spec_model
            val_loss, mae, pstd, val_tl = evaluate(
                eval_model, val_loader, criterion, device,
                target_scaler, branch="spec")
            log_epoch(epoch, args.epochs_spec, tr_loss, val_loss, mae,
                      target_cols, tr_tl, tr_w, val_tl, pstd,
                      lr=optimizer.param_groups[0]["lr"])

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                torch.save({
                    "model_state": eval_model.state_dict(),
                    **common_meta,
                }, spec_ckpt_path)
                print(f"  ✓ 保存最佳 SpectrumModel (val={val_loss:.4f})")
            else:
                patience += 1
                if patience >= args.patience:
                    print(f"  ⏹ 早停于 epoch {epoch}")
                    break
        print(f"阶段 1 完成, best val_loss={best_val:.4f}")
    else:
        print("跳过阶段 1 (SpectrumModel)")

    # =========================================================
    # 阶段 2: StellarModel
    # =========================================================
    stellar_ckpt_path = os.path.join(args.save_dir, "best_stellar.pt")
    if not args.skip_stellar and stellar_cols:
        print("\n" + "=" * 60)
        print("阶段 2: 训练 StellarModel (MLP)")
        print("=" * 60)

        stellar_model = StellarModel(
            input_dim=len(stellar_cols), num_targets=num_targets,
            hidden=args.stellar_hidden, dropout=args.dropout,
        ).to(device)
        print(f"参数量: {sum(p.numel() for p in stellar_model.parameters()):,}")

        criterion = AdaptiveTargetHuberLoss(
            num_targets=num_targets, delta=args.huber_delta,
            momentum=args.target_weight_momentum,
            min_weight=args.target_weight_min, max_weight=args.target_weight_max,
        ).to(device)
        optimizer = torch.optim.AdamW(stellar_model.parameters(),
                                      lr=args.lr_stellar,
                                      weight_decay=args.weight_decay)
        total_steps = max(1, len(train_loader) * args.epochs_stellar)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr_stellar, total_steps=total_steps,
            pct_start=0.1, anneal_strategy="cos")
        ema = ModelEMA(stellar_model, args.ema_decay) if args.use_ema else None

        best_val = float("inf")
        patience = 0
        for epoch in range(1, args.epochs_stellar + 1):
            tr_loss, tr_tl, tr_w = train_one_epoch(
                stellar_model, train_loader, optimizer, scheduler, criterion,
                device, branch="stellar", ema=ema, grad_clip=args.grad_clip)
            eval_model = ema.ema if ema is not None else stellar_model
            val_loss, mae, pstd, val_tl = evaluate(
                eval_model, val_loader, criterion, device,
                target_scaler, branch="stellar")
            log_epoch(epoch, args.epochs_stellar, tr_loss, val_loss, mae,
                      target_cols, tr_tl, tr_w, val_tl, pstd,
                      lr=optimizer.param_groups[0]["lr"])

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                torch.save({
                    "model_state": eval_model.state_dict(),
                    **common_meta,
                }, stellar_ckpt_path)
                print(f"  ✓ 保存最佳 StellarModel (val={val_loss:.4f})")
            else:
                patience += 1
                if patience >= args.patience:
                    print(f"  ⏹ 早停于 epoch {epoch}")
                    break
        print(f"阶段 2 完成, best val_loss={best_val:.4f}")
    elif not stellar_cols:
        print("无恒星参数列, 跳过阶段 2 / 3")
        return
    else:
        print("跳过阶段 2 (StellarModel)")

    # =========================================================
    # 阶段 3: FusionHead
    # =========================================================
    if args.skip_fusion:
        print("跳过阶段 3 (FusionHead)")
        return

    print("\n" + "=" * 60)
    print(f"阶段 3: 训练 FusionHead (mode={args.fusion_mode})")
    print("=" * 60)

    # 加载阶段 1 / 2
    spec_model = SpectrumModel(
        num_targets=num_targets, base_ch=args.base_ch,
        dropout=args.dropout, use_transformer=args.use_transformer,
        n_trans_layers=args.n_trans_layers, n_heads=args.n_heads,
    ).to(device)
    spec_ckpt = torch.load(spec_ckpt_path, map_location=device, weights_only=False)
    spec_model.load_state_dict(spec_ckpt["model_state"])
    spec_model.eval()

    stellar_model = StellarModel(
        input_dim=len(stellar_cols), num_targets=num_targets,
        hidden=args.stellar_hidden, dropout=args.dropout,
    ).to(device)
    stellar_ckpt = torch.load(stellar_ckpt_path, map_location=device, weights_only=False)
    stellar_model.load_state_dict(stellar_ckpt["model_state"])
    stellar_model.eval()

    fusion = FusionHead(
        num_targets=num_targets, mode=args.fusion_mode,
        feat_dim_spec=spec_model.feat_dim,
        feat_dim_stellar=stellar_model.feat_dim,
        hidden=args.fusion_hidden, dropout=args.dropout,
    ).to(device)
    n_fusion = sum(p.numel() for p in fusion.parameters())
    print(f"FusionHead 参数量: {n_fusion:,}")

    if n_fusion == 0:
        # mode='avg' 没有可学习参数, 直接评估
        print("FusionHead 无参数, 直接评估融合效果...")
        criterion = AdaptiveTargetHuberLoss(
            num_targets=num_targets, delta=args.huber_delta).to(device)
        val_loss, mae, pstd, val_tl = evaluate_fusion(
            spec_model, stellar_model, fusion, val_loader,
            criterion, device, target_scaler)
        print(f"  Val={val_loss:.4f}, MAE={fmt_arr(mae)}")
        torch.save({
            "model_state": fusion.state_dict(),
            "spec_ckpt":    spec_ckpt_path,
            "stellar_ckpt": stellar_ckpt_path,
            **common_meta,
        }, os.path.join(args.save_dir, "best_fusion.pt"))
        return

    criterion = AdaptiveTargetHuberLoss(
        num_targets=num_targets, delta=args.huber_delta,
        momentum=args.target_weight_momentum,
        min_weight=args.target_weight_min, max_weight=args.target_weight_max,
    ).to(device)
    optimizer = torch.optim.AdamW(fusion.parameters(),
                                  lr=args.lr_fusion,
                                  weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs_fusion)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr_fusion, total_steps=total_steps,
        pct_start=0.1, anneal_strategy="cos")
    ema = ModelEMA(fusion, args.ema_decay) if args.use_ema else None

    fusion_ckpt_path = os.path.join(args.save_dir, "best_fusion.pt")
    best_val = float("inf")
    patience = 0
    for epoch in range(1, args.epochs_fusion + 1):
        tr_loss, tr_tl, tr_w = train_fusion_epoch(
            spec_model, stellar_model, fusion, train_loader,
            optimizer, scheduler, criterion, device,
            ema=ema, grad_clip=args.grad_clip)
        eval_fusion = ema.ema if ema is not None else fusion
        val_loss, mae, pstd, val_tl = evaluate_fusion(
            spec_model, stellar_model, eval_fusion, val_loader,
            criterion, device, target_scaler)
        log_epoch(epoch, args.epochs_fusion, tr_loss, val_loss, mae,
                  target_cols, tr_tl, tr_w, val_tl, pstd,
                  lr=optimizer.param_groups[0]["lr"])

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save({
                "model_state":  eval_fusion.state_dict(),
                "spec_ckpt":    spec_ckpt_path,
                "stellar_ckpt": stellar_ckpt_path,
                **common_meta,
            }, fusion_ckpt_path)
            print(f"  ✓ 保存最佳 FusionHead (val={val_loss:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  ⏹ 早停于 epoch {epoch}")
                break

    print(f"\n训练完成. 三阶段 ckpt 保存在 {args.save_dir}/")
    print(f"  → best_spec.pt")
    print(f"  → best_stellar.pt")
    print(f"  → best_fusion.pt")


if __name__ == "__main__":
    main()