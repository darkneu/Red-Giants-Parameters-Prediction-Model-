"""
两阶段融合模型推理脚本

用法:
    # 仅频谱推理
    python infer_num.py --data test.csv --ckpt checkpoints/best_spec.pt --mode spec

    # 仅恒星参数推理
    python infer_num.py --data test.csv --ckpt checkpoints/best_stellar.pt --mode stellar

    # 融合推理 (推荐, 自动加载三个 ckpt)
    python infer_num.py --data test.csv --ckpt checkpoints/best_fusion.pt --mode fusion

    # 同时输出三种结果对比
    python infer_num.py --data test.csv --ckpt_dir checkpoints --mode all
"""

import os
import math
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# 1. 模型定义 (必须与训练脚本完全一致)
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

    def forward(self, x):
        w = torch.softmax(self.attn(x).squeeze(-1), dim=-1)
        return (x * w.unsqueeze(-1)).sum(dim=1)


class SpectrumModel(nn.Module):
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
# 2. 预处理 (与训练完全一致)
# ================================================================

def pad_and_align_sequences(sequences, patch_size=1, target_len=None):
    """target_len 不为 None 时, 强制对齐到该长度 (推理时与训练一致)"""
    if isinstance(sequences, np.ndarray) and sequences.ndim == 2:
        N, actual_len = sequences.shape
        data_max_len = actual_len
    else:
        raw_lengths = np.array([len(s) for s in sequences], dtype=np.int64)
        data_max_len = int(raw_lengths.max())
        N = len(sequences)

    if target_len is not None:
        aligned_len = target_len
    else:
        if patch_size > 1:
            aligned_len = math.ceil(data_max_len / patch_size) * patch_size
        else:
            aligned_len = data_max_len

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


def inverse_transform_targets(scaled_preds, scaler_mean, scaler_scale,
                               log_target_indices):
    """
    1) StandardScaler 反归一化
    2) 对原本做了 log10 的列做 10**x 还原
    """
    out = scaled_preds * scaler_scale + scaler_mean   # 反 StandardScaler
    out = out.copy()
    for idx in log_target_indices:
        out[:, idx] = np.power(10.0, out[:, idx])
    return out


# ================================================================
# 3. ckpt 加载工具
# ================================================================

def load_meta(ckpt):
    """从 ckpt 中提取通用元信息"""
    args = ckpt["args"]
    meta = {
        "target_cols":         ckpt["target_cols"],
        "log_target_cols":     ckpt.get("log_target_cols", []),
        "log_target_indices":  ckpt.get("log_target_indices", []),
        "target_scaler_mean":  np.asarray(ckpt["target_scaler_mean"]),
        "target_scaler_scale": np.asarray(ckpt["target_scaler_scale"]),
        "stellar_cols":        ckpt.get("stellar_cols", []),
        "log_stellar_cols":    ckpt.get("log_stellar_cols", []),
        "log_sp_indices":      ckpt.get("log_sp_indices", []),
        "sp_mean":             ckpt.get("sp_mean", None),
        "sp_std":              ckpt.get("sp_std", None),
        "seq_norm":            ckpt.get("seq_norm", "asinh_robust"),
        "seq_clip_sigma":      ckpt.get("seq_clip_sigma", 6.0),
        "aligned_len":         ckpt["aligned_len"],
        "patch_size":          ckpt.get("patch_size", 1),
        "args":                args,
    }
    return meta


def build_spec_model(meta, device):
    a = meta["args"]
    model = SpectrumModel(
        num_targets=len(meta["target_cols"]),
        base_ch=a.get("base_ch", 32),
        dropout=a.get("dropout", 0.15),
        use_transformer=a.get("use_transformer", True),
        n_trans_layers=a.get("n_trans_layers", 2),
        n_heads=a.get("n_heads", 4),
    ).to(device)
    return model


def build_stellar_model(meta, device):
    a = meta["args"]
    model = StellarModel(
        input_dim=len(meta["stellar_cols"]),
        num_targets=len(meta["target_cols"]),
        hidden=a.get("stellar_hidden", 128),
        dropout=a.get("dropout", 0.1),
    ).to(device)
    return model


def build_fusion_head(meta, device):
    a = meta["args"]
    head = FusionHead(
        num_targets=len(meta["target_cols"]),
        mode=a.get("fusion_mode", "mlp_pred"),
        feat_dim_spec=64,
        feat_dim_stellar=64,
        hidden=a.get("fusion_hidden", 128),
        dropout=a.get("dropout", 0.1),
    ).to(device)
    return head


# ================================================================
# 4. 数据准备
# ================================================================

def prepare_data(df, meta, id_col=None, spectrum_prefix=None, max_seq_len=None):
    """根据 meta 中保存的列名 / 归一化方式还原训练时一致的输入"""
    target_cols  = meta["target_cols"]
    stellar_cols = meta["stellar_cols"]

    # ---- 频谱列 ----
    if spectrum_prefix:
        spec_cols = [c for c in df.columns if c.startswith(spectrum_prefix)]
    else:
        excl = set(stellar_cols) | set(target_cols)
        for k in ("KIC", "TIC"):
            if k in df.columns:
                excl.add(k)
        if id_col and id_col in df.columns:
            excl.add(id_col)
        spec_cols = [c for c in df.columns if c not in excl]
    print(f"  频谱列数: {len(spec_cols)}")

    sequences_raw = df[spec_cols].to_numpy(dtype=np.float32)
    print(f"  原始频谱形状: {sequences_raw.shape}")

    padded_seq, lengths, aligned_len = pad_and_align_sequences(
        sequences_raw, patch_size=meta["patch_size"],
        target_len=meta["aligned_len"])
    print(f"  对齐长度: {aligned_len} (训练时长度: {meta['aligned_len']})")

    clip = meta["seq_clip_sigma"] if meta["seq_clip_sigma"] > 0 else None
    padded_seq = normalize_per_spectrum(
        padded_seq, lengths, meta["seq_norm"], clip_sigma=clip)

    # ---- 恒星参数 ----
    if stellar_cols:
        miss = [c for c in stellar_cols if c not in df.columns]
        if miss:
            raise ValueError(f"输入数据缺少恒星参数列: {miss}")
        stellar_raw = df[stellar_cols].to_numpy(dtype=np.float64)
        for idx in meta["log_sp_indices"]:
            col = stellar_raw[:, idx]
            if np.any(col <= 0):
                raise ValueError(f"恒星参数 '{stellar_cols[idx]}' 含非正值, 无法 log10")
            stellar_raw[:, idx] = np.log10(col)
        sp_mean = np.asarray(meta["sp_mean"])
        sp_std  = np.asarray(meta["sp_std"])
        stellar_scaled = ((stellar_raw - sp_mean) / sp_std).astype(np.float32)
    else:
        stellar_scaled = np.zeros((len(padded_seq), 0), dtype=np.float32)

    # ---- 真值 (可选, 用于评估) ----
    has_gt = all(c in df.columns for c in target_cols)
    if has_gt:
        targets_raw = df[target_cols].to_numpy(dtype=np.float64)
    else:
        targets_raw = None

    return padded_seq, stellar_scaled, lengths, targets_raw, has_gt


# ================================================================
# 5. 推理函数
# ================================================================

@torch.no_grad()
def infer_spec(model, sequences, lengths, batch_size=128, device="cpu"):
    model.eval()
    n = len(sequences)
    preds, feats = [], []
    for i in range(0, n, batch_size):
        x = torch.tensor(sequences[i:i+batch_size], dtype=torch.float32, device=device)
        L = torch.tensor(lengths[i:i+batch_size], dtype=torch.long, device=device)
        p, h = model(x, lengths=L, return_feat=True)
        preds.append(p.cpu().numpy())
        feats.append(h.cpu().numpy())
    return np.concatenate(preds), np.concatenate(feats)


@torch.no_grad()
def infer_stellar(model, stellar, batch_size=128, device="cpu"):
    model.eval()
    n = len(stellar)
    preds, feats = [], []
    for i in range(0, n, batch_size):
        x = torch.tensor(stellar[i:i+batch_size], dtype=torch.float32, device=device)
        p, h = model(x, return_feat=True)
        preds.append(p.cpu().numpy())
        feats.append(h.cpu().numpy())
    return np.concatenate(preds), np.concatenate(feats)


@torch.no_grad()
def infer_fusion(fusion, pred_spec, pred_stellar, feat_spec, feat_stellar,
                 batch_size=512, device="cpu"):
    fusion.eval()
    n = len(pred_spec)
    out = []
    for i in range(0, n, batch_size):
        ps = torch.tensor(pred_spec[i:i+batch_size], device=device)
        pp = torch.tensor(pred_stellar[i:i+batch_size], device=device)
        fs = torch.tensor(feat_spec[i:i+batch_size], device=device)
        fp = torch.tensor(feat_stellar[i:i+batch_size], device=device)
        out.append(fusion(ps, pp, feat_spec=fs, feat_stellar=fp).cpu().numpy())
    return np.concatenate(out)


# ================================================================
# 6. 评估指标
# ================================================================

def compute_metrics(pred_real, target_real, target_cols):
    """对真实尺度下的 MAE / RMSE / MAPE / R2"""
    metrics = {}
    for i, name in enumerate(target_cols):
        p = pred_real[:, i]
        t = target_real[:, i]
        mae = np.mean(np.abs(p - t))
        rmse = np.sqrt(np.mean((p - t) ** 2))
        mape = np.mean(np.abs((p - t) / (np.abs(t) + 1e-8))) * 100
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        metrics[name] = {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape, "R2": r2}
    return metrics


def print_metrics(metrics, title=""):
    if title:
        print(f"\n--- {title} ---")
    header = f"{'Target':<10} {'MAE':>12} {'RMSE':>12} {'MAPE(%)':>10} {'R2':>8}"
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        print(f"{name:<10} {m['MAE']:>12.4f} {m['RMSE']:>12.4f} "
              f"{m['MAPE(%)']:>10.2f} {m['R2']:>8.4f}")


# ================================================================
# 7. 主流程（已修改ID输出逻辑）
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="待预测的 CSV")
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "spec", "stellar", "fusion", "all"],
                        help="auto = 根据 ckpt 自动判断; all = 三种模式都跑")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="单 ckpt 路径 (mode=spec/stellar/fusion 必选)")
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="ckpt 目录, mode=fusion/all 时使用 (内含 best_*.pt)")
    parser.add_argument("--id_col", type=str, default=None,
                        help="输入CSV里样本唯一ID列名，输出文件会保留；不传则自动生成 sample_id")
    parser.add_argument("--spectrum_prefix", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--save_metrics", type=str, default=None,
                        help="若提供路径, 把评估指标存为 json")
    args = parser.parse_args()

    # 设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"设备: {device}")

    # ---- 解析 ckpt 路径 ----
    if args.mode in ("fusion", "all"):
        if args.ckpt_dir is None:
            base = os.path.dirname(args.ckpt) if args.ckpt else "."
        else:
            base = args.ckpt_dir
        spec_path    = os.path.join(base, "best_spec.pt")
        stellar_path = os.path.join(base, "best_stellar.pt")
        fusion_path  = os.path.join(base, "best_fusion.pt")
        if not all(os.path.exists(p) for p in [spec_path, stellar_path, fusion_path]):
            raise FileNotFoundError(
                f"mode={args.mode} 需要三个 ckpt: {spec_path}, {stellar_path}, {fusion_path}")
        primary_ckpt = torch.load(fusion_path, map_location=device, weights_only=False)
    elif args.mode == "auto":
        if args.ckpt is None:
            raise ValueError("mode=auto 需要 --ckpt")
        primary_ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        # 推断 mode
        fname = os.path.basename(args.ckpt).lower()
        if "fusion" in fname:
            args.mode = "fusion"
            base = os.path.dirname(args.ckpt) or "."
            spec_path    = primary_ckpt.get("spec_ckpt",    os.path.join(base, "best_spec.pt"))
            stellar_path = primary_ckpt.get("stellar_ckpt", os.path.join(base, "best_stellar.pt"))
            fusion_path  = args.ckpt
        elif "stellar" in fname:
            args.mode = "stellar"
        else:
            args.mode = "spec"
        print(f"自动检测 mode = {args.mode}")
    else:  # spec / stellar 单模型
        if args.ckpt is None:
            raise ValueError(f"mode={args.mode} 需要 --ckpt")
        primary_ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    meta = load_meta(primary_ckpt)
    target_cols = meta["target_cols"]
    print(f"目标列: {target_cols}")
    print(f"  log10 还原索引: {meta['log_target_indices']}")
    print(f"恒星参数列: {meta['stellar_cols']}")

    # ---- 读数据 ----
    df = pd.read_csv(args.data)
    n_sample = len(df)
    print(f"输入数据形状: {df.shape}")
    sequences, stellar, lengths, targets_raw, has_gt = prepare_data(
        df, meta, id_col=args.id_col, spectrum_prefix=args.spectrum_prefix)
    print(f"GT 可用: {has_gt}")

    # ---- 建立模型 ----
    spec_model    = None
    stellar_model = None
    fusion_head   = None

    if args.mode in ("spec", "fusion", "all"):
        spec_model = build_spec_model(meta, device)
        if args.mode == "spec":
            spec_model.load_state_dict(primary_ckpt["model_state"])
        else:
            ckpt_s = torch.load(spec_path, map_location=device, weights_only=False)
            spec_model.load_state_dict(ckpt_s["model_state"])
        spec_model.eval()
        print(f"SpectrumModel 加载完成 ({sum(p.numel() for p in spec_model.parameters()):,} 参数)")

    if args.mode in ("stellar", "fusion", "all"):
        if not meta["stellar_cols"]:
            print("⚠ 训练时无恒星参数列, 无法运行 stellar/fusion 模式")
            if args.mode == "stellar":
                return
        else:
            stellar_model = build_stellar_model(meta, device)
            if args.mode == "stellar":
                stellar_model.load_state_dict(primary_ckpt["model_state"])
            else:
                ckpt_p = torch.load(stellar_path, map_location=device, weights_only=False)
                stellar_model.load_state_dict(ckpt_p["model_state"])
            stellar_model.eval()
            print(f"StellarModel 加载完成 ({sum(p.numel() for p in stellar_model.parameters()):,} 参数)")

    if args.mode in ("fusion", "all") and stellar_model is not None:
        fusion_head = build_fusion_head(meta, device)
        ckpt_f = torch.load(fusion_path, map_location=device, weights_only=False)
        # mode='avg' 可能 state_dict 为空
        try:
            fusion_head.load_state_dict(ckpt_f["model_state"], strict=False)
        except Exception as e:
            print(f"⚠ FusionHead 加载告警: {e}")
        fusion_head.eval()
        print(f"FusionHead 加载完成 (mode={meta['args'].get('fusion_mode')})")

    # ---- 推理 ----
    print("\n开始推理...")
    pred_spec_real    = None
    pred_stellar_real = None
    pred_fusion_real  = None
    ps_scaled, pp_scaled, fs, fp = None, None, None, None

    if spec_model is not None:
        ps_scaled, fs = infer_spec(spec_model, sequences, lengths,
                                    batch_size=args.batch_size, device=device)
        pred_spec_real = inverse_transform_targets(
            ps_scaled, meta["target_scaler_mean"], meta["target_scaler_scale"],
            meta["log_target_indices"])
        print(f"  Spectrum 分支预测完成: {pred_spec_real.shape}")

    if stellar_model is not None:
        pp_scaled, fp = infer_stellar(stellar_model, stellar,
                                       batch_size=args.batch_size, device=device)
        pred_stellar_real = inverse_transform_targets(
            pp_scaled, meta["target_scaler_mean"], meta["target_scaler_scale"],
            meta["log_target_indices"])
        print(f"  Stellar 分支预测完成: {pred_stellar_real.shape}")

    if fusion_head is not None:
        pf_scaled = infer_fusion(fusion_head, ps_scaled, pp_scaled, fs, fp,
                                  batch_size=args.batch_size * 4, device=device)
        pred_fusion_real = inverse_transform_targets(
            pf_scaled, meta["target_scaler_mean"], meta["target_scaler_scale"],
            meta["log_target_indices"])
        print(f"  Fusion 预测完成: {pred_fusion_real.shape}")

    # ---- 评估 (如有 GT) ----
    metrics_all = {}
    if has_gt:
        if pred_spec_real is not None:
            m = compute_metrics(pred_spec_real, targets_raw, target_cols)
            metrics_all["spec"] = m
            print_metrics(m, "Spectrum 分支")
        if pred_stellar_real is not None:
            m = compute_metrics(pred_stellar_real, targets_raw, target_cols)
            metrics_all["stellar"] = m
            print_metrics(m, "Stellar 分支")
        if pred_fusion_real is not None:
            m = compute_metrics(pred_fusion_real, targets_raw, target_cols)
            metrics_all["fusion"] = m
            print_metrics(m, "Fusion 融合")

    # ==================================================
    # 【修改点】输出CSV强制包含样本ID
    # ==================================================
    out_df = pd.DataFrame()

    # 1. 优先使用用户指定的ID列
    if args.id_col and args.id_col in df.columns:
        out_df[args.id_col] = df[args.id_col].values
    else:
        # 2. 未指定ID列 / 列不存在：自动生成自增 sample_id
        out_df["sample_id"] = np.arange(n_sample)

    # 选择主预测结果
    if pred_fusion_real is not None:
        primary = pred_fusion_real
        primary_name = "fusion"
    elif pred_spec_real is not None:
        primary = pred_spec_real
        primary_name = "spec"
    elif pred_stellar_real is not None:
        primary = pred_stellar_real
        primary_name = "stellar"
    else:
        raise RuntimeError("没有可用的预测结果")

    for i, c in enumerate(target_cols):
        out_df[f"{c}_pred"] = primary[:, i]

    # mode=all 时, 把所有分支都附上
    if args.mode == "all":
        if pred_spec_real is not None and primary_name != "spec":
            for i, c in enumerate(target_cols):
                out_df[f"{c}_pred_spec"] = pred_spec_real[:, i]
        if pred_stellar_real is not None and primary_name != "stellar":
            for i, c in enumerate(target_cols):
                out_df[f"{c}_pred_stellar"] = pred_stellar_real[:, i]

    if has_gt:
        for i, c in enumerate(target_cols):
            out_df[f"{c}_true"] = targets_raw[:, i]
            out_df[f"{c}_err"]  = primary[:, i] - targets_raw[:, i]

    out_df.to_csv(args.output, index=False)
    print(f"\n✓ 预测结果已保存: {args.output}  (主结果分支: {primary_name})")
    print(f"  共 {len(out_df)} 行, {len(out_df.columns)} 列")

    if args.save_metrics and metrics_all:
        with open(args.save_metrics, "w") as f:
            json.dump(metrics_all, f, indent=2, default=float)
        print(f"✓ 评估指标已保存: {args.save_metrics}")


if __name__ == "__main__":
    main()
