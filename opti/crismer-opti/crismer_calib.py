# -*- coding: utf-8 -*-
"""
CRISMER-BERT / CRISPR-Transformer — Non-Circular Threshold Selection
=====================================================================
Stripped-down version: NO calibration step on Change-seq + Site-seq. The
model's already-trained weights are used as-is; scores are plain softmax
probabilities at a fixed temperature (THRESHOLD_CV_T), RESCALED to [0, 1]
with a per-fold MinMax transform to undo the compression that a high
temperature introduces (see note below). Everything here operates ONLY on
the four independent evaluation assays (Circle-seq, Guide-seq, Surro-seq,
TTISS) — never on the training data — so the threshold is never justified
using the same samples it's evaluated on.

WHY THE MINMAX STEP: softmax(logits / T) with T=10 divides logits by a
large constant before exponentiating, which shrinks the gaps between
classes and pulls every score toward ~0.5. Left alone, this compression
makes `TARGET_PRECISION` threshold search noisy (all the signal is
squeezed into a tiny slice of [0, 1]) and makes WEIGHT_BIN_EDGES, which
assumes scores actually span [0, 1], meaningless. A MinMax rescale
stretches the compressed scores back out to [0, 1] using only the
min/max observed in that fit's data.

To keep this non-circular, the MinMax scaler is ALWAYS fit only on the
calibration/pool side of a split and then applied (with clipping) to the
held-out side — the holdout never influences its own rescaling. A single
"deployment" scaler, fit once on the full pooled eval set, is saved
alongside the frozen threshold and weight-bin table for use at inference
time on new data.

Two threshold-selection schemes, run independently, each producing its own
frozen threshold, its own fitted deployment MinMax scaler, and its own
confidence-weight bin table:

  (a) run_pooled_sgRNA_kfold_threshold_cv
      Pools all four eval datasets, groups by sgRNA (so a guide's off-target
      sites never span the calibration/holdout split of a fold), and rotates
      POOLED_KFOLD_SPLITS folds (default 8). For each fold: fit a MinMax
      scaler on the calibration portion, rescale both sides with it, pick
      the min rescaled-score threshold hitting TARGET_PRECISION on
      calibration, then check precision on the truly-unseen rescaled
      holdout.

  (b) run_leave_one_dataset_out_threshold_cv
      Pools 3 of the 4 assays to pick a threshold, validates on the 4th
      (an entirely unseen assay technology), rotates through all 4. The
      MinMax scaler for each fold is fit on the pooled 3-assay calibration
      side only and applied to the 4th (holdout) assay. Optionally excludes
      any sgRNA that overlaps between the held-out assay and the
      calibration pool, to avoid a guide informing both sides of the split.

Both report per-fold/per-assay thresholds (stability), the mean/std frozen
threshold, and the precision of that frozen threshold on the pooled
out-of-fold data (all computed in each fold's own rescaled score space).
"""

import os
import random
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
SEED = 12345

base_data_path = "../dataset"
base_model_path = "models/"
results_path = os.path.join(base_model_path, "results")
plots_path = os.path.join(base_model_path, "plots")

MODEL_FILENAME = "change_site_circleseq_model.pth"
MODEL_PATH = os.path.join(base_model_path, MODEL_FILENAME)

MODEL_CONFIG = {
    "num_layers": 2,
    "num_heads": 4,
    "number_hidder_layers": 2,
    "dropout_prob": 0.2,
    "batch_size": 128,
    "epochs": 50,
    "learning_rate": 0.001,
    "pos_weight": 30,
    "attn": False,
    "seq_length": 20,
}

# The four INDEPENDENT evaluation assays. These are the only datasets used
# anywhere in this script — no training/calibration data is touched here.
EVAL_DATASETS = {
    "Circle-seq": os.path.join(base_data_path, "circleseq_all.csv"),
    "Guide-seq": os.path.join(base_data_path, "guideseq.csv"),
    "Surro-seq": os.path.join(base_data_path, "surroseq.csv"),
    "TTISS": os.path.join(base_data_path, "ttiss.csv"),
}

WEIGHT_BIN_EDGES = [0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]

# Fixed softmax temperature used for every score computed in this script.
# NOTE: T=10 compresses softmax(logits / T) toward ~0.5; that's exactly why
# every threshold/weight-table step below rescales with a MinMax fit only
# on the calibration/pool side of its split (see module docstring).
THRESHOLD_CV_T = 10

# Target precision for "min confidence threshold at X% precision".
TARGET_PRECISION = 0.80

# Column used as the sgRNA / guide identifier for grouping folds so that no
# guide's off-target sites are split across the calibration and holdout
# portions of a fold. ASSUMPTION: the "On" (on-target/guide) sequence
# uniquely identifies the sgRNA in these datasets. Change this if you have
# a dedicated sgRNA ID column.
SGRNA_GROUP_COLUMN = "On"

POOLED_KFOLD_SPLITS = 8  # use 8-10 per the plan; adjust as needed

POOLED_KFOLD_WEIGHTS_PATH = os.path.join(base_model_path, "bin_weights_pooled_kfold.pkl")
LODO_WEIGHTS_PATH = os.path.join(base_model_path, "bin_weights_leave_one_dataset_out.pkl")
POOLED_KFOLD_RESULTS_PATH = os.path.join(results_path, "pooled_kfold_threshold_cv.pkl")
LODO_RESULTS_PATH = os.path.join(results_path, "leave_one_dataset_out_threshold_cv.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    """Make the whole pipeline deterministic."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# 2. MODEL ARCHITECTURE
# =============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MultiBranchConv(nn.Module):
    def __init__(self, output_channels=16, attention=True):
        super(MultiBranchConv, self).__init__()

        self.branch1 = nn.Conv2d(in_channels=1, out_channels=output_channels, kernel_size=(1, 16))
        self.branch2 = nn.Conv2d(in_channels=1, out_channels=output_channels, kernel_size=(2, 16))
        self.branch3 = nn.Conv2d(in_channels=1, out_channels=output_channels, kernel_size=(3, 16))
        self.branch4 = nn.Conv2d(in_channels=1, out_channels=output_channels, kernel_size=(4, 16))

        self.attn = attention
        self.ca1 = ChannelAttention(output_channels)
        self.ca2 = ChannelAttention(output_channels)
        self.ca3 = ChannelAttention(output_channels)
        self.ca4 = ChannelAttention(output_channels)
        self.sa1 = SpatialAttention(kernel_size=7)
        self.sa2 = SpatialAttention(kernel_size=7)
        self.sa3 = SpatialAttention(kernel_size=7)
        self.sa4 = SpatialAttention(kernel_size=7)

    def forward(self, x):
        # Branch 1: No padding needed
        out1 = F.relu(self.branch1(x))

        # Branch 2: Pad to the right (end) so shape becomes (bs, 1, 24, 16)
        x_pad2 = F.pad(x, (0, 0, 0, 1))
        out2 = F.relu(self.branch2(x_pad2))

        # Branch 3: Pad one row at the beginning and one at the end (bs, 1, 25, 16)
        x_pad3 = F.pad(x, (0, 0, 1, 1))
        out3 = F.relu(self.branch3(x_pad3))

        # Branch 4: Pad two rows at the beginning and one at the end (bs, 1, 26, 16)
        x_pad4 = F.pad(x, (0, 0, 1, 2))
        out4 = F.relu(self.branch4(x_pad4))

        if self.attn:
            out1 = out1 * self.ca1(out1)
            out1 = out1 * self.sa1(out1)

            out2 = out2 * self.ca2(out2)
            out2 = out2 * self.sa2(out2)

            out3 = out3 * self.ca3(out3)
            out3 = out3 * self.sa3(out3)

            out4 = out4 * self.ca4(out4)
            out4 = out4 * self.sa4(out4)

        # Remove last dimension of size 1 (from Conv2D)
        out1 = out1.squeeze(-1)
        out2 = out2.squeeze(-1)
        out3 = out3.squeeze(-1)
        out4 = out4.squeeze(-1)

        # Transpose to shape (bs, 23, 16) for concatenation later
        out1 = out1.transpose(1, 2)
        out2 = out2.transpose(1, 2)
        out3 = out3.transpose(1, 2)
        out4 = out4.transpose(1, 2)

        # Concatenate along the last dimension to get shape (bs, 23, 64)
        output = torch.cat((out1, out2, out3, out4), dim=-1)

        return output


class CRISPRTransformerModel(nn.Module):
    def __init__(self, config):
        super(CRISPRTransformerModel, self).__init__()

        self.input_dim = 64
        self.num_layers = config.get("num_layers", 2)
        self.num_heads = config.get("num_heads", 8)
        self.dropout_prob = config["dropout_prob"]
        self.number_hidden_layers = config["number_hidder_layers"]
        self.seq_length = config.get("seq_length", 23)

        self.pos_encoder = nn.Parameter(torch.randn(1, self.seq_length, self.input_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=self.num_heads,
            dim_feedforward=self.input_dim * 4,
            dropout=self.dropout_prob,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.conv = MultiBranchConv(attention=config["attn"])

        self.hidden_layers = []
        start_size = self.seq_length * self.input_dim
        for _ in range(self.number_hidden_layers):
            layer = nn.Sequential(
                nn.Linear(start_size, start_size // 2),
                nn.GELU(),
                nn.Dropout(self.dropout_prob),
            )
            self.hidden_layers.append(layer)
            start_size = start_size // 2
        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        self.output = nn.Linear(start_size, 2)

    def forward(self, x, src_mask=None):
        x = self.conv(x)
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.view(x.size(0), -1)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output(x)
        return x


def load_model(model_path=MODEL_PATH, config=MODEL_CONFIG):
    """Instantiate the model, load trained weights, and move it to `device`."""
    model = CRISPRTransformerModel(config)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    return model


# =============================================================================
# 3. DATASET / INFERENCE HELPERS
# =============================================================================
class TrainerDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(np.asarray(inputs), dtype=torch.float32).unsqueeze(1)
        self.targets = torch.tensor(np.asarray(targets), dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def tester(model, test_x, test_y, batch_size=128):
    test_dataset = TrainerDataset(test_x, test_y)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    results = []
    true_labels = []
    with torch.no_grad():
        for test_features, test_labels in test_dataloader:
            outputs = model(test_features.to(device)).detach().to("cpu")
            results.extend(outputs)
            true_labels.extend(test_labels)
    return true_labels, results


def get_confidence_scores(model, test_x, test_y, T=THRESHOLD_CV_T):
    """
    Runs inference and returns RAW softmax(logits / T)[:, 1] as the
    confidence score for the "active" class. No rescaling happens here —
    this function stays a pure model-forward-pass. Because T is large,
    these raw scores are compressed toward ~0.5; the MinMax rescaling that
    corrects for that is applied downstream, per-fold, on calibration data
    only (see `_fit_minmax` / the CV functions below), never here, so this
    function can't accidentally leak holdout information into a scaler.
    """
    _, results = tester(model, test_x, test_y)
    predictions = [torch.nn.functional.softmax(r / T, dim=0) for r in results]
    scores = np.array([p[1].item() for p in predictions])
    return scores


def one_hot_features(df):
    """Builds (n_samples, 20, 16) pairwise one-hot features from On/Off columns."""
    nucleotides = ["A", "T", "G", "C"]
    pairs = [f"{n1}{n2}" for n1 in nucleotides for n2 in nucleotides]
    pair_to_idx = {p: i for i, p in enumerate(pairs)}

    pairwise_features = np.zeros((len(df), 20, len(pairs)))

    for idx, row in df.reset_index(drop=True).iterrows():
        on_seq = row["On"]
        off_seq = row["Off"]

        for pos in range(min(20, len(on_seq), len(off_seq))):
            pair = on_seq[pos] + off_seq[pos]
            if pair in pair_to_idx:
                pairwise_features[idx, pos, pair_to_idx[pair]] = 1

    return pairwise_features[:, :20, :]


# =============================================================================
# 4. MINMAX RESCALING HELPER (undoes high-T softmax compression, per-fold)
# =============================================================================
def fit_minmax_on_calibration(calib_scores):
    """
    Fits a MinMaxScaler using ONLY calibration/pool scores. This is the one
    and only place a scaler is ever fit in this script, and it never sees
    holdout data — keeping the rescale step as non-circular as the
    threshold search itself.
    """
    scaler = MinMaxScaler()
    scaler.fit(np.asarray(calib_scores, dtype=float).reshape(-1, 1))
    return scaler


def apply_minmax(scaler, scores):
    """
    Applies an already-fit MinMaxScaler to `scores` and clips to [0, 1].
    Clipping matters because a holdout fold can contain scores outside the
    calibration fold's observed min/max; without clipping those samples
    would land outside [0, 1] and break WEIGHT_BIN_EDGES / precision logic
    downstream.
    """
    scaled = scaler.transform(np.asarray(scores, dtype=float).reshape(-1, 1)).ravel()
    return np.clip(scaled, 0.0, 1.0)


# =============================================================================
# 5. WEIGHT-BIN HELPER (used to save the two confidence-weight tables)
# =============================================================================
def calculate_weights(scores, test_y, bins):
    """
    Calculate weights (active ratios) for a binary classification problem.

    Args:
        scores: Predicted/scaled scores (expected in/near [0, 1]).
        test_y: True binary labels.
        bins (list): Bin edges for score ranges.

    Returns:
        pd.DataFrame: bin ranges, counts, and active ratios.
    """
    data = pd.DataFrame({"score": np.asarray(scores), "Active": np.asarray(test_y)})
    data["bin"] = pd.cut(data["score"], bins=bins, include_lowest=True)

    def _agg(x):
        return pd.Series(
            {
                "active_count": (x["Active"] == 1).sum(),
                "inactive_count": (x["Active"] == 0).sum(),
                "total_count": len(x),
            }
        )

    grouped = data.groupby("bin", observed=False)
    try:
        bin_stats = grouped.apply(_agg, include_groups=False).reset_index()
    except TypeError:
        bin_stats = grouped.apply(_agg).reset_index()

    bin_stats["active_ratio"] = bin_stats["active_count"] / bin_stats["total_count"]
    bin_stats["active_ratio"] = bin_stats["active_ratio"].fillna(0)

    return bin_stats


# =============================================================================
# 6. DATA LOADING
# =============================================================================
def load_single(path):
    """Reads a single dataset CSV, keeping On/Off/Active and dropping
    duplicate rows within that dataset."""
    df = pd.read_csv(path)[["On", "Off", "Active"]]
    df = df.drop_duplicates().reset_index(drop=True)
    return df


# =============================================================================
# 7. THRESHOLD-STABILITY CROSS-VALIDATION  (fixes the circularity comment)
# =============================================================================
# Both schemes below operate ONLY on the four independent EVAL_DATASETS
# (Circle-seq, Guide-seq, Surro-seq, TTISS) — no training/calibration data
# is used anywhere — and never use the same samples to both pick and
# validate a threshold, or to both fit and apply a MinMax rescale.


def find_threshold_for_precision(scores, true_y, target_precision=TARGET_PRECISION):
    """
    Finds the smallest score threshold such that predictions with
    score >= threshold achieve at least `target_precision`.

    Taking the SMALLEST qualifying threshold maximizes recall/sensitivity
    subject to the precision floor. Uses sklearn's precision_recall_curve so
    precision is computed cumulatively (score >= threshold), which is far
    less noisy than histogram binning on small fold sizes.

    Returns:
        float threshold, or None if no threshold reaches target_precision.
    """
    scores = np.asarray(scores, dtype=float)
    true_y = np.asarray(true_y, dtype=int)

    if len(np.unique(true_y)) < 2:
        return None

    precision, recall, thresholds = precision_recall_curve(true_y, scores)
    # precision/recall have one more element than thresholds (the last entry
    # corresponds to threshold = +inf, recall = 0); drop it so precision[i]
    # aligns with thresholds[i].
    precision_aligned = precision[:-1]

    valid_idx = np.where(precision_aligned >= target_precision)[0]
    if len(valid_idx) == 0:
        return None

    best_idx = valid_idx[np.argmin(thresholds[valid_idx])]
    return float(thresholds[best_idx])


def precision_at_threshold(scores, true_y, threshold):
    """Empirical precision among samples with score >= threshold.
    Returns np.nan if threshold is None or no sample clears it."""
    if threshold is None:
        return np.nan
    scores = np.asarray(scores, dtype=float)
    true_y = np.asarray(true_y, dtype=int)
    predicted_positive = scores >= threshold
    if predicted_positive.sum() == 0:
        return np.nan
    return float(true_y[predicted_positive].mean())


def _score_eval_datasets(model, T=THRESHOLD_CV_T):
    """Loads each of the four EVAL_DATASETS, scores them (RAW, unscaled) at
    temperature T, and returns a dict keyed by dataset name ->
    {"scores", "true_y", "groups"}.

    `groups` is the sgRNA id (SGRNA_GROUP_COLUMN) used to prevent a guide's
    sites from being split across calibration/holdout in the pooled scheme.

    Scores here are intentionally left un-rescaled: MinMax fitting happens
    per-fold, downstream, on calibration data only.
    """
    per_dataset = {}
    for name, path in EVAL_DATASETS.items():
        df = load_single(path)
        test_x = one_hot_features(df)
        test_y = df["Active"].to_numpy()
        scores = get_confidence_scores(model, test_x, test_y, T=T)
        per_dataset[name] = {
            "scores": np.asarray(scores),
            "true_y": test_y,
            "groups": df[SGRNA_GROUP_COLUMN].to_numpy(),
        }
    return per_dataset


def plot_threshold_stability(labels, thresholds, title, save_path=None):
    """Simple bar chart of per-fold thresholds, for a supplementary figure
    showing threshold stability across folds/datasets."""
    plt.figure(figsize=(6, 4))
    x = np.arange(len(labels))
    plt.bar(x, thresholds, edgecolor="k", alpha=0.75)
    valid = [t for t in thresholds if t is not None and not np.isnan(t)]
    if valid:
        mean_t = np.mean(valid)
        plt.axhline(mean_t, color="red", linestyle="--", label=f"mean = {mean_t:.3f}")
        plt.legend()
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Selected threshold (MinMax-rescaled)")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


def run_pooled_sgRNA_kfold_threshold_cv(
    model,
    k=POOLED_KFOLD_SPLITS,
    T=THRESHOLD_CV_T,
    target_precision=TARGET_PRECISION,
):
    """
    Scheme (a): Pool Circle-seq + Guide-seq + Surro-seq + TTISS, group by
    sgRNA (SGRNA_GROUP_COLUMN) so a guide's sites never span the
    calibration/holdout split, and rotate `k` folds (8-10 recommended).

    For each fold:
      - Raw scores on the (k-1)/k calibration portion are used to fit a
        MinMax scaler (undoing the high-T softmax compression), which is
        then applied to BOTH the calibration scores and the held-out 1/k
        holdout scores (clipped to [0, 1]).
      - The minimum rescaled threshold hitting `target_precision` is
        picked on the rescaled calibration portion.
      - The rescaled held-out portion (never used to fit that fold's
        scaler or pick that fold's threshold) is used to check empirical
        precision at that threshold.

    Reports per-fold thresholds (stability), the mean/std frozen threshold,
    and the precision of the mean frozen threshold evaluated on the pooled
    out-of-fold data (each in its own fold's rescaled space). At the end, a
    single deployment MinMax scaler is fit on the FULL pooled raw score set
    and saved alongside the frozen threshold and weight-bin table, for use
    on new data going forward.
    """
    per_dataset = _score_eval_datasets(model, T=T)

    dataset_col, scores_list, y_list, group_list = [], [], [], []
    for name, d in per_dataset.items():
        n = len(d["scores"])
        dataset_col.extend([name] * n)
        scores_list.append(d["scores"])
        y_list.append(d["true_y"])
        group_list.append(d["groups"])

    scores = np.concatenate(scores_list)  # RAW scores, pre-rescale
    true_y = np.concatenate(y_list)
    groups = np.concatenate(group_list)
    dataset_col = np.array(dataset_col)

    n_unique_groups = len(np.unique(groups))
    if n_unique_groups < k:
        print(
            f"[pooled_kfold] WARNING: only {n_unique_groups} unique sgRNA groups "
            f"but k={k} folds requested. Reducing k to {n_unique_groups}."
        )
        k = max(2, n_unique_groups)

    gkf = GroupKFold(n_splits=k)

    fold_records = []
    oof_scores_scaled = np.full_like(scores, fill_value=np.nan, dtype=float)

    for fold_idx, (calib_idx, holdout_idx) in enumerate(gkf.split(scores, true_y, groups=groups)):
        calib_scores_raw, calib_y = scores[calib_idx], true_y[calib_idx]
        holdout_scores_raw, holdout_y = scores[holdout_idx], true_y[holdout_idx]

        # Fit MinMax on calibration ONLY, then apply to both sides.
        fold_scaler = fit_minmax_on_calibration(calib_scores_raw)
        calib_scores = apply_minmax(fold_scaler, calib_scores_raw)
        holdout_scores = apply_minmax(fold_scaler, holdout_scores_raw)

        thr = find_threshold_for_precision(calib_scores, calib_y, target_precision)
        holdout_precision = precision_at_threshold(holdout_scores, holdout_y, thr)

        fold_records.append(
            {
                "fold": fold_idx,
                "threshold": thr,
                "scaler_calib_min": float(fold_scaler.data_min_[0]),
                "scaler_calib_max": float(fold_scaler.data_max_[0]),
                "n_calibration_sites": len(calib_idx),
                "n_holdout_sites": len(holdout_idx),
                "n_holdout_groups": len(np.unique(groups[holdout_idx])),
                "holdout_precision_at_threshold": holdout_precision,
            }
        )

        oof_scores_scaled[holdout_idx] = holdout_scores

        print(
            f"[pooled_kfold] fold {fold_idx}: threshold={thr} "
            f"(calib raw range [{fold_scaler.data_min_[0]:.4f}, {fold_scaler.data_max_[0]:.4f}]), "
            f"holdout_precision={holdout_precision}"
        )

    fold_df = pd.DataFrame(fold_records)
    valid_thresholds = fold_df["threshold"].dropna().to_numpy(dtype=float)
    mean_threshold = float(np.mean(valid_thresholds)) if len(valid_thresholds) else None
    std_threshold = float(np.std(valid_thresholds)) if len(valid_thresholds) else None

    # Overall out-of-fold precision, computed entirely in each sample's own
    # fold-rescaled space (every sample appears in exactly one holdout fold).
    overall_precision = precision_at_threshold(oof_scores_scaled, true_y, mean_threshold)

    print("\n[pooled_kfold] Per-fold thresholds:", fold_df["threshold"].tolist())
    print(f"[pooled_kfold] Mean threshold = {mean_threshold}, std = {std_threshold}")
    print(f"[pooled_kfold] Frozen mean threshold precision (all pooled, out-of-fold union) = {overall_precision}")

    plot_threshold_stability(
        [f"fold {i}" for i in fold_df["fold"]],
        fold_df["threshold"].tolist(),
        title=f"Pooled sgRNA-grouped {k}-fold threshold stability (target precision={target_precision})",
        save_path=os.path.join(plots_path, "pooled_kfold_threshold_stability.png"),
    )

    # Deployment scaler: fit once on the FULL pooled raw score set (this is
    # not a "fold" being evaluated — it's the transform saved for scoring
    # brand-new data at inference time) and use it to build the weight table.
    deployment_scaler = fit_minmax_on_calibration(scores)
    scores_deployment_scaled = apply_minmax(deployment_scaler, scores)

    weight_table = calculate_weights(scores_deployment_scaled, true_y, WEIGHT_BIN_EDGES)
    os.makedirs(base_model_path, exist_ok=True)
    with open(POOLED_KFOLD_WEIGHTS_PATH, "wb") as f:
        pickle.dump([WEIGHT_BIN_EDGES, weight_table["active_ratio"], deployment_scaler], f)
    print(f"[pooled_kfold] Saved confidence-weight bins + deployment scaler -> {POOLED_KFOLD_WEIGHTS_PATH}")

    summary = {
        "scheme": "pooled_sgRNA_grouped_kfold",
        "k": k,
        "T": T,
        "target_precision": target_precision,
        "fold_table": fold_df,
        "mean_threshold": mean_threshold,
        "std_threshold": std_threshold,
        "frozen_threshold_overall_precision": overall_precision,
        "weight_table": weight_table,
        "deployment_minmax_scaler": deployment_scaler,
        "dataset_composition": pd.Series(dataset_col).value_counts().to_dict(),
    }

    os.makedirs(results_path, exist_ok=True)
    with open(POOLED_KFOLD_RESULTS_PATH, "wb") as f:
        pickle.dump(summary, f)
    print(f"[pooled_kfold] Saved full results -> {POOLED_KFOLD_RESULTS_PATH}")

    return summary


def run_leave_one_dataset_out_threshold_cv(
    model,
    T=THRESHOLD_CV_T,
    target_precision=TARGET_PRECISION,
    exclude_overlapping_sgRNAs=True,
):
    """
    Scheme (b): Leave-one-dataset-out CV across the four independent assays.

    For each of Circle-seq / Guide-seq / Surro-seq / TTISS:
      - Pool the OTHER three datasets' raw scores, fit a MinMax scaler on
        that pool ONLY, and rescale both the pool and the held-out dataset
        with it (clipped to [0, 1]).
      - Find the minimum rescaled-score threshold hitting `target_precision`
        on the rescaled pool.
      - Validate that threshold on the rescaled held-out dataset (an entire
        unseen assay technology the threshold — and the scaler — never saw).

    If `exclude_overlapping_sgRNAs` is True, any sgRNA present in BOTH the
    held-out dataset and the calibration pool is dropped from the
    calibration pool for that fold (before fitting the scaler or the
    threshold), to avoid the same guide informing both sides of the split.

    Reports the 4 per-assay thresholds (stability across assays), mean/std,
    and precision of the frozen mean threshold on the full pooled set
    (rescaled with a deployment scaler fit on all four assays). Saves a
    dedicated confidence-weight bin table plus that deployment scaler.
    """
    per_dataset = _score_eval_datasets(model, T=T)
    names = list(per_dataset.keys())

    fold_records = []
    holdout_scores_scaled_by_name = {}

    for held_out in names:
        calib_names = [n for n in names if n != held_out]

        calib_scores_parts, calib_y_parts = [], []
        holdout_groups = set(per_dataset[held_out]["groups"].tolist())

        n_excluded = 0
        for n in calib_names:
            d = per_dataset[n]
            if exclude_overlapping_sgRNAs:
                mask = ~np.isin(d["groups"], list(holdout_groups))
                n_excluded += int((~mask).sum())
            else:
                mask = np.ones(len(d["scores"]), dtype=bool)
            calib_scores_parts.append(d["scores"][mask])
            calib_y_parts.append(d["true_y"][mask])

        calib_scores_raw = np.concatenate(calib_scores_parts)
        calib_y = np.concatenate(calib_y_parts)

        holdout_scores_raw = per_dataset[held_out]["scores"]
        holdout_y = per_dataset[held_out]["true_y"]

        # Fit MinMax on the pooled calibration side ONLY.
        fold_scaler = fit_minmax_on_calibration(calib_scores_raw)
        calib_scores = apply_minmax(fold_scaler, calib_scores_raw)
        holdout_scores = apply_minmax(fold_scaler, holdout_scores_raw)
        holdout_scores_scaled_by_name[held_out] = holdout_scores

        thr = find_threshold_for_precision(calib_scores, calib_y, target_precision)
        holdout_precision = precision_at_threshold(holdout_scores, holdout_y, thr)

        fold_records.append(
            {
                "held_out_dataset": held_out,
                "calibration_datasets": ",".join(calib_names),
                "threshold": thr,
                "scaler_calib_min": float(fold_scaler.data_min_[0]),
                "scaler_calib_max": float(fold_scaler.data_max_[0]),
                "n_calibration_sites": len(calib_y),
                "n_excluded_overlapping_sgRNA_sites": n_excluded,
                "n_holdout_sites": len(holdout_y),
                "holdout_precision_at_threshold": holdout_precision,
            }
        )

        print(
            f"[leave_one_out] held out {held_out}: threshold={thr} "
            f"(calib raw range [{fold_scaler.data_min_[0]:.4f}, {fold_scaler.data_max_[0]:.4f}]), "
            f"holdout_precision={holdout_precision}, excluded_overlap_sites={n_excluded}"
        )

    fold_df = pd.DataFrame(fold_records)
    valid_thresholds = fold_df["threshold"].dropna().to_numpy(dtype=float)
    mean_threshold = float(np.mean(valid_thresholds)) if len(valid_thresholds) else None
    std_threshold = float(np.std(valid_thresholds)) if len(valid_thresholds) else None

    # Overall precision: each assay's holdout scores, each rescaled by that
    # assay's own fold scaler (fit on the other three), pooled together.
    all_scores_scaled = np.concatenate([holdout_scores_scaled_by_name[n] for n in names])
    all_y = np.concatenate([per_dataset[n]["true_y"] for n in names])
    overall_precision = precision_at_threshold(all_scores_scaled, all_y, mean_threshold)

    print("\n[leave_one_out] Per-assay thresholds:", fold_df["threshold"].tolist())
    print(f"[leave_one_out] Mean threshold = {mean_threshold}, std = {std_threshold}")
    print(f"[leave_one_out] Frozen mean threshold precision (all 4 datasets pooled) = {overall_precision}")

    plot_threshold_stability(
        fold_df["held_out_dataset"].tolist(),
        fold_df["threshold"].tolist(),
        title=f"Leave-one-dataset-out threshold stability (target precision={target_precision})",
        save_path=os.path.join(plots_path, "leave_one_dataset_out_threshold_stability.png"),
    )

    # Deployment scaler: fit once on ALL FOUR pooled raw scores (again, this
    # is the saved transform for future/new data, not a held-out fold).
    all_scores_raw = np.concatenate([per_dataset[n]["scores"] for n in names])
    deployment_scaler = fit_minmax_on_calibration(all_scores_raw)
    all_scores_deployment_scaled = apply_minmax(deployment_scaler, all_scores_raw)

    weight_table = calculate_weights(all_scores_deployment_scaled, all_y, WEIGHT_BIN_EDGES)
    os.makedirs(base_model_path, exist_ok=True)
    with open(LODO_WEIGHTS_PATH, "wb") as f:
        pickle.dump([WEIGHT_BIN_EDGES, weight_table["active_ratio"], deployment_scaler], f)
    print(f"[leave_one_out] Saved confidence-weight bins + deployment scaler -> {LODO_WEIGHTS_PATH}")

    summary = {
        "scheme": "leave_one_dataset_out",
        "T": T,
        "target_precision": target_precision,
        "fold_table": fold_df,
        "mean_threshold": mean_threshold,
        "std_threshold": std_threshold,
        "frozen_threshold_overall_precision": overall_precision,
        "weight_table": weight_table,
        "deployment_minmax_scaler": deployment_scaler,
    }

    os.makedirs(results_path, exist_ok=True)
    with open(LODO_RESULTS_PATH, "wb") as f:
        pickle.dump(summary, f)
    print(f"[leave_one_out] Saved full results -> {LODO_RESULTS_PATH}")

    return summary


def run_threshold_cv_experiments(model):
    """Runs BOTH threshold-stability CV schemes and prints a short combined
    comparison at the end (do the two independently-derived thresholds
    agree? -- note both are on their own MinMax-rescaled [0, 1] score
    space, so the numbers are directly comparable to each other)."""
    pooled_summary = run_pooled_sgRNA_kfold_threshold_cv(model)
    lodo_summary = run_leave_one_dataset_out_threshold_cv(model)

    print("\n================ Threshold CV comparison ================")
    if pooled_summary["mean_threshold"] is not None:
        print(
            f"Pooled {pooled_summary['k']}-fold sgRNA-grouped:  "
            f"mean threshold = {pooled_summary['mean_threshold']:.4f} "
            f"(+/- {pooled_summary['std_threshold']:.4f})"
        )
    else:
        print("Pooled k-fold: no valid threshold found")

    if lodo_summary["mean_threshold"] is not None:
        print(
            f"Leave-one-dataset-out (4-fold):        "
            f"mean threshold = {lodo_summary['mean_threshold']:.4f} "
            f"(+/- {lodo_summary['std_threshold']:.4f})"
        )
    else:
        print("Leave-one-dataset-out: no valid threshold found")
    print("===========================================================")

    return {"pooled_kfold": pooled_summary, "leave_one_dataset_out": lodo_summary}


# =============================================================================
# 8. MAIN
# =============================================================================
def main():
    set_seed()
    os.makedirs(base_model_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    print(device)

    model = load_model()

    run_threshold_cv_experiments(model)


if __name__ == "__main__":
    main()