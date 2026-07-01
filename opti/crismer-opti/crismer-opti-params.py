# -*- coding: utf-8 -*-
"""
CRISMER-BERT / CRISPR-Transformer — Score Calibration & Active-Ratio Evaluation
================================================================================
Reorganized from `crismer-opti-params.ipynb`.

Pipeline
--------
1. Load the trained CRISPR Transformer model.
2. CALIBRATION step — combine Change-seq + Site-seq into `changeseq_siteseq`,
   fit the MinMax score scaler on it (T = 10), draw its active-ratio graph,
   and compute the confidence-weight bins used downstream.
3. EVALUATION step — for Circle-seq, Guide-seq, Surro-seq and TTISS
   (evaluated independently, i.e. NOT combined with each other or with the
   calibration set), reuse the already-fitted scaler to score each dataset
   at T = 1, 5 and 10, drawing one active-ratio graph per (dataset, T).
4. Every artifact needed to regenerate a graph later (raw scores, labels,
   bin centers, active ratios) is pickled to disk — no need to rerun
   inference just to redraw a plot.
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

import joblib
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    accuracy_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
SEED = 12345

# Root folder that holds the raw dataset CSVs.
base_data_path = "../dataset"

# Root folder that holds model weights, scalers, and all pickled artifacts.
base_model_path = "models/"

# Sub-folder (inside base_model_path) where every graph's underlying data is
# stashed, so plots can be regenerated later without rerunning inference.
graph_data_path = os.path.join(base_model_path, "graph_data")
plots_path = os.path.join(base_model_path, "plots")

MODEL_FILENAME = "change_site_circleseq_model.pth"
MODEL_PATH = os.path.join(base_model_path, MODEL_FILENAME)
SCALER_PATH = os.path.join(base_model_path, "minmax_scaler.pkl")
WEIGHTS_PATH = os.path.join(base_model_path, "bin_weights.pkl")

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

# Datasets combined ONLY to fit the scaler + calibrate the confidence-weight
# bins (T = 10). Kept separate from the four independently-evaluated sets.
CALIBRATION_DATASETS = {
    "Change-seq": os.path.join(base_data_path, "changeseq_siteseq.csv"),
}
CALIBRATION_NAME = "changeseq_siteseq"
CALIBRATION_T_VALUES = [1, 5, 10]
CALIBRATION_WEIGHT_T = 10  # bin weights are only computed/saved for this T

# Datasets evaluated independently (NOT combined together). One graph per
# (dataset, T) combination is produced for each of these.
EVAL_DATASETS = {
    "Circle-seq": os.path.join(base_data_path, "circleseq_all.csv"),
    "Guide-seq": os.path.join(base_data_path, "guideseq.csv"),
    "Surro-seq": os.path.join(base_data_path, "surroseq.csv"),
    "TTISS": os.path.join(base_data_path, "ttiss.csv"),
}
EVAL_T_VALUES = [1, 5, 10]

NUM_BINS = 50
WEIGHT_BIN_EDGES = [0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]

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
        self.inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1)
        self.targets = torch.tensor(targets, dtype=torch.long)

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


class Stats:
    def __init__(self):
        self.acc = 0
        self.pre = 0
        self.re = 0
        self.f1 = 0
        self.roc = 0
        self.prc = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0
        self.scores = []
        self.results = []

    def print(self):
        print("Accuracy: %.4f" % self.acc)
        print("Precision: %.4f" % self.pre)
        print("Recall: %.4f" % self.re)
        print("F1 Score: %.4f" % self.f1)
        print("ROC: %.4f" % self.roc)
        print("PR AUC: %.4f" % self.prc)
        print("Confusion Matrix")
        print(self.tn, "\t", self.fp)
        print(self.fn, "\t", self.tp)


def eval_matrices(model, test_x, test_y, debug=True, scaler="minmax"):
    true_y, results = tester(model, test_x, test_y)

    predictions = [torch.nn.functional.softmax(r, dim=0) for r in results]
    pred_y = np.array([y[1].item() for y in predictions])
    test_y = np.array([y.item() for y in true_y])

    raw_logits = np.array([r[1].item() for r in results])

    if scaler == "minmax":
        scaled_scores = MinMaxScaler().fit_transform(raw_logits.reshape(-1, 1)).flatten()
    elif scaler == "standard":
        scaled_scores = StandardScaler().fit_transform(raw_logits.reshape(-1, 1)).flatten()
    else:
        scaled_scores = raw_logits

    threshold = 0.5
    pred_y_list = (pred_y > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(test_y, pred_y_list).ravel()

    precision, recall, _ = precision_recall_curve(test_y, pred_y)
    auc_score = auc(recall, precision)

    acc = accuracy_score(test_y, pred_y_list)
    pr = tp / (tp + fp) if (tp + fp) > 0 else -1
    re = tp / (tp + fn) if (tp + fn) > 0 else -1
    f1 = (2 * pr * re / (pr + re)) if (pr + re) > 0 else -1

    stats = Stats()
    stats.acc = acc
    stats.pre = pr
    stats.re = re
    stats.f1 = f1
    stats.roc = roc_auc_score(test_y, raw_logits)
    stats.prc = auc_score
    stats.tn = tn
    stats.fp = fp
    stats.fn = fn
    stats.tp = tp
    stats.scores = scaled_scores
    stats.results = results

    if debug:
        print("Accuracy: %.4f" % acc)
        print("Precision: %.4f" % pr)
        print("Recall: %.4f" % re)
        print("F1 Score: %.4f" % f1)
        print("ROC AUC: %.4f" % stats.roc)
        print("PR AUC: %.4f" % auc_score)

    return stats


def getScore(model, test_x, test_y, T=10, scaler="softmax", fit_scaler=False, scaler_path=SCALER_PATH):
    """
    Computes scaled confidence scores from a model.

    The MinMax scaler is fit exactly ONCE, on the calibration dataset
    (Change-seq + Site-seq), and reused (transform-only) for every other
    dataset so that all scores live on the same scale.

    Args:
        model: The trained model.
        test_x: Test input features.
        test_y: Test labels (binary).
        T (float): Softmax temperature.
        scaler (str): 'minmax', 'standard', 'softmax', or None.
        fit_scaler (bool): If True, fit a new MinMaxScaler and persist it to
            `scaler_path`. If False, load the previously-fit scaler from
            `scaler_path` and only transform.
        scaler_path (str): Where to save/load the MinMaxScaler.

    Returns:
        np.array: Scaled scores.
    """
    true_y, results = tester(model, test_x, test_y)
    raw_logits = np.array([r[1].item() for r in results])

    if scaler == "minmax":
        if fit_scaler:
            scaler_instance = MinMaxScaler()
            scaled_scores = scaler_instance.fit_transform(raw_logits.reshape(-1, 1)).flatten()
            joblib.dump(scaler_instance, scaler_path)
            print("MinMaxScaler parameters:", scaler_instance.min_, scaler_instance.scale_)
        else:
            scaler_instance = joblib.load(scaler_path)
            scaled_scores = scaler_instance.transform(raw_logits.reshape(-1, 1)).flatten()

    elif scaler == "standard":
        scaled_scores = StandardScaler().fit_transform(raw_logits.reshape(-1, 1)).flatten()

    elif scaler == "softmax":
        predictions = [torch.nn.functional.softmax(r / T, dim=0) for r in results]
        raw_scaled = np.array([y[1].item() for y in predictions])

        if fit_scaler:
            scaler_instance = MinMaxScaler()
            scaled_scores = scaler_instance.fit_transform(raw_scaled.reshape(-1, 1)).flatten()
            joblib.dump(scaler_instance, scaler_path)
            print("MinMaxScaler parameters:", scaler_instance.min_, scaler_instance.scale_)
        else:
            scaler_instance = joblib.load(scaler_path)
            scaled_scores = scaler_instance.transform(raw_scaled.reshape(-1, 1)).flatten()

    else:
        scaled_scores = raw_logits

    return scaled_scores


def one_hot_features(df):
    """Builds (n_samples, 20, 16) pairwise one-hot features from On/Off columns."""
    nucleotides = ["A", "T", "G", "C"]
    pairs = [f"{n1}{n2}" for n1 in nucleotides for n2 in nucleotides]

    pairwise_features = np.zeros((len(df), 20, len(pairs)))

    for idx, row in df.reset_index(drop=True).iterrows():
        on_seq = row["On"]
        off_seq = row["Off"]

        for pos in range(20):
            pair = on_seq[pos] + off_seq[pos]
            if pair in pairs:
                pair_idx = pairs.index(pair)
                pairwise_features[idx, pos, pair_idx] = 1

    return pairwise_features[:, :20, :]


# =============================================================================
# 4. GRAPHING & WEIGHT-CALIBRATION HELPERS
# =============================================================================
def compute_active_ratio_bins(scores, true_y, num_bins=NUM_BINS):
    """Bins scores into `num_bins` equal-width bins in [0, 1] and computes the
    fraction of true positives ("active ratio") in each bin."""
    scores = np.array(scores)
    true_y = np.array(true_y)

    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(scores, bins, right=True)

    bin_counts = np.zeros(num_bins)
    bin_positives = np.zeros(num_bins)
    for i in range(num_bins):
        bin_mask = bin_indices == i + 1
        bin_counts[i] = np.sum(bin_mask)
        bin_positives[i] = np.sum(true_y[bin_mask])

    active_ratios = np.divide(
        bin_positives, bin_counts, where=bin_counts > 0, out=np.zeros_like(bin_counts)
    )
    bin_centers = (bins[:-1] + bins[1:]) / 2

    return bin_centers, active_ratios


def plot_active_ratio(bin_centers, active_ratios, num_bins=NUM_BINS, title=None, save_path=None):
    """Renders the active-ratio bar chart. Kept separate from the score
    computation so a graph can be redrawn later purely from saved data."""
    plt.figure(figsize=(5, 4))
    plt.bar(bin_centers, active_ratios, width=1.0 / num_bins, edgecolor="k", alpha=0.7)
    plt.xlabel("Score Ranges")
    plt.ylabel("ratio of active off-targets")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    if title:
        plt.title(title)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()
    plt.close()


def graphActiveRatio(scores, true_y, num_bins=NUM_BINS, title=None, save_path=None):
    """Computes and plots the active-ratio bar chart in one call.
    Returns (bin_centers, active_ratios) for persistence."""
    bin_centers, active_ratios = compute_active_ratio_bins(scores, true_y, num_bins)
    plot_active_ratio(bin_centers, active_ratios, num_bins, title=title, save_path=save_path)
    return bin_centers, active_ratios


def calculate_weights(scores, test_y, bins):
    """
    Calculate weights (active ratios) for a binary classification problem.

    Args:
        scores: Predicted/scaled scores.
        test_y: True binary labels.
        bins (list): Bin edges for score ranges.

    Returns:
        pd.DataFrame: bin ranges, counts, and active ratios.
    """
    data = pd.DataFrame({"score": scores, "Active": test_y})
    data["bin"] = pd.cut(data["score"], bins=bins, include_lowest=True)

    bin_stats = (
        data.groupby("bin", observed=False)
        .apply(
            lambda x: pd.Series(
                {
                    "active_count": (x["Active"] == 1).sum(),
                    "inactive_count": (x["Active"] == 0).sum(),
                    "total_count": len(x),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    bin_stats["active_ratio"] = bin_stats["active_count"] / bin_stats["total_count"]
    bin_stats["active_ratio"] = bin_stats["active_ratio"].fillna(0)

    return bin_stats


# =============================================================================
# 5. DATA LOADING HELPERS
# =============================================================================
def _load_csv_subset(path):
    df = pd.read_csv(path)
    return df[["On", "Off", "Active"]]


def load_and_combine(dataset_paths):
    """Reads and concatenates multiple CSVs, keeping On/Off/Active and
    dropping duplicate rows across the combined set."""
    dataframes = [_load_csv_subset(path) for path in dataset_paths.values()]
    combined_df = pd.concat(dataframes, ignore_index=True).reset_index(drop=True)
    print(f"Size before dropping duplicates: {combined_df.shape}")

    deduplicated_df = combined_df.drop_duplicates().reset_index(drop=True)
    print(f"Size after dropping duplicates: {deduplicated_df.shape}")
    print(deduplicated_df.head())

    return deduplicated_df


def load_single(path):
    """Reads a single dataset CSV, keeping On/Off/Active and dropping
    duplicate rows within that dataset."""
    df = _load_csv_subset(path)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


# =============================================================================
# 6. PERSISTENCE HELPERS — everything needed to regenerate a graph later
# =============================================================================
def save_graph_data(name, T, scores, true_y, bin_centers, active_ratios, num_bins=NUM_BINS):
    """Pickles the raw scores/labels plus the derived bin stats for a given
    (dataset, T) so `plot_active_ratio` can redraw the graph with no
    inference required."""
    os.makedirs(graph_data_path, exist_ok=True)
    out_path = os.path.join(graph_data_path, f"{name}_T{T}.pkl")

    payload = {
        "dataset": name,
        "T": T,
        "num_bins": num_bins,
        "scores": np.asarray(scores),
        "true_y": np.asarray(true_y),
        "bin_centers": bin_centers,
        "active_ratios": active_ratios,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved graph data -> {out_path}")
    return out_path


def load_graph_data(name, T):
    """Loads back a previously-saved (dataset, T) graph payload."""
    in_path = os.path.join(graph_data_path, f"{name}_T{T}.pkl")
    with open(in_path, "rb") as f:
        return pickle.load(f)


def regenerate_graph(name, T, save_path=None):
    """Convenience helper: reload saved data for (name, T) and redraw the
    active-ratio graph without touching the model."""
    payload = load_graph_data(name, T)
    plot_active_ratio(
        payload["bin_centers"],
        payload["active_ratios"],
        num_bins=payload["num_bins"],
        title=f"{payload['dataset']} (T={payload['T']})",
        save_path=save_path,
    )
    return payload


# =============================================================================
# 7. PIPELINE STEPS
# =============================================================================
def run_calibration(model):
    """
    Combines Change-seq + Site-seq into `changeseq_siteseq`, fits the
    MinMax scaler on it (T = 10), draws its active-ratio graph, computes the
    confidence-weight bins, and persists every artifact.
    """
    df = load_and_combine(CALIBRATION_DATASETS)

    test_x = one_hot_features(df)
    test_y = df["Active"]

    scores = getScore(model, test_x, test_y, T=CALIBRATION_T, scaler="softmax", fit_scaler=True)

    bin_centers, active_ratios = graphActiveRatio(
        scores,
        true_y=test_y,
        num_bins=NUM_BINS,
        title=f"{CALIBRATION_NAME} (T={CALIBRATION_T})",
        save_path=os.path.join(plots_path, f"{CALIBRATION_NAME}_T{CALIBRATION_T}.png"),
    )

    save_graph_data(CALIBRATION_NAME, CALIBRATION_T, scores, test_y, bin_centers, active_ratios)

    wt = calculate_weights(scores, test_y, WEIGHT_BIN_EDGES)
    weights = wt["active_ratio"]
    print(WEIGHT_BIN_EDGES)
    print(weights.to_numpy())

    os.makedirs(base_model_path, exist_ok=True)
    with open(WEIGHTS_PATH, "wb") as f:
        pickle.dump([WEIGHT_BIN_EDGES, weights], f)
    print(f"Saved confidence-weight bins -> {WEIGHTS_PATH}")

    # Full weight table too, in case more than bins/active_ratio is needed later.
    weight_table_path = os.path.join(graph_data_path, f"{CALIBRATION_NAME}_weight_table.pkl")
    os.makedirs(graph_data_path, exist_ok=True)
    with open(weight_table_path, "wb") as f:
        pickle.dump(wt, f)

    return {"scores": scores, "true_y": test_y, "weights": weights}


def run_eval_datasets(model):
    """
    For each of Circle-seq, Guide-seq, Surro-seq and TTISS (each evaluated
    independently), reuse the fitted scaler to score the dataset at
    T = 1, 5, 10 and draw + persist one active-ratio graph per T.
    """
    results = {}

    for name, path in EVAL_DATASETS.items():
        df = load_single(path)
        test_x = one_hot_features(df)
        test_y = df["Active"]

        for T in EVAL_T_VALUES:
            scores = getScore(model, test_x, test_y, T=T, scaler="softmax", fit_scaler=False)

            bin_centers, active_ratios = graphActiveRatio(
                scores,
                true_y=test_y,
                num_bins=NUM_BINS,
                title=f"{name} (T={T})",
                save_path=os.path.join(plots_path, f"{name}_T{T}.png"),
            )

            save_graph_data(name, T, scores, test_y, bin_centers, active_ratios)
            results[(name, T)] = {"scores": scores, "true_y": test_y}

    return results


# =============================================================================
# 8. MAIN
# =============================================================================
def main():
    set_seed()
    os.makedirs(base_model_path, exist_ok=True)
    os.makedirs(graph_data_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    print(device)

    model = load_model()

    # 1) Fit scaler + calibrate confidence-weight bins on changeseq_siteseq (T=10)
    run_calibration(model)

    # 2) Evaluate each remaining dataset independently at T = 1, 5, 10
    run_eval_datasets(model)


if __name__ == "__main__":
    main()