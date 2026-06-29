import os
import random
import pickle
import argparse
import sys
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    auc
)
import joblib

# Set seed
seed = 12345
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Import utilities from utils
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_rnafm_model():
    cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    dest_path = os.path.join(cache_dir, "RNA-FM_pretrained.pth")
    if os.path.exists(dest_path):
        print(f"RNA-FM model checkpoint already exists at {dest_path}")
        return
    
    url = "https://huggingface.co/cuhkaih/rnafm/resolve/b02d3594e4a25cd8b331c7319aa3746a8dfed2f0/RNA-FM_pretrained.pth"
    print("RNA-FM pretrained model not found in cache. Starting download...")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 100.0 / totalsize
            sys.stdout.write(f"\rProgress: {percent:.1f}% [{readsofar}/{totalsize} bytes]")
            if readsofar >= totalsize:
                sys.stdout.write("\n")
        else:
            sys.stdout.write(f"\rRead so far: {readsofar} bytes")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print(f"Successfully downloaded RNA-FM pretrained weights to {dest_path}")
    except Exception as e:
        print(f"Error downloading RNA-FM model: {e}")
        print("Please download it manually from:")
        print(url)
        print(f"and place it at {dest_path}")

class Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, batch_size=128, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.alphabet = model.get_alphabet()
        self.batch_size = batch_size

        self.train_loader = DataLoader(
            train_dataset,
            batch_sampler=BalancedBatchSampler(train_dataset, batch_size),
            collate_fn=lambda b: collate_fn(b, self.alphabet)
        )

        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda b: collate_fn(b, self.alphabet)
            )

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Param groups (different LR)
        backbone_params = []
        head_params = []
        for name, p in model.named_parameters():
            if "rna_model" in name:
                backbone_params.append(p)
            else:
                head_params.append(p)

        self.optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": 5e-4},
            {"params": head_params, "lr": 1e-3},
        ])

        # Store base LRs separately
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]

        self.epochs = 10
        self.warmup_epochs = 5
        self.history = {'train_loss': []}

        # For best model saving
        self.best_val_pr_auc = -float('inf')

    def _set_lr(self, epoch):
        scale = (epoch + 1) / self.warmup_epochs if epoch < self.warmup_epochs else 1.0
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = base_lr * scale

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        for step, (tokens, labels) in enumerate(self.train_loader):
            tokens = tokens.to(self.device)
            labels = labels.float().to(self.device).unsqueeze(1)

            preds = self.model(tokens)
            loss = self.criterion(preds, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if step % 20 == 0:
                print(f"Epoch {epoch} | Step {step}/{len(self.train_loader)} | Loss {loss.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        self.history['train_loss'].append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        all_labels = []
        all_probs = []

        for tokens, labels in self.val_loader:
            tokens = tokens.to(self.device)
            logits = self.model(tokens)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

        preds_binary = [1 if p >= 0.5 else 0 for p in all_probs]

        acc = accuracy_score(all_labels, preds_binary)
        pr_auc = average_precision_score(all_labels, all_probs)
        f1 = f1_score(all_labels, preds_binary)

        return acc, pr_auc, f1

    def fit(self):
        import time
        start_time = time.time()

        for epoch in range(self.epochs):
            self._set_lr(epoch)

            avg_loss = self.train_one_epoch(epoch)
            print(f"\nEpoch {epoch} completed | Avg Loss: {avg_loss:.4f}\n")

            if self.val_loader is not None:
                acc, pr_auc, f1 = self.evaluate()
                print(f"Epoch {epoch} | Val Accuracy: {acc:.4f} | Val PR-AUC: {pr_auc:.4f} | Val F1: {f1:.4f}\n")

                if pr_auc > self.best_val_pr_auc:
                    self.best_val_pr_auc = pr_auc
                    os.makedirs("models", exist_ok=True)
                    torch.save(self.model.state_dict(), "models/cclmoff_model.pth")
                    print(f"New best model saved at epoch {epoch} with PR-AUC: {pr_auc:.4f}\n")
            else:
                print()

        if self.val_loader is None:
            os.makedirs("models", exist_ok=True)
            torch.save(self.model.state_dict(), "models/cclmoff_model.pth")
            print("No validation set provided, saving the latest model to models/cclmoff_model.pth.\n")

        total_time = time.time() - start_time
        print(f"Total training time: {total_time:.2f} seconds\n")

        return self.history

class Tester:
    def __init__(self, model, dataset, batch_size=128, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.alphabet = model.get_alphabet()

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, self.alphabet)
        )

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        all_labels = []
        all_probs = []

        for tokens, labels in self.loader:
            tokens = tokens.to(self.device)
            # Logits returned by ProtRNA model
            preds = self.model(tokens).squeeze(1).cpu().numpy()

            all_probs.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

        y_true = np.array(all_labels)
        y_logits = np.array(all_probs)
        
        # Calculate probabilities from logits
        y_prob = 1.0 / (1.0 + np.exp(-y_logits))
        y_pred = (y_prob >= 0.5).astype(int)

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)

        stats = Stats(
            acc=accuracy_score(y_true, y_pred),
            pre=precision_score(y_true, y_pred, zero_division=0),
            re=recall_score(y_true, y_pred, zero_division=0),
            f1=f1_score(y_true, y_pred, zero_division=0),
            roc=roc_auc_score(y_true, y_prob),
            prc=average_precision_score(y_true, y_prob)
        )

        curve_data = {
            'fpr': fpr,
            'tpr': tpr,
            'recall': recall,
            'precision': precision,
            'pred_y': y_prob,
            'pred_y_list': y_pred,
            'test_y': y_true
        }

        return stats, curve_data

def compute_metric_bootstraps(test_y, pred_y, pred_y_list, n_bootstraps=100, alpha=0.05, seed=12345):
    """
    Computes both 95% Confidence Intervals and Mean +- Std for model metrics using bootstrapping.
    """
    bootstrapped_roc = []
    bootstrapped_f1 = []
    bootstrapped_pr_auc = []

    n_samples = len(test_y)
    rng = np.random.default_rng(seed=seed)

    print(f"Running {n_bootstraps} bootstrap iterations...\n")

    for i in range(n_bootstraps):
        boot_indices = rng.choice(n_samples, size=n_samples, replace=True)

        boot_y = test_y[boot_indices]
        boot_pred = pred_y[boot_indices]
        boot_pred_labels = pred_y_list[boot_indices]

        if len(np.unique(boot_y)) < 2:
            continue

        boot_roc = roc_auc_score(boot_y, boot_pred)
        boot_f1 = f1_score(boot_y, boot_pred_labels, zero_division=0)

        precision, recall, _ = precision_recall_curve(boot_y, boot_pred)
        boot_pr = auc(recall, precision)

        bootstrapped_roc.append(boot_roc)
        bootstrapped_f1.append(boot_f1)
        bootstrapped_pr_auc.append(boot_pr)

    mean_roc, std_roc = np.mean(bootstrapped_roc), np.std(bootstrapped_roc)
    mean_f1, std_f1 = np.mean(bootstrapped_f1), np.std(bootstrapped_f1)
    mean_pr, std_pr = np.mean(bootstrapped_pr_auc), np.std(bootstrapped_pr_auc)

    low_p = (alpha / 2.0) * 100
    high_p = (1.0 - alpha / 2.0) * 100

    ci_roc = (np.percentile(bootstrapped_roc, low_p), np.percentile(bootstrapped_roc, high_p))
    ci_f1 = (np.percentile(bootstrapped_f1, low_p), np.percentile(bootstrapped_f1, high_p))
    ci_pr = (np.percentile(bootstrapped_pr_auc, low_p), np.percentile(bootstrapped_pr_auc, high_p))

    print("\n--- Research Report: Model Performance Metrics with Bootstrapping ---\n")
    print(f"ROC AUC  : {mean_roc:.4f} \u00b1 {std_roc:.4f}  (95% CI: [{ci_roc[0]:.4f}, {ci_roc[1]:.4f}])\n")
    print(f"F1-Score : {mean_f1:.4f} \u00b1 {std_f1:.4f}  (95% CI: [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}])\n")
    print(f"PR AUC   : {mean_pr:.4f} \u00b1 {std_pr:.4f}  (95% CI: [{ci_pr[0]:.4f}, {ci_pr[1]:.4f}])\n")

    return {
        'roc': {'mean': mean_roc, 'std': std_roc, 'ci': ci_roc},
        'f1': {'mean': mean_f1, 'std': std_f1, 'ci': ci_f1},
        'pr_auc': {'mean': mean_pr, 'std': std_pr, 'ci': ci_pr}
    }

def main():
    parser = argparse.ArgumentParser(description="Train and Calibrate CCLMoff Model")
    parser.add_argument("--data", type=str, default=None, help="Path to all_off_target.csv file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--no_plots", action="store_true", help="Disable plotting graph curve saving")
    args = parser.parse_args()

    # Search for all_off_target.csv
    data_path = args.data
    if data_path is None:
        potential_paths = [
            "../../all_off_target.csv",
            "../all_off_target.csv",
            "all_off_target.csv",
            "E:/CRISMER-Extended/all_off_target.csv"
        ]
        for path in potential_paths:
            if os.path.exists(path):
                data_path = path
                break
    
    if data_path is None or not os.path.exists(data_path):
        raise FileNotFoundError("Could not find all_off_target.csv. Please specify its path using --data.")

    # 1. Download pretrained model if not found
    download_rnafm_model()

    print(f"Loading dataset from: {os.path.abspath(data_path)}")
    df = pd.read_csv(data_path)
    df['label'] = df['label'].astype(int)

    # Split dataset
    positive_df = df[df['label'] == 1]
    negative_df = df[df['label'] == 0]

    train_neg, test_neg = train_test_split(negative_df, test_size=0.2, random_state=42)
    train_pos, test_pos = train_test_split(positive_df, test_size=0.2, random_state=42)

    initial_train_data = pd.concat([train_neg, train_pos]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_data = pd.concat([test_neg, test_pos]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Split 10% of the initial_train_data for validation
    actual_train_data, val_data = train_test_split(initial_train_data, test_size=0.1, random_state=42)

    print(f"Training data shape: {actual_train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Create Datasets
    train_dataset = TrainerDataset(actual_train_data)
    val_dataset = TestDataset(val_data)
    test_dataset = TestDataset(test_data)

    # Initialize model
    print("Initializing ProtRNA model...")
    model = ProtRNA()

    # Initialize Trainer and train
    trainer = Trainer(model, train_dataset, val_dataset=val_dataset, batch_size=args.batch_size, device=device)
    trainer.epochs = args.epochs
    history = trainer.fit()

    # Load the best model weights
    print("Loading the best fine-tuned model...")
    model.load_state_dict(torch.load("models/cclmoff_model.pth"))

    # Perform evaluation on test set
    print("Evaluating on test set...")
    tester = Tester(model, test_dataset, batch_size=args.batch_size, device=device)
    stats, curve_data = tester.evaluate()

    print(stats.__dict__)
    print(f"\n--- Research Report: Model Performance Metrics ---\n")
    print(f"Accuracy: {stats.acc:.4f}")
    print(f"Precision: {stats.pre:.4f}")
    print(f"Recall: {stats.re:.4f}")
    print(f"F1-Score: {stats.f1:.4f}")
    print(f"ROC AUC: {stats.roc:.4f}")
    print(f"PR AUC: {stats.prc:.4f}\n")

    # Calibration on entire dataset (just like train_dipoff.py)
    print("Performing temperature scaling and MinMaxScaler fitting for calibration...")
    model.eval()

    full_dataset = TestDataset(df)
    full_loader = DataLoader(
        full_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=lambda b: collate_fn(b, model.get_alphabet())
    )

    results = []
    with torch.no_grad():
        for tokens, _ in full_loader:
            tokens = tokens.to(device)
            # ProtRNA logits
            logits = model(tokens).squeeze(1).cpu().numpy()
            results.extend(logits.tolist())

    results = np.array(results)
    
    # Temperature scaling T=10, then Sigmoid
    T = 10
    scores = 1.0 / (1.0 + np.exp(-results / T))
    labels_np = df['label'].to_numpy()

    # Fit and save MinMaxScaler
    scaler_instance = MinMaxScaler()
    scaled_scores = scaler_instance.fit_transform(scores.reshape(-1, 1)).flatten()
    joblib.dump(scaler_instance, "models/minmax_scaler.pkl")
    print("Scaler calibrated and saved to models/minmax_scaler.pkl")

    # Bins and weights calculation
    bins = [0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
    data_cal = pd.DataFrame({
        'score': scaled_scores,
        'Active': labels_np
    })
    data_cal['bin'] = pd.cut(data_cal['score'], bins=bins, include_lowest=True)

    bin_stats = (
        data_cal.groupby('bin', observed=False)
        .apply(
            lambda x: pd.Series({
                'active_count': (x['Active'] == 1).sum(),
                'total_count': len(x)
            })
        )
        .reset_index()
    )
    bin_stats['active_ratio'] = bin_stats['active_count'] / bin_stats['total_count']
    bin_stats['active_ratio'] = bin_stats['active_ratio'].fillna(0)
    weights = bin_stats['active_ratio']

    # Save bin weights
    with open("models/bin_weights.pkl", 'wb') as f:
        pickle.dump([bins, weights], f)
    print("Bin weights calculated and saved to models/bin_weights.pkl")
    print("Bins:", bins)
    print("Weights:", weights.to_numpy())

    # Bootstrapping (optional reporting)
    bootstrap_results = compute_metric_bootstraps(
        curve_data['test_y'],
        curve_data['pred_y'],
        curve_data['pred_y_list'],
        n_bootstraps=100
    )

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(20, 5.5))

            # Training Loss
            plt.subplot(1, 3, 1)
            epochs_range = range(1, len(history['train_loss']) + 1)
            plt.plot(epochs_range, history['train_loss'], marker='o', color='#1f77b4', linewidth=2, label='Train Loss')
            plt.title('A. Training Loss Convergence', fontsize=13, fontweight='bold')
            plt.xlabel('Epochs', fontsize=11)
            plt.ylabel('Cross-Entropy Loss', fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(fontsize=10)

            # ROC AUC
            plt.subplot(1, 3, 2)
            plt.plot(curve_data['fpr'], curve_data['tpr'], color='#ff7f0e', linewidth=2.5,
                     label=f"Our Model (AUC = {stats.roc:.4f})")
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random Guess (AUC = 0.5000)')
            plt.title('B. Receiver Operating Characteristic (ROC)', fontsize=13, fontweight='bold')
            plt.xlabel('False Positive Rate (FPR)', fontsize=11)
            plt.ylabel('True Positive Rate (TPR)', fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc="lower right", fontsize=10)

            # PR AUC
            plt.subplot(1, 3, 3)
            plt.plot(curve_data['recall'], curve_data['precision'], color='#2ca02c', linewidth=2.5,
                     label=f"Our Model (AUC = {stats.prc:.4f})")
            plt.title('C. Precision-Recall Curve (PRC)', fontsize=13, fontweight='bold')
            plt.xlabel('Recall', fontsize=11)
            plt.ylabel('Precision', fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc="lower left", fontsize=10)

            plt.tight_layout()
            plt.savefig('cclmoff_model_curves.png')
            print("Performance graph saved as 'cclmoff_model_curves.png'.")

            np.savez('cclmoff_model_curves.npz',
                     epochs=np.array(list(epochs_range)),
                     train_loss=np.array(history['train_loss']),
                     fpr=curve_data['fpr'],
                     tpr=curve_data['tpr'],
                     recall=curve_data['recall'],
                     precision=curve_data['precision'])
            print("Raw data coordinates saved as 'cclmoff_model_curves.npz'.")
        except Exception as plot_err:
            print(f"Warning: Could not save performance plots: {plot_err}")

    print("CCLMoff training and calibration script successfully completed!")

if __name__ == "__main__":
    main()
