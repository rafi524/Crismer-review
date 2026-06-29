import os
import pickle
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler

try:
    import fm
except ModuleNotFoundError:
    fm = None

pwd = os.path.dirname(os.path.realpath(__file__))

def save_pkl(data, pkl):
    with open(pkl, 'wb') as f:
        pickle.dump(data, f)

def load_pkl(pkl):
    with open(pkl, 'rb') as f:
        return pickle.load(f)

class ProtRNA(nn.Module):
    def __init__(self):
        super().__init__()
        if fm is None:
            raise ModuleNotFoundError(
                "The 'rna-fm' package is not installed. Please install it using 'pip install rna-fm' to use CCLMoff."
            )

        # Load pretrained RNA-FM
        self.rna_model, self.rna_alphabet = fm.pretrained.rna_fm_t12()

        # === ADD <sep> TOKEN ===
        self._add_sep_token()

        # MLP head
        self.dense1 = nn.Linear(640, 64)
        self.dense2 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.2)
        self.act = nn.Sigmoid()
        self.elu = nn.ELU()

        # Init linear layers
        for m in self.children():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)

    def _add_sep_token(self):
        alphabet = self.rna_alphabet

        # Add token to alphabet
        sep_token = "<sep>"
        if sep_token not in alphabet.tok_to_idx:
            new_idx = len(alphabet.all_toks)
            alphabet.all_toks.append(sep_token)
            alphabet.tok_to_idx[sep_token] = new_idx

            # Resize embedding
            old_emb = self.rna_model.embed_tokens
            old_num, dim = old_emb.weight.shape

            new_emb = nn.Embedding(old_num + 1, dim)
            new_emb.weight.data[:old_num] = old_emb.weight.data

            # Initialize <sep> from <eos>
            eos_id = alphabet.tok_to_idx["<eos>"]
            new_emb.weight.data[new_idx] = old_emb.weight.data[eos_id]

            self.rna_model.embed_tokens = new_emb

    def get_alphabet(self):
        return self.rna_alphabet

    def forward(self, tokens):
        rna_results = self.rna_model(tokens, repr_layers=[12])
        seq_emb = rna_results["representations"][12]

        # CLS token representation
        seq_emb = seq_emb[:, 0, :]

        x = self.elu(self.dropout(self.dense1(seq_emb)))
        x = self.dense2(self.dropout(x))

        return x

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=128):
        assert batch_size % 2 == 0, "Batch size must be even"

        self.dataset = dataset
        self.batch_size = batch_size
        self.half = batch_size // 2

        self.pos_idx = []
        self.neg_idx = []

        for i in range(len(dataset)):
            label = int(dataset.data.iloc[i]["label"])
            if label == 1:
                self.pos_idx.append(i)
            else:
                self.neg_idx.append(i)

        self.num_batches = len(dataset) // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            pos_batch = random.sample(
                self.pos_idx,
                min(self.half, len(self.pos_idx))
            )
            neg_batch = random.sample(
                self.neg_idx,
                min(self.half, len(self.neg_idx))
            )
            batch = pos_batch + neg_batch
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

class TrainerDataset(Dataset):
    def __init__(self, data):
        self.data = data.copy().reset_index(drop=True)

        # convert DNA -> RNA
        self.data["On"] = (
            self.data["Target sgRNA"]
            .astype(str)
            .str.upper()
            .str.replace("T", "U", regex=False)
        )

        self.data["Off"] = (
            self.data["Off Target sgRNA"]
            .astype(str)
            .str.upper()
            .str.replace("T", "U", regex=False)
        )

        self.labels = self.data["label"].to_numpy()

        # class stats
        counts = self.data["label"].value_counts().sort_index()
        self.class_num_list = counts.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq = f"{row['On']}<sep>{row['Off']}".replace("_", "-")
        label = int(row["label"])

        return {
            "seq": seq,
            "label": torch.tensor(label, dtype=torch.long)
        }

class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data.copy().reset_index(drop=True)

        self.data["On"] = (
            self.data["Target sgRNA"]
            .astype(str)
            .str.upper()
            .str.replace("T", "U", regex=False)
        )

        self.data["Off"] = (
            self.data["Off Target sgRNA"]
            .astype(str)
            .str.upper()
            .str.replace("T", "U", regex=False)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq = f"{row['On']}<sep>{row['Off']}".replace("_", "-")
        label = int(row["label"]) if "label" in row else 0

        return {
            "seq": seq,
            "label": torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch, alphabet):
    sequences = [b["seq"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])

    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([(i, s) for i, s in enumerate(sequences)])

    return tokens, labels

class Stats:
    def __init__(self, acc=0, pre=0, re=0, f1=0, roc=0, prc=0):
        self.acc = acc
        self.pre = pre
        self.re = re
        self.f1 = f1
        self.roc = roc
        self.prc = prc

def get_score_without_label(model, df, scaler, T=10, device="cuda"):
    """
    Computes scaled score predictions for on-target and off-target sequences in a DataFrame.
    """
    scoring_df = pd.DataFrame()
    scoring_df['Target sgRNA'] = df['On']
    scoring_df['Off Target sgRNA'] = df['Off']
    scoring_df['label'] = 0

    dataset = TestDataset(scoring_df)
    alphabet = model.get_alphabet()

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, alphabet)
    )

    model.eval()
    results = []
    with torch.no_grad():
        for tokens, _ in loader:
            tokens = tokens.to(device)
            # Logits are [batch, 1]
            logits = model(tokens).squeeze(1).cpu().numpy()
            results.extend(logits.tolist())

    results = np.array(results)
    
    # Calculate probabilities with Temperature-scaled Sigmoid
    probs = 1.0 / (1.0 + np.exp(-results / T))

    # Scale scores using MinMaxScaler
    scaled_scores = scaler.transform(probs.reshape(-1, 1)).flatten()
    return scaled_scores
