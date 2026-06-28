import pandas as pd
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

pwd = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_pkl(data, pkl):
    with open(pkl, 'wb') as f:
        pickle.dump(data, f)
    return 

def load_pkl(pkl):
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
    return data

def encoder(RNAseq, order=['A','T','C','G']):
    lookup_table = {order[0]:[1,0,0,0],
                    order[1]:[0,1,0,0],
                    order[2]:[0,0,1,0],
                    order[3]:[0,0,0,1]}
    encoded = np.zeros((len(RNAseq), len(order)))

    for i in range(len(RNAseq)):
        nu = RNAseq[i]
        if nu in lookup_table:
            encoded[i] = np.array(lookup_table[nu])
        else:
            # Handle unknown nucleotides by leaving them as zero representation
            pass

    return encoded

def superpose(encoded1, encoded2):
    if len(encoded1) != len(encoded2):
        print("Size Mismatch")
        return encoded1

    superposed = np.zeros(encoded1.shape)

    for i in range(len(encoded1)):
        for j in range(len(encoded1[i])):
            if encoded1[i][j] == encoded2[i][j]:
                superposed[i][j] = encoded1[i][j]
            else:
                superposed[i][j] = encoded1[i][j] + encoded2[i][j]
    return superposed

def one_hot_features(df):
    enc_superposed = []
    
    # Ensure columns 'On' and 'Off' are present
    on_col = 'On' if 'On' in df.columns else df.columns[0]
    off_col = 'Off' if 'Off' in df.columns else df.columns[1]
    
    for idx, row in df.iterrows():
        on_seq = row[on_col]
        off_seq = row[off_col]
        
        target = encoder(on_seq)
        off_target = encoder(off_seq)
        superposed = superpose(target, off_target)
        enc_superposed.append(superposed)
        
    return np.array(enc_superposed)

class RNN_Model_Generic(nn.Module):
    def __init__(self, config, model_type="LSTM"):
        super(RNN_Model_Generic, self).__init__()
        
        self.model_type = model_type
        self.vocab_size = config.get("vocab_size", 0)
        self.emb_size = config.get("emb_size", 4)
        self.hidden_size = config.get("hidden_size", 512)
        self.lstm_layers = config.get("lstm_layers", 1)
        self.bi_lstm = config.get("bi_lstm", True)
        self.reshape = config.get("reshape", False)

        self.number_hidden_layers = config.get("number_hidder_layers", 2)
        self.dropout_prob = config.get("dropout_prob", 0.4)
        
        self.hidden_shape = self.hidden_size * 2 if self.bi_lstm else self.hidden_size

        self.embedding = None
        if self.vocab_size > 0:
            self.embedding = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=0)

        if model_type == "LSTM":
            self.lstm = nn.LSTM(self.emb_size, self.hidden_size, num_layers=self.lstm_layers,
                                batch_first=True, bidirectional=self.bi_lstm)
        elif model_type == "GRU":
            self.lstm = nn.GRU(self.emb_size, self.hidden_size, num_layers=self.lstm_layers,
                               batch_first=True, bidirectional=self.bi_lstm)
        else:
            self.lstm = nn.RNN(self.emb_size, self.hidden_size, num_layers=self.lstm_layers,
                               batch_first=True, bidirectional=self.bi_lstm)

        start_size = self.hidden_shape
        self.hidden_layers = []
        for i in range(self.number_hidden_layers):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(start_size, start_size // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_prob)))
            start_size = start_size // 2

        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.output = nn.Linear(start_size, 2)

    def forward(self, x):
        dir = 2 if self.bi_lstm else 1
        h = torch.zeros((self.lstm_layers * dir, x.size(0), self.hidden_size)).to(device)
        c = torch.zeros((self.lstm_layers * dir, x.size(0), self.hidden_size)).to(device)

        if self.embedding is not None:
            x = x.type(torch.LongTensor).to(device)
            x = self.embedding(x)
        elif self.reshape:
            x = x.view(x.shape[0], x.shape[1], 1)

        if self.model_type == "LSTM":
            x, (hidden, cell) = self.lstm(x, (h, c))
        else:
            x, hidden = self.lstm(x, h)

        x = x[:, -1, :]
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output(x)
        return x

class TrainerDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        if isinstance(targets, np.ndarray):
            self.targets = torch.from_numpy(targets)
        else:
            self.targets = torch.tensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.Tensor(self.inputs[idx]), self.targets[idx]

def tester(model, test_x, test_y):
    test_dataset = TrainerDataset(test_x, test_y)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
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
        print('Accuracy: %.4f' % self.acc)
        print('Precision: %.4f' % self.pre)
        print('Recall: %.4f' % self.re)
        print('F1 Score: %.4f' % self.f1)
        print('ROC: %.4f' % self.roc)
        print('PR AUC: %.4f' % self.prc)
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
        stats.print()

    return stats

class PredictDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

def predictor(model, test_x):
    test_dataset = PredictDataset(test_x)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    model.eval()
    results = []
    with torch.no_grad():
        for batch_features in test_dataloader:
            outputs = model(batch_features.to(device))
            results.extend(outputs.detach().cpu())
    return results

def get_score_without_label(model, test_x, scaler, T=10):
    results = predictor(model, test_x)
    predictions = [torch.nn.functional.softmax(r/T, dim=0) for r in results]
    scaled_scores = np.array([y[1].item() for y in predictions])
    scaled_scores = scaler.transform(scaled_scores.reshape(-1, 1)).flatten()
    return scaled_scores
