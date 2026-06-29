import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
os.environ['TF_KERAS'] = '1'
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import utils as np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc, accuracy_score, f1_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Global base directory resolution
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

# Path resolution helper
def get_dataset_path(filename):
    paths_to_try = [
        os.path.join(base_dir, "datasets", filename),
        os.path.join(base_dir, filename),
        filename
    ]
    for p in paths_to_try:
        if os.path.exists(p):
            return p
    return os.path.join(base_dir, "datasets", filename)

# Dynamic import of components
import sys
sys.path.append(base_dir)

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

from Encoder import Encoder, token_dict, BERT_encode, C_RNN_encode
from model_ts1 import build_bert

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

def preprocess_bert_data(df, on_col='Target sgRNA', off_col='Off Target sgRNA', label_col='label'):
    Negative, Positive = [], []
    for idx, row in df.iterrows():
        on_seq = str(row[on_col]).lower().replace('n', 'x')
        off_seq = str(row[off_col]).lower().replace('n', 'x')
        label_val = int(row[label_col])
        
        # Interleave target and off-target nucleotides into space-separated lowercase pairs
        token_list = [on_seq[k] + off_seq[k] for k in range(min(len(on_seq), len(off_seq)))]
        # Pad up to 24 tokens to match BERT's input size
        while len(token_list) < 24:
            token_list.append('xx')
            
        combined_str = " ".join(token_list)
        if label_val == 0:
            Negative.append([combined_str, label_val])
        else:
            Positive.append([combined_str, label_val])
    return Negative, Positive

def Train_DataGenerator(Negative_token, Negative_segment, Positive_token, Positive_segment, Negative_encode, Positive_encode, batchsize, positive_num, negative_num):
    Num_Positive, Num_Negative = len(Positive_token), len(Negative_token)
    Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')
    Index_Negative = [i for i in range(Num_Negative)]
    np.random.seed(2020)
    random.shuffle(Index_Negative)
    Total_num_batch = int(Num_Negative / batchsize)
    X1_input, X2_input, X_input, Y_input = [], [], [], []
    
    while True:
        for i in range(Total_num_batch):
            Negative_num = int(negative_num * (batchsize / 50))
            for j in range(Negative_num):
                neg_idx = Index_Negative[j + i * Negative_num]
                X_input.append(Negative_encode[neg_idx])
                X1_input.append(Negative_token[neg_idx])
                X2_input.append(Negative_segment[neg_idx])
                Y_input.append(0)
            for k in range(int(positive_num * batchsize / 50)):
                pos_idx = Index_Positive[k]
                X_input.append(Positive_encode[pos_idx])
                X1_input.append(Positive_token[pos_idx])
                X2_input.append(Positive_segment[pos_idx])
                Y_input.append(1)
            Y_input = np_utils.to_categorical(Y_input)
            yield [np.array(X_input), np.array(X1_input), np.array(X2_input)], np.array(Y_input)
            X1_input, X2_input, X_input, Y_input = [], [], [], []
            Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')

def DataGenerator(Negative_token, Negative_segment, Positive_token, Positive_segment, Negative_encode, Positive_encode, batchsize, positive_num, negative_num):
    Num_Positive, Num_Negative = len(Positive_token), len(Negative_token)
    Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')
    Index_Negative = [i for i in range(Num_Negative)]
    np.random.seed(2020)
    random.shuffle(Index_Negative)
    Total_num_batch = int(Num_Negative / batchsize)
    X1_input, X2_input, X_input, Y_input = [], [], [], []
    
    while True:
        for i in range(Total_num_batch):
            Negative_num = int(negative_num * (batchsize / 50))
            for j in range(Negative_num):
                neg_idx = Index_Negative[j + i * Negative_num]
                X_input.append(Negative_encode[neg_idx])
                X1_input.append(Negative_token[neg_idx])
                X2_input.append(Negative_segment[neg_idx])
                Y_input.append(0)
            for k in range(int(positive_num * batchsize / 50)):
                pos_idx = Index_Positive[k]
                X_input.append(Positive_encode[pos_idx])
                X1_input.append(Positive_token[pos_idx])
                X2_input.append(Positive_segment[pos_idx])
                Y_input.append(1)
            Y_input = np_utils.to_categorical(Y_input)
            yield [np.array(X_input), np.array(X1_input), np.array(X2_input)], np.array(Y_input)
            X1_input, X2_input, X_input, Y_input = [], [], [], []
            Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')

class Test_DataGenerator:
    def __init__(self, Test_Data_token, Test_Data_segment, Test_Data_encode, batch_size):
        self.Test_Data_token = Test_Data_token
        self.Test_Data_segment = Test_Data_segment
        self.Test_Data_encode = Test_Data_encode
        self.batch_size = batch_size
        self.steps = len(self.Test_Data_token) // self.batch_size
        if len(self.Test_Data_token) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.Test_Data_token)))
            X, X1, X2 = [], [], []
            for i in idxs:
                X.append(self.Test_Data_encode[i])
                X1.append(self.Test_Data_token[i])
                X2.append(self.Test_Data_segment[i])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    yield [np.array(X), np.array(X1), np.array(X2)]
                    X, X1, X2 = [], [], []

def Shuffle(x):
    x = np.array(x)
    Seq = [i for i in range(len(x))]
    random.shuffle(Seq)
    x = x[Seq]
    return x

def eval_matrices(model, test_generator, test_y, debug=True):
    y_hat_pairs = model.predict_generator(test_generator.__iter__(), steps=len(test_generator))
    y_prob = y_hat_pairs[:, 1]
    pred_y_list = (y_prob > 0.5).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(test_y, pred_y_list).ravel()
    precision, recall, _ = precision_recall_curve(test_y, y_prob)
    fpr, tpr, _ = roc_curve(test_y, y_prob)
    
    auc_score = auc(recall, precision)
    acc = accuracy_score(test_y, pred_y_list)
    
    pr = tp / (tp + fp) if tp + fp > 0 else -1
    re = tp / (tp + fn) if tp + fn > 0 else -1
    f1 = 2 * pr * re / (pr + re) if pr + re > 0 else -1
    
    stats = Stats()
    stats.acc = acc
    stats.pre = pr
    stats.re = re
    stats.f1 = f1
    stats.roc = roc_auc_score(test_y, y_prob)
    stats.prc = auc_score
    stats.tn = tn
    stats.fp = fp
    stats.fn = fn
    stats.tp = tp
    
    if debug:
        print('Accuracy: %.4f' % acc)
        print('Precision: %.4f' % pr)
        print('Recall: %.4f' % re)
        print('F1 Score: %.4f' % f1)
        print('ROC: %.4f' % stats.roc)
        print('PR AUC: %.4f' % auc_score)
        
    curve_data = {
        'fpr': fpr,
        'tpr': tpr,
        'recall': recall,
        'precision': precision,
        'pred_y': y_prob,
        'pred_y_list': pred_y_list,
        'test_y': test_y
    }
    return stats, curve_data

def compute_metric_bootstraps(test_y, pred_y, pred_y_list, n_bootstraps=1000, alpha=0.05, seed=12345):
    bootstrapped_roc = []
    bootstrapped_f1 = []
    bootstrapped_pr_auc = []
    
    n_samples = len(test_y)
    rng = np.random.default_rng(seed=seed)
    
    print(f"Running {n_bootstraps} bootstrap iterations...")
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
    
    print("\n--- Research Report: Model Performance Metrics (Bootstrap) ---")
    print(f"ROC AUC  : {mean_roc:.4f} ± {std_roc:.4f}  (95% CI: [{ci_roc[0]:.4f}, {ci_roc[1]:.4f}])")
    print(f"F1-Score : {mean_f1:.4f} ± {std_f1:.4f}  (95% CI: [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}])")
    print(f"PR AUC   : {mean_pr:.4f} ± {std_pr:.4f}  (95% CI: [{ci_pr[0]:.4f}, {ci_pr[1]:.4f}])")
    
    return {
        'roc': {'mean': mean_roc, 'std': std_roc, 'ci': ci_roc},
        'f1': {'mean': mean_f1, 'std': std_f1, 'ci': ci_f1},
        'pr_auc': {'mean': mean_pr, 'std': std_pr, 'ci': ci_pr}
    }

if __name__ == '__main__':
    log_file_path = os.path.join(base_dir, "ts1_output.log")
    log_file = open(log_file_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    
    data_path = get_dataset_path("all_off_target.csv")
    df = pd.read_csv(data_path)
    
    Negative, Positive = preprocess_bert_data(df)
    
    IR = len(Positive)/len(Negative)
    positive_num = 0
    negative_num = 0
    positive_weight = 0
    if IR > 250:
        if IR > 2000:
            positive_weight = 50
            positive_num = 25
            negative_num = 25
        else:
            positive_weight = 30
            positive_num = 20
            negative_num = 30
    else:
        positive_weight = 5   
        positive_num = 15
        negative_num = 35
    class_weights = {0: 0.5, 1: positive_weight}
    
    Negative = Shuffle(Negative)
    Positive = Shuffle(Positive)
    
    Train_Validation_Negative, Test_Negative = train_test_split(Negative, test_size=0.15, random_state=42)
    Train_Validation_Positive, Test_Positive = train_test_split(Positive, test_size=0.15, random_state=42)
    Train_Negative, Validation_Negative = train_test_split(Train_Validation_Negative, test_size=0.2, random_state=42)
    Train_Positive, Validation_Positive = train_test_split(Train_Validation_Positive, test_size=0.2, random_state=42)
    
    Train_Negative_token, Train_Negative_segment = BERT_encode(Train_Negative)
    Validation_Negative_token, Validation_Negative_segment = BERT_encode(Validation_Negative)
    Train_Positive_token, Train_Positive_segment = BERT_encode(Train_Positive)
    Validation_Positive_token, Validation_Positive_segment = BERT_encode(Validation_Positive)
    
    Train_Positive = pd.DataFrame(Train_Positive)
    Train_Negative = pd.DataFrame(Train_Negative)
    Validation_Positive = pd.DataFrame(Validation_Positive)
    Validation_Negative = pd.DataFrame(Validation_Negative)
    
    Train_Negative_encode = np.array(C_RNN_encode(Train_Negative))
    Validation_Negative_encode = np.array(C_RNN_encode(Validation_Negative))
    Train_Positive_encode = np.array(C_RNN_encode(Train_Positive))
    Validation_Positive_encode = np.array(C_RNN_encode(Validation_Positive))
    
    Xtest = np.vstack((Test_Negative, Test_Positive)) 
    Xtest = Shuffle(Xtest)
    Test_Data_token, Test_Data_segment = BERT_encode(Xtest)
    Test_DATA_encode = np.array(C_RNN_encode(pd.DataFrame(Xtest)))
    
    BATCH_SIZE = 256
    Train_NUM_BATCH = int(len(Train_Negative) / BATCH_SIZE)
    Valid_NUM_BATCH = int(len(Validation_Negative) / BATCH_SIZE)
    
    test_D = Test_DataGenerator(Test_Data_token, Test_Data_segment, Test_DATA_encode, batch_size=BATCH_SIZE)
    
    earlyStop = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=25, mode='auto', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', min_delta=0.01, factor=0.1, patience=10, min_lr=0.000001, verbose=1)
    
    model = build_bert()
    EPOCH = 30
    
    print("begin model training...")
    history_fit = model.fit_generator(
        Train_DataGenerator(Train_Negative_token, Train_Negative_segment, Train_Positive_token, Train_Positive_segment, Train_Negative_encode, Train_Positive_encode, batchsize=BATCH_SIZE, positive_num=positive_num, negative_num=negative_num),
        validation_data=DataGenerator(Validation_Negative_token, Validation_Negative_segment, Validation_Positive_token, Validation_Positive_segment, Validation_Negative_encode, Validation_Positive_encode, batchsize=BATCH_SIZE, positive_num=positive_num, negative_num=negative_num),
        callbacks=[reduce_lr, earlyStop],
        epochs=EPOCH,
        class_weight=class_weights,
        steps_per_epoch=Train_NUM_BATCH,
        validation_steps=1,
        verbose=1
    )
    
    y_test = [1 if float(i) > 0.0 else 0 for i in Xtest[:, 1]]
    y_test = np.array(y_test)
    
    stats, curve_data = eval_matrices(model, test_D, y_test, debug=True)
    
    model.save_weights('crispr_bert_model_ts1.h5')
    print("Model weights saved to 'crispr_bert_model_ts1.h5'")
    
    history = {'train_loss': history_fit.history['loss']}
    
    plt.figure(figsize=(20, 5.5))
    
    plt.subplot(1, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], color='#1f77b4', linewidth=2, label='Train Loss')
    plt.title('A. Training Loss Convergence', fontsize=13, fontweight='bold')
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('Cross-Entropy Loss', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    
    plt.subplot(1, 3, 2)
    plt.plot(curve_data['fpr'], curve_data['tpr'], color='#ff7f0e', linewidth=2.5, 
             label=f"Our Model (AUC = {stats.roc:.4f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random Guess (AUC = 0.5000)')
    plt.title('B. Receiver Operating Characteristic (ROC)', fontsize=13, fontweight='bold')
    plt.xlabel('False Positive Rate (FPR)', fontsize=11)
    plt.ylabel('True Positive Rate (TPR)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    
    plt.subplot(1, 3, 3)
    plt.plot(curve_data['recall'], curve_data['precision'], color='#2ca02c', linewidth=2.5, 
             label=f"Our Model (AUC = {stats.prc:.4f})")
    plt.title('C. Precision-Recall Curve (PRC)', fontsize=13, fontweight='bold')
    plt.xlabel('Recall', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'crispr_bert_curves_ts1.png'))
    print("Saved curve plots to 'crispr_bert_curves_ts1.png'")
    
    np.savez(os.path.join(base_dir, 'crispr_bert_curves_ts1.npz'), 
             epochs=np.array(list(epochs)),
             train_loss=np.array(history['train_loss']),
             fpr=curve_data['fpr'], 
             tpr=curve_data['tpr'], 
             recall=curve_data['recall'], 
             precision=curve_data['precision'])
             
    print("🎉 Success! Graphs generated and raw data coordinates saved as 'crispr_bert_curves_ts1.npz'.")
    
    bootstrap_results = compute_metric_bootstraps(
        curve_data['test_y'], 
        curve_data['pred_y'], 
        curve_data['pred_y_list']
    )
    
    # Save metrics summary to ts1_metrics.txt
    metrics_path = os.path.join(base_dir, 'ts1_metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("=== CRISPR-BERT TS1 Evaluation Metrics ===\n\n")
        f.write("--- Standard Evaluation ---\n")
        f.write(f"Accuracy : {stats.acc:.4f}\n")
        f.write(f"Precision: {stats.pre:.4f}\n")
        f.write(f"Recall   : {stats.re:.4f}\n")
        f.write(f"F1 Score : {stats.f1:.4f}\n")
        f.write(f"ROC AUC  : {stats.roc:.4f}\n")
        f.write(f"PR AUC   : {stats.prc:.4f}\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"   TN: {stats.tn}\tFP: {stats.fp}\n")
        f.write(f"   FN: {stats.fn}\tTP: {stats.tp}\n\n")
        
        f.write("--- Bootstrap Evaluation (1000 iterations, 95% CI) ---\n")
        roc_res = bootstrap_results['roc']
        f1_res = bootstrap_results['f1']
        prc_res = bootstrap_results['pr_auc']
        f.write(f"ROC AUC  : {roc_res['mean']:.4f} ± {roc_res['std']:.4f}  (95% CI: [{roc_res['ci'][0]:.4f}, {roc_res['ci'][1]:.4f}])\n")
        f.write(f"F1-Score : {f1_res['mean']:.4f} ± {f1_res['std']:.4f}  (95% CI: [{f1_res['ci'][0]:.4f}, {f1_res['ci'][1]:.4f}])\n")
        f.write(f"PR AUC   : {prc_res['mean']:.4f} ± {prc_res['std']:.4f}  (95% CI: [{prc_res['ci'][0]:.4f}, {prc_res['ci'][1]:.4f}])\n")
    print(f"Metrics saved to '{metrics_path}'")
    
    log_file.close()
