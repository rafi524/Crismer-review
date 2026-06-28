import os
import random
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
from utils import *

# Set seed
seed = 12345
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description="Train and Calibrate DIPOFF Model")
    parser.add_argument("--data", type=str, default=None, help="Path to all_off_target.csv file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate (default: 0.0001)")
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

    print(f"Loading dataset from: {os.path.abspath(data_path)}")
    df = pd.read_csv(data_path)
    
    print("Encoding sequences...")
    enc_superposed = []
    labels = []
    
    for i in range(df.shape[0]):
        df_row = df.iloc[i]
        target = encoder(df_row['Target sgRNA'])
        off_target = encoder(df_row['Off Target sgRNA'])
        superposed = superpose(target, off_target)
        
        enc_superposed.append(superposed)
        labels.append(df_row['label'])
        
        if (i+1) % 20000 == 0 or i == df.shape[0]-1:
            print(f"Encoded {i+1}/{df.shape[0]} sequences.")

    data_x = np.array(enc_superposed)
    data_y = np.array(labels)

    # Train / Test split
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,
                                                        stratify=data_y,
                                                        test_size=0.20,
                                                        random_state=5)
    print(f"Train set: {train_x.shape}, Test set: {test_x.shape}")

    # Set up balanced loader
    train_pos_idx = np.where(train_y == 1)[0]
    train_neg_idx = np.where(train_y == 0)[0]

    train_xp = train_x[train_pos_idx]
    train_xn = train_x[train_neg_idx]
    train_yp = train_y[train_pos_idx]
    train_yn = train_y[train_neg_idx]

    batch_size = args.batch_size
    train_dataset_pos = TrainerDataset(train_xp, train_yp)
    train_dataloader_pos = DataLoader(train_dataset_pos, batch_size=batch_size//2, shuffle=True)
    train_dataset_neg = TrainerDataset(train_xn, train_yn)
    train_dataloader_neg = DataLoader(train_dataset_neg, batch_size=batch_size//2, shuffle=True)

    config = {
        'vocab_size': 0,
        'emb_size': 4,
        'hidden_size': 512,
        'lstm_layers': 1,
        'bi_lstm': True,
        'number_hidder_layers': 2,
        'dropout_prob': 0.4,
        'reshape': False
    }

    print(f"Initializing BiLSTM model on device: {device}...")
    model = RNN_Model_Generic(config, model_type="LSTM").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_total_steps = len(train_dataloader_neg)
    model.train()

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pos_iter = iter(train_dataloader_pos)
        
        for i, (train_features_neg, train_labels_neg) in enumerate(train_dataloader_neg):
            try:
                train_features_pos, train_labels_pos = next(pos_iter)
            except StopIteration:
                pos_iter = iter(train_dataloader_pos)
                train_features_pos, train_labels_pos = next(pos_iter)
            
            # Slice to match size if final batch is smaller
            if len(train_features_neg) != len(train_features_pos):
                min_sz = min(len(train_features_neg), len(train_features_pos))
                train_features_neg = train_features_neg[:min_sz]
                train_labels_neg = train_labels_neg[:min_sz]
                train_features_pos = train_features_pos[:min_sz]
                train_labels_pos = train_labels_pos[:min_sz]
                
            train_features = torch.cat((train_features_pos, train_features_neg), 0)
            train_labels = torch.cat((train_labels_pos, train_labels_neg), 0)

            outputs = model(train_features.to(device))
            loss = criterion(outputs, train_labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= n_total_steps
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss:.4f}")

    # Create models folder if not exist
    os.makedirs("models", exist_ok=True)
    
    # Save the model weights
    model_path = 'models/dipoff_lstm_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {model_path}")

    # Evaluation & Calibration
    print("Running evaluation and calibration...")
    model.eval()
    
    print("Predicting on entire dataset for calibration...")
    results = []
    true_labels = []
    
    full_dataset = TrainerDataset(data_x, data_y)
    full_loader = DataLoader(full_dataset, batch_size=128, shuffle=False)
    
    with torch.no_grad():
        for features, batch_labels in full_loader:
            outputs = model(features.to(device)).detach().cpu()
            results.extend(outputs)
            true_labels.extend(batch_labels)
            
    # Apply temperature T=10 scaling and Softmax
    T = 10
    predictions = [torch.nn.functional.softmax(r/T, dim=0) for r in results]
    scores = np.array([y[1].item() for y in predictions])
    labels_np = np.array([y.item() for y in true_labels])
    
    # Fit and save MinMaxScaler
    scaler_instance = MinMaxScaler()
    scaled_scores = scaler_instance.fit_transform(scores.reshape(-1, 1)).flatten()
    joblib.dump(scaler_instance, "models/minmax_scaler.pkl")
    print("Scaler calibrated and saved to models/minmax_scaler.pkl")
    
    # Calculate bins and weights
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
    
    print("Calibration complete. DIPOFF is now fully configured!")

if __name__ == "__main__":
    main()
