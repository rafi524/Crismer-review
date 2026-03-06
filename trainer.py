import torch
import torch.nn as nn
import os
from DataLoader import TrainerDataset
from torch.utils.data import DataLoader
from model import CRISPRTransformerModel
import numpy as np
import random

def train_model(model, train_loader, optimizer, scheduler, criterion, device, num_epochs):
    model = model.to(device)
    history = {'train_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # Training loop
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Loss tracking
            train_loss += loss.item()
        
        # Average loss for the epoch
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f}")
        scheduler.step()
    
    return model, history



def trainer(config, train_x, train_y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    seed = 12345
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset = TrainerDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    model = CRISPRTransformerModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    class_weights = torch.tensor([1.0, config['pos_weight']]).to(device)  # Adjust based on your class distribution
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    trained_model, history = train_model(model, train_loader, optimizer, scheduler, criterion, device, config['epochs'])
    return trained_model, history




