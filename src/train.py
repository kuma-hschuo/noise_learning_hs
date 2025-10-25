# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import copy
from tqdm import tqdm

def evaluate(model, data_loader, device):
    """Evaluates the model on the given data loader."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.round(torch.sigmoid(outputs))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, all_labels, all_preds

def run_experiment(model, train_loader, val_loader, test_loader, config):
    """Runs a single full experiment: training and final evaluation."""
    device = torch.device(config['device'])
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['training_params']['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    best_epoch = -1
    best_model_state = None
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    epochs = config['training_params']['epochs']
    epoch_pbar = tqdm(range(epochs), desc="Epochs", leave=False) # leave=Falseでループ終了後にバーを消す

    for epoch in epoch_pbar:
        # --- Training ---
        model.train()
        # BCEWithLogitsLossは内部でSigmoidを適用するため、手動でのSigmoidは不要
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # --- Evaluation ---
        avg_train_loss, train_acc, _, _ = evaluate(model, train_loader, device)
        avg_val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        epoch_pbar.set_postfix(train_loss=f"{avg_train_loss:.4f}", val_loss=f"{avg_val_loss:.4f}", val_acc=f"{val_acc:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())


    # --- Final Evaluation with Best Model ---
    model.load_state_dict(best_model_state)
    
    final_train_loss, final_train_acc, _, _ = evaluate(model, train_loader, device)
    final_val_loss, final_val_acc, _, _ = evaluate(model, val_loader, device)
    final_test_loss, final_test_acc, test_labels, test_preds = evaluate(model, test_loader, device)
    
    # --- Collect Metrics ---
    results = {
        'total_epochs': epochs,
        'final_loss': { 'train': final_train_loss, 'validation': final_val_loss, 'test': final_test_loss },
        'accuracy': { 'train': final_train_acc, 'validation': final_val_acc, 'test': final_test_acc },
        'best_epoch': best_epoch,
        'precision': precision_score(test_labels, test_preds, zero_division=0),
        'recall': recall_score(test_labels, test_preds, zero_division=0),
        'f1_score': f1_score(test_labels, test_preds, zero_division=0),
        'confusion_matrix': confusion_matrix(test_labels, test_preds).tolist()
    }
    return results, best_model_state