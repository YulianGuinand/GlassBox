import torch
import torch.nn as nn
import torch.optim as optim
import time
import queue
import threading
import math
from typing import Dict, Any
from sklearn.metrics import r2_score, mean_squared_error
from glassbox.config import DEVICE

def training_worker(ts_queue: queue.Queue, params: Dict[str, Any], stop_event: threading.Event):
    """
    Background worker function for training.
    """
    try:
        model = params['model']
        train_loader = params['train_loader']
        test_loader = params['test_loader']
        epochs = params['epochs']
        learning_rate = params['learning_rate']
        criterion_name = params.get('criterion', 'CrossEntropyLoss')
        optimizer_name = params.get('optimizer', 'Adam')
        is_classification = params.get('is_classification', True)
        
        model = model.to(DEVICE)
        
        # Setup Loss and Optimizer
        if criterion_name == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
            
        if optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
        ts_queue.put({"status": "starting", "device": str(DEVICE)})
        
        for epoch in range(epochs):
            if stop_event.is_set():
                ts_queue.put({"status": "stopped"})
                return

            # --- TRAIN ---
            model.train()
            train_loss = 0.0
            
            # Classification Metrics
            correct = 0
            total = 0
            
            # Regression Metrics (Store for epoch Calculation)
            all_preds_train = []
            all_targets_train = []
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
                
                if is_classification:
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
                else:
                    all_preds_train.extend(outputs.detach().cpu().numpy())
                    all_targets_train.extend(y_batch.detach().cpu().numpy())
            
            avg_train_loss = train_loss / len(train_loader.dataset)
            
            train_metrics = {}
            if is_classification:
                train_acc = correct / total if total > 0 else 0.0
                train_metrics['acc'] = train_acc
            else:
                mse_train = mean_squared_error(all_targets_train, all_preds_train)
                rmse_train = math.sqrt(mse_train)
                r2_train = r2_score(all_targets_train, all_preds_train)
                train_metrics['rmse'] = rmse_train
                train_metrics['r2'] = r2_train
            
            # --- EVALUATE ---
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            all_preds_val = []
            all_targets_val = []
            
            with torch.no_grad():
                for X_val, y_val in test_loader:
                    X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                    outputs = model(X_val)
                    loss = criterion(outputs, y_val)
                    val_loss += loss.item() * X_val.size(0)
                    
                    if is_classification:
                        _, predicted = torch.max(outputs.data, 1)
                        total_val += y_val.size(0)
                        correct_val += (predicted == y_val).sum().item()
                    else:
                        all_preds_val.extend(outputs.cpu().numpy())
                        all_targets_val.extend(y_val.cpu().numpy())
            
            avg_val_loss = val_loss / len(test_loader.dataset)
            
            val_metrics = {}
            if is_classification:
                val_acc = correct_val / total_val if total_val > 0 else 0.0
                val_metrics['acc'] = val_acc
            else:
                mse_val = mean_squared_error(all_targets_val, all_preds_val)
                rmse_val = math.sqrt(mse_val)
                r2_val = r2_score(all_targets_val, all_preds_val)
                val_metrics['rmse'] = rmse_val
                val_metrics['r2'] = r2_val
            
            # Send metrics
            msg = {
                "status": "running",
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "is_classification": is_classification,
                # Flatten metrics for easier consumption
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()}
            }
            ts_queue.put(msg)
            
            # Simulate a tiny delay so UI has time to update if training is super fast on tiny data
            time.sleep(0.05)
            
        ts_queue.put({"status": "finished"})
        
    except Exception as e:
        ts_queue.put({"status": "error", "message": str(e)})
        raise e
