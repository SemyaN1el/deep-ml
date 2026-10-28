# --- cell 1 ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a PyTorch model and return training history.
    
    This is the standard PyTorch training pattern you'll use everywhere.
    Now you can use torch.optim to handle the gradient updates!
    
    Args:
        model: nn.Module to train
        X_train: training features, shape (N, ...)
        y_train: training labels, shape (N,)
        X_val: validation features, shape (M, ...)
        y_val: validation labels, shape (M,)
        epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate
    
    Returns:
        history: List of dicts, one per epoch, with keys:
            - 'epoch': epoch number (starting from 1)
            - 'train_loss': average training loss for the epoch
            - 'val_loss': validation loss after the epoch
            - 'val_accuracy': validation accuracy after the epoch
    
    Steps:
        1. Create optimizer: optim.Adam(model.parameters(), lr=lr)
        2. Create loss function: nn.CrossEntropyLoss()
        3. For each epoch:
            a. Shuffle training data
            b. Loop over mini-batches:
                - optimizer.zero_grad()
                - Forward pass
                - Compute loss
                - loss.backward()
                - optimizer.step()
            c. Compute validation accuracy
            d. Append metrics to history
        4. Return history
    
    Hints:
        - torch.randperm(n) gives a random permutation for shuffling
        - Use model.train() before training, model.eval() before validation
        - Use torch.no_grad() during validation
        - logits.argmax(dim=1) gives predicted classes
    """
    # TODO: Implement the training loop
    
    
    # Your code here
    optimizer = optim.AdamW(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    
    device = next(model.parameters()).device 
    
    def prepare(X):
        if X.dtype == torch.uint8:
            X = X.float() / 255.0
        else:
            X = X.float()
        
        if X.ndim > 2:
            X = X.flatten(start_dim = 1)
        
        return X
    
    X_train = prepare(X_train)
    X_val = prepare(X_val)
    
    y_train = y_train.long()
    y_val = y_val.long()
    
    history = []
    
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_train))
        total_loss = 0.0
        
        for start in range(0, len(X_train), batch_size):
            batch_indices = indices[start:start + batch_size]
            
            batch_X = X_train[batch_indices].to(device)
            batch_y = y_train[batch_indices].to(device)
            
            optimizer.zero_grad()
            
            prediction = model(batch_X)
            
            loss = criterion(prediction, batch_y)
            loss.backward()
            
            optimizer.step()
            total_loss += loss.item() * len(batch_X)
        
        train_loss = total_loss / len(X_train)
        
        model.eval()
        
        with torch.no_grad():
            val_X = X_val.to(device)
            val_y = y_val.to(device)
            
            val_logits = model(val_X)
            val_loss = criterion(val_logits, val_y).item()
            
            predicted = val_logits.argmax(dim=1)
            val_accuracy = (predicted == val_y).float().mean().item()
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })
    return history