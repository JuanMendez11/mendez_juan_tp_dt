# 🎯 IMPLEMENTAR: Loop de entrenamiento
# Código de referencia en el TP

import torch
import torch.nn.functional as F

def train_decision_transformer(model, train_loader, val_loader, 
                               optimizer, device, num_epochs=50):
    """
    Entrena el Decision Transformer.
    
    Args:
        model: Instancia de DecisionTransformer
        train_loader: DataLoader de training
        val_loader: DataLoader de validación
        optimizer: torch.optim.Optimizer (ej: Adam)
        device: 'cuda' o 'cpu'
        num_epochs: Número de épocas
    
    Returns:
        model: Modelo entrenado
        history: Dict con losses por época
    """
    model.to(device)
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # === TRAINING ===
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            # TODO: Mover batch a device
            # states = batch['states'].to(device)
            # actions = ...
            # rtg = ...
            # timesteps = ...
            # groups = ...
            # targets = ...
            
            # TODO: Forward pass
            # logits = model(states, actions, rtg, timesteps, groups)
            
            # TODO: Compute loss (cross-entropy)
            # Hint: Reshape logits y targets para cross_entropy
            # loss = F.cross_entropy(...)
            
            # TODO: Backprop
            # optimizer.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # === VALIDATION ===
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for batch in val_loader:
                # TODO: Similar a training pero sin backprop
                pass
            avg_val_loss = total_val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
    
    return model, history