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
            states = batch['states'].to(device)      # (B, L)
            actions = batch['actions'].to(device)    # (B, L)
            rtg = batch['rtg'].to(device)            # (B, L, 1)
            timesteps = batch['timesteps'].to(device) # (B, L)
            groups = batch['groups'].to(device)      # (B,)
            targets = batch['targets'].to(device)    # (B, L) - next items
            
            # Forward pass
            logits = model(states, actions, rtg, timesteps, groups)
            
            # Compute loss (cross-entropy)
            loss = F.cross_entropy(
                logits.reshape(-1, model.num_items),
                targets.reshape(-1),
                ignore_index=-1  # para padding
            )
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # === VALIDATION ===
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for batch in val_loader:
                states = batch['states'].to(device)      # (B, L)
                actions = batch['actions'].to(device)    # (B, L)
                rtg = batch['rtg'].to(device)            # (B, L, 1)
                timesteps = batch['timesteps'].to(device) # (B, L)
                groups = batch['groups'].to(device)      # (B,)
                targets = batch['targets'].to(device)    # (B, L) - next items
                
                # Forward pass
                logits = model(states, actions, rtg, timesteps, groups)
                
                # Compute loss
                loss = F.cross_entropy(
                    logits.reshape(-1, model.num_items),
                    targets.reshape(-1),
                    ignore_index=-1  # para padding
                )
                
                total_val_loss += loss.item()
                
            avg_val_loss = total_val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
    
    return model, history