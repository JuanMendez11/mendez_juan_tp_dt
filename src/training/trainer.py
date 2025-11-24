import torch
import torch.nn.functional as F

def train_decision_transformer(model, train_loader, 
                               optimizer, device, num_epochs=50):
    """
    Entrena el Decision Transformer.
    
    Args:
        model: Instancia de DecisionTransformer
        train_loader: DataLoader de training
        optimizer: torch.optim.Optimizer (ej: Adam)
        device: 'cuda' o 'cpu'
        num_epochs: Número de épocas
    
    Returns:
        model: Modelo entrenado
        history: Dict con losses por época
    """
    model.to(device)
    history = {'train_loss': []}
    
    for epoch in range(num_epochs):
        # === TRAINING ===
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            states = batch['states'].to(device).long()      # (B, L)
            actions = batch['actions'].to(device).long()
            print(batch.keys())    # (B, L)
            rtg = batch['rtgs'].to(device)            # (B, L, 1)
            timesteps = batch['timesteps'].to(device).long() # (B, L)
            groups = batch['groups'].to(device).long()      # (B,)
            targets = batch['attention_mask'].to(device)    # (B, L) - next items
            
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
            
        history['train_loss'].append(avg_train_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
    
    return model, history