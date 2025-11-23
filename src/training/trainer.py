import torch
import torch.nn as nn
from tqdm import tqdm

def train_decision_transformer_hf(
    model,
    train_loader,
    num_epochs=10,
    learning_rate=1e-4,
    weight_decay=1e-4,
    device='cuda'
):
    """
    Entrena el Decision Transformer con HuggingFace.
    
    Args:
        model: DecisionTransformerHF
        train_loader: DataLoader con datos de entrenamiento
        num_epochs: número de épocas
        learning_rate: learning rate
        weight_decay: regularización L2
        device: 'cuda' o 'cpu'
    """
    model = model.to(device)
    model.train()
    
    # Optimizer: AdamW (recomendado para transformers)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Loss: Cross-Entropy (classification sobre items)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in pbar:
            # Mover a device
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            rtgs = batch['rtgs'].to(device)
            timesteps = batch['timesteps'].to(device)
            groups = batch['groups'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            action_logits = model(
                states=states,
                actions=actions,
                rtgs=rtgs,
                timesteps=timesteps,
                groups=groups,
                attention_mask=attention_mask
            )  # (batch, seq_len, num_items)
            
            # Preparar targets: queremos predecir la SIGUIENTE acción
            # Predicción en t → target en t+1
            # Descartamos última predicción (no hay target)
            logits = action_logits[:, :-1, :].reshape(-1, model.num_items)
            targets = actions[:, 1:].reshape(-1)
            
            # Crear máscara para ignorar padding
            mask = attention_mask[:, 1:].reshape(-1).bool()
            
            # Calcular loss solo en posiciones válidas
            loss = criterion(logits[mask], targets[mask])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (evitar explosión de gradientes)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Logging
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1} - Average Loss: {avg_loss:.4f}')
    
    return model