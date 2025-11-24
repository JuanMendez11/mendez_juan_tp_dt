import torch
import numpy as np
from src.evaluation.metrics import hit_rate_at_k, ndcg_at_k, mrr

@torch.no_grad()
def evaluate_model(model, test_data, device, target_return=None, k_list=[5, 10, 20]):
    """
    Evalúa el modelo en test set (cold-start users).

    Args:
        model: Decision Transformer entrenado
        test_data: lista de usuarios de test
        target_return: R̂ objetivo para conditioning (si None, usa max del training)
        k_list: lista de K para métricas @K
    
    Returns:
        metrics: dict con Hit Rate, NDCG, MRR para cada K
    """
    model.eval()
    
    # TODO: Seguir lógica del TP:
    # 1. Para cada usuario de test
    # 2. Simular sesión: empezar con history vacío
    # 3. Ir "recomendando" items y observando ratings
    # 4. Calcular métricas
    
    metrics = {f'HR@{k}': [] for k in k_list}
    metrics.update({f'NDCG@{k}': [] for k in k_list})
    metrics['MRR'] = []

    for user in test_data:
        group = user['group']
        items = user['items']
        ratings = user['ratings']
        
        # Simular sesión: tomar primeros N items como history, predecir siguiente
        context_len = 20
        
        for t in range(context_len, len(items)):
            # History
            history_items = items[t-context_len:t]
            history_ratings = ratings[t-context_len:t]
            
            # Calcular return-to-go
            if target_return is None:
                rtg = sum(history_ratings)  # o usar promedio del training
            else:
                rtg = target_return
            
            # Preparar inputs
            states = torch.tensor(history_items, dtype=torch.long).unsqueeze(0).to(device)
            actions = torch.tensor(history_items, dtype=torch.long).unsqueeze(0).to(device)
            rtg_input = torch.full((1, context_len, 1), rtg, dtype=torch.float32).to(device)
            timesteps = torch.arange(context_len, dtype=torch.long).unsqueeze(0).to(device)
            groups = torch.tensor([group], dtype=torch.long).to(device)
            
            # Predecir siguiente item
            # El modelo devuelve logits de forma (batch, seq_len, num_items)
            # batch=1, seq_len=t+1 (historia hasta ahora), num_items=752 (o 472)
            logits = model(states, actions, rtg_input, timesteps, groups)
            
            # Extraer predicción para el siguiente item
            # logits[0, -1, :] toma:
            #   - [0]: primer (y único) ejemplo del batch
            #   - [-1]: última posición temporal (la más reciente)
            #   - [:]: scores (logits) para todos los items posibles
            # 
            # IMPORTANTE: Son LOGITS (scores sin normalizar), NO probabilidades
            # - Logits pueden ser cualquier valor real: -5, 0, 3.2, 100, etc.
            # - Para las métricas (HR@K, NDCG@K, MRR) solo importa el RANKING
            # - NO necesitamos aplicar softmax porque el orden no cambia
            # 
            # Resultado: vector (num_items,) con score para cada item
            predictions = logits[0, -1, :]
            
            # Target
            target_item = items[t]
            
            # Calcular métricas
            for k in k_list:
                hr = hit_rate_at_k(predictions.unsqueeze(0), 
                                   torch.tensor([target_item]).to(device), k)
                metrics[f'HR@{k}'].append(hr)
                
                ndcg = ndcg_at_k(predictions.unsqueeze(0), 
                                torch.tensor([target_item]).to(device), k)
                metrics[f'NDCG@{k}'].append(ndcg)
            
            mrr_val = mrr(predictions.unsqueeze(0), 
                         torch.tensor([target_item]).to(device))
            metrics['MRR'].append(mrr_val)
    
    # Promediar métricas
    return {key: np.mean(val) for key, val in metrics.items()}