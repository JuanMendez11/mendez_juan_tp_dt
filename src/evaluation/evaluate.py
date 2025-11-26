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


import numpy as np

def evaluate_popularity_baseline(model, test_data, k_list=[5, 10, 20]):
    """
    Evalúa el modelo de Popularidad basándose en la lógica secuencial del TP.

    Args:
        model: Instancia de PopularityRecommender ya entrenada (.fit())
        test_data: lista de usuarios de test
        k_list: lista de cortes K para métricas
    
    Returns:
        metrics: dict con promedios de HR@K, NDCG@K, MRR
    """
    
    # Inicializar diccionarios para guardar resultados
    metrics = {f'HR@{k}': [] for k in k_list}
    metrics.update({f'NDCG@{k}': [] for k in k_list})
    metrics['MRR'] = []

    # Necesitamos pedir al modelo el K máximo necesario para cubrir todas las métricas
    max_k = max(k_list)
    
    # Context length para igualar la evaluación del Transformer (empezar a predecir tras N items)
    context_len = 20

    for user in test_data:
        items = user['items']
        # Nota: Popularity no usa ratings ni groups, solo la secuencia de items vistos
        
        # Si el usuario tiene menos items que el contexto, saltamos (igual que en el DT)
        if len(items) <= context_len:
            continue

        # Simular sesión secuencial
        for t in range(context_len, len(items)):
            
            # 1. Definir historia y target
            # history_items: todo lo visto antes del momento t (para filtrar ya vistos)
            history_items = items[:t] 
            target_item = items[t]
            
            # 2. Obtener recomendaciones
            # El modelo devuelve una lista ordenada de IDs, ej: [50, 12, 3...]
            recommendations = model.recommend(history_items, k=max_k)
            
            # 3. Calcular métricas manualmente
            # Buscamos en qué posición (rank) quedó el item objetivo
            try:
                # index lanza ValueError si no está en la lista. 
                # Sumamos 1 porque el índice empieza en 0.
                rank = recommendations.index(target_item) + 1
            except ValueError:
                rank = None # El target no apareció en el top max_k

            # Calcular HR y NDCG para cada k en k_list
            for k in k_list:
                if rank is not None and rank <= k:
                    # Hit Rate: 1 si está en el top k
                    metrics[f'HR@{k}'].append(1)
                    # NDCG: 1 / log2(rank + 1) si está
                    metrics[f'NDCG@{k}'].append(1.0 / np.log2(rank + 1))
                else:
                    metrics[f'HR@{k}'].append(0)
                    metrics[f'NDCG@{k}'].append(0)
            
            # MRR (Mean Reciprocal Rank)
            if rank is not None:
                metrics['MRR'].append(1.0 / rank)
            else:
                metrics['MRR'].append(0)

    # 4. Promediar métricas
    return {key: np.mean(val) for key, val in metrics.items()}