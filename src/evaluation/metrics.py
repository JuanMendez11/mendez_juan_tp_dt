import torch
import numpy as np

def hit_rate_at_k(predictions, targets, k=10):
    """
    Calcula Hit Rate @K.
    
    Args:
        predictions: (batch, num_items) - scores para cada item
        targets: (batch,) - item verdadero
        k: top-K items
    
    Returns:
        hit_rate: float entre 0 y 1
    """
    # Obtener los índices de los top-K items con mayor score
    top_k = torch.topk(predictions, k, dim=1).indices  # (batch, k)
    # Verificar si el target está en alguna de las k posiciones
    hits = (top_k == targets.unsqueeze(1)).any(dim=1).float()  # (batch,)
    # Promedio de hits en todo el batch
    return hits.mean().item()

def ndcg_at_k(predictions, targets, k=10):
    """
    Normalized Discounted Cumulative Gain @K.
    
    Args:
        predictions: (batch, num_items) - scores para cada item
        targets: (batch,) - item verdadero
        k: top-K items a considerar
    
    Returns:
        ndcg: float entre 0 y 1 (promedio en el batch)
    """
    # Obtener top-k items predichos
    top_k_indices = torch.topk(predictions, k, dim=1).indices  # (batch, k)    
    # Crear vector de relevancia (1 si es el target, 0 si no)
    # Esto marca en qué posición (si alguna) está el target
    relevance = (top_k_indices == targets.unsqueeze(1)).float()  # (batch, k)
    # Calcular DCG (Discounted Cumulative Gain)
    # DCG penaliza items relevantes en posiciones bajas con log2(rank+1)
    # ranks = [1, 2, 3, ..., k]
    ranks = torch.arange(1, k+1, device=predictions.device).float()  # (k,)
    # DCG = Σ (relevancia_i / log2(posición_i + 1))
    # Nota: log2 = logaritmo en base 2 (ej: log2(2)=1, log2(4)=2, log2(8)=3)
    # El +1 en el denominador hace que:
    #   - posición 1 → log2(2) = 1.0
    #   - posición 2 → log2(3) ≈ 1.585
    #   - posición 3 → log2(4) = 2.0
    dcg = (relevance / torch.log2(ranks + 1)).sum(dim=1)  # (batch,)
    # Calcular IDCG (Ideal DCG)
    # Es el DCG máximo posible = cuando el target está en posición 1
    # IDCG = 1.0 / log2(1+1) = 1.0 / log2(2) = 1.0 / 1.0 = 1.0
    # (porque log2(2) = 1, ya que 2^1 = 2)
    idcg = 1.0 / np.log2(2)
    # Normalizar: NDCG = DCG / IDCG
    # Esto hace que NDCG esté siempre entre 0 y 1
    ndcg = dcg / idcg  # (batch,)
    return ndcg.mean().item()

def mrr(predictions, targets):
    """
    Mean Reciprocal Rank.

    Args:
        predictions: (batch, num_items) - scores para cada item
        targets: (batch,) - item verdadero
    
    Returns:
        mrr: float entre 0 y 1 (promedio de reciprocal ranks)
    """
    # Ordenar todos los items por score (de mayor a menor)
    # sorted_indices[i] contiene los items ordenados por probabilidad
    sorted_indices = torch.argsort(predictions, dim=1, descending=True)  # (batch, num_items)   
    # Encontrar en qué posición está el target para cada ejemplo
    # nonzero() encuentra dónde está True
    # [:,1] obtiene la columna (posición en el ranking)
    # +1 porque las posiciones empiezan en 0 pero queremos ranks desde 1
    ranks = (sorted_indices == targets.unsqueeze(1)).nonzero()[:, 1] + 1  # (batch,)
    # Calcular Reciprocal Rank = 1 / posición
    # Ej: si target está en posición 3 → RR = 1/3 = 0.333
    rr = 1.0 / ranks.float()  # (batch,)
    # Mean Reciprocal Rank = promedio de todos los RR
    return rr.mean().item()