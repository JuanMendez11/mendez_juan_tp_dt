# 🎯 IMPLEMENTAR: Funciones de métricas
# Código de referencia disponible en el TP

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
    # TODO: Ver código de referencia en el TP
    # Hint: Usar torch.topk para obtener top-k predicciones
    # Hint: Verificar si target está en top-k
    pass

def ndcg_at_k(predictions, targets, k=10):
    """
    Normalized Discounted Cumulative Gain @K.
    """
    # TODO: Ver fórmula en el TP
    # NDCG = DCG / IDCG
    pass

def mrr(predictions, targets):
    """
    Mean Reciprocal Rank.
    """
    # TODO: MRR = promedio de 1/rank del item verdadero
    pass