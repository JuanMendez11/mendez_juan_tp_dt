# 🎯 IMPLEMENTAR: Baselines simples (código casi completo en el TP)

import numpy as np

class PopularityRecommender:
    """
    Recomienda items más populares (no personalizados).
    """
    def __init__(self):
        self.item_counts = None
        self.popular_items = None
    
    def fit(self, trajectories):
        """
        Args:
            trajectories: Lista de trayectorias (formato DT)
        """
        # TODO: Contar frecuencia de cada item en el dataset
        # Hint: Concatenar todos los 'items' de todas las trayectorias
        # all_items = np.concatenate([traj['items'] for traj in trajectories])
        # self.item_counts = np.bincount(all_items, minlength=752)
        
        # TODO: Ordenar por frecuencia (más popular primero)
        # self.popular_items = np.argsort(self.item_counts)[::-1]
        pass
    
    def recommend(self, user_history, k=10):
        """
        Recomienda top-k items populares no vistos.
        
        Args:
            user_history: lista de item IDs ya vistos
            k: número de recomendaciones
        
        Returns:
            recommendations: lista de k item IDs
        """
        # TODO: Filtrar items ya vistos y retornar top-k
        pass