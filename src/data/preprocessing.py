# 🎯 IMPLEMENTAR: Función de preprocesamiento
# El código de referencia está en el TP - pueden copiarlo y adaptarlo

import numpy as np

def create_dt_dataset(df_train):
    """
    Convierte DataFrame raw a formato Decision Transformer.
    
    REFERENCIA: Ver código completo en 03_REFERENCIA_COMPLETA.md
    
    Args:
        df_train: DataFrame con [user_id, user_group, items, ratings]
    
    Returns:
        trajectories: List[Dict] con formato específico
    """
    trajectories = []
    
    for idx, row in df_train.iterrows():
        # TODO: Extraer items, ratings, group
        
        # TODO: Calcular returns-to-go (R̂)
        # Hint: Iterar hacia atrás desde el final
        # returns[t] = ratings[t] + returns[t+1]
        
        # TODO: Crear diccionario con formato correcto
        trajectory = {
            'items': ...,
            'ratings': ...,
            'returns_to_go': ...,
            'timesteps': ...,
            'user_group': ...
        }
        
        trajectories.append(trajectory)
    
    return trajectories


def validate_preprocessing(trajectories):
    """
    Valida que el preprocesamiento sea correcto.
    """
    # TODO: Verificar que:
    # - Todas las trayectorias tienen las keys correctas
    # - len(items) == len(ratings) == len(returns_to_go)
    # - returns_to_go[0] == sum(ratings)
    # - returns_to_go[-1] == ratings[-1]
    pass