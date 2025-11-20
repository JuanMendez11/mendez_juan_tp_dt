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
        items = row['items']        # numpy array de item IDs
        ratings = row['ratings']    # numpy array de ratings
        group = row['user_group']   # int (0-7)
        
        # TODO: Calcular returns-to-go (R̂)
        # Hint: Iterar hacia atrás desde el final
        # returns[t] = ratings[t] + returns[t+1]
        returns = np.zeros(len(ratings))
        
        # Último timestep: R̂_T = r_T
        returns[-1] = ratings[-1]
        
        # Iterar hacia atrás: R̂_t = r_t + R̂_{t+1}
        for t in range(len(ratings)-2, -1, -1):
            returns[t] = ratings[t] + returns[t+1]
        
        # TODO: Crear diccionario con formato correcto
        trajectory = {
            'items': items,
            'ratings': ratings,
            'returns_to_go': returns,
            'timesteps': np.arange(len(items)),
            'user_group': group
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
    for traj in trajectories:
        assert 'items' in traj
        assert 'ratings' in traj
        assert 'returns_to_go' in traj
        assert 'timesteps' in traj
        assert 'user_group' in traj
        
        items = traj['items']
        ratings = traj['ratings']
        returns = traj['returns_to_go']
        
        assert len(items) == len(ratings) == len(returns)
        assert returns[0] == np.sum(ratings)
        assert returns[-1] == ratings[-1]


# if __name__ == "__main__":
#     from load_data import load_training_data
#     import pickle
#     # Cargar datos de entrenamiento
#     df_train = load_training_data(dataset='netflix')
#     # Preprocesar datos
#     trajectories = create_dt_dataset(df_train)
#     # Validar preprocesamiento
#     validate_preprocessing(trajectories)
#     print("Preprocesamiento validado correctamente.")  
#     # Guardar dataset preprocesado
#     with open('../../data/processed/trajectories_train.pkl', 'wb') as f:
#         pickle.dump(trajectories, f)
