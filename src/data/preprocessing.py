import numpy as np

def create_dt_dataset(df_train):
    """
    Convierte DataFrame raw a formato Decision Transformer.
    
    Args:
        df_train: DataFrame con [user_id, user_group, items, ratings]
    
    Returns:
        trajectories: List[Dict] con formato específico
    """
    trajectories = []
    
    for idx, row in df_train.iterrows():
        # Extraer items, ratings, group
        items = row['items']        # numpy array de item IDs
        ratings = row['ratings']    # numpy array de ratings
        group = row['user_group']   # int (0-7)

        # Calcular returns-to-go (R̂)
        returns = np.zeros(len(ratings))
        returns[-1] = ratings[-1]
        for t in range(len(ratings)-2, -1, -1):
            returns[t] = ratings[t] + returns[t+1]
        
        
        # Crear diccionario con formato correcto
        trajectory = {
            'items': items,                        # Secuencia de películas
            'ratings': ratings,                    # Ratings correspondientes
            'returns_to_go': returns,              # R̂ para cada timestep
            'timesteps': np.arange(len(items)),    # [0, 1, 2, ..., T-1]
            'user_group': group                    # Cluster del usuario
        }
        
        trajectories.append(trajectory)
    
    return trajectories


def validate_preprocessing(trajectories):
    """
    Valida que el preprocesamiento sea correcto.
    """
    for traj in trajectories:
        assert all(key in traj for key in ['items', 'ratings', 'returns_to_go', 'timesteps', 'user_group']), "Faltan keys en la trayectoria"
        n = len(traj['items'])
        assert n == len(traj['ratings']) == len(traj['returns_to_go']), "Longitudes inconsistentes"
        assert traj['returns_to_go'][0] == np.sum(traj['ratings']), "Returns-to-go inicial incorrecto"
        assert traj['returns_to_go'][-1] == traj['ratings'][-1], "Returns-to-go final incorrecto"

