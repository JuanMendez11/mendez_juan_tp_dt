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
