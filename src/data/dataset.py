# 🎯 IMPLEMENTAR: PyTorch Dataset customizado
# Código de referencia en el TP - adaptar

from torch.utils.data import Dataset
import torch
import numpy as np

class RecommendationDataset(Dataset):
    """
    Dataset para entrenar Decision Transformer.
    """
    def __init__(self, trajectories, context_length=20):
        """
        Args:
            trajectories: Lista de dicts con formato de create_dt_dataset()
            context_length: Ventana de contexto (cuántos timesteps usar)
        """
        # TODO: Guardar trajectories y context_length
        pass
    
    def __len__(self):
        # TODO: Retornar número de trayectorias
        pass
    
    def __getitem__(self, idx):
        """
        Retorna un sample para training.
        
        Returns:
            Dict con keys:
                - states: (context_length,) LongTensor de item IDs
                - actions: (context_length,) LongTensor de item IDs  
                - rtg: (context_length, 1) FloatTensor de returns-to-go
                - timesteps: (context_length,) LongTensor de posiciones
                - groups: () LongTensor del grupo del usuario
                - targets: (context_length,) LongTensor - next items a predecir
        """
        # TODO: Ver código de referencia en el TP
        # Hint: Extraer ventana de la trayectoria
        # Hint: Targets son los items shifted (próximo item a predecir)
        pass