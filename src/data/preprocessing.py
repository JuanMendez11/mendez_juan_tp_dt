import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class DTDataset(Dataset):
    """
    Dataset para Decision Transformer.
    Mismo que en la implementación PyTorch pura.
    """
    
    def __init__(self, data, max_len=20):
        """
        Args:
            data: lista de diccionarios con:
                  {'states', 'actions', 'rtgs', 'timesteps', 'groups', 'attention_mask'}
            max_len: longitud máxima de secuencia (padding)
        """
        self.data = data
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        episode = self.data[idx]
        
        # Extraer datos
        states = episode['states']
        actions = episode['actions']
        rtgs = episode['rtgs']
        timesteps = episode['timesteps']
        group = episode['group']
        
        seq_len = len(actions)
        
        # Padding si es necesario
        if seq_len < self.max_len:
            pad_len = self.max_len - seq_len
            
            states = np.concatenate([states, np.zeros((pad_len, 1))])
            actions = np.concatenate([actions, np.zeros(pad_len)])
            rtgs = np.concatenate([rtgs, np.zeros((pad_len, 1))])
            timesteps = np.concatenate([timesteps, np.zeros(pad_len)])
            
            # Attention mask: 1 = válido, 0 = padding
            attention_mask = np.concatenate([
                np.ones(seq_len),
                np.zeros(pad_len)
            ])
        else:
            attention_mask = np.ones(seq_len)
        
        return {
            'states': torch.FloatTensor(states),
            'actions': torch.LongTensor(actions),
            'rtgs': torch.FloatTensor(rtgs),
            'timesteps': torch.LongTensor(timesteps),
            'groups': torch.LongTensor([group]),
            'attention_mask': torch.FloatTensor(attention_mask)
        }


def create_dt_dataset(df_train, max_len=20):
    """
    Convierte DataFrame a formato Decision Transformer.
    """
    data = []
    
    for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Creating dataset"):
        items = row['items']
        ratings = row['ratings']
        group = row['user_group']
        
        seq_len = len(items)
        
        # Calcular return-to-go (suma acumulada inversa)
        returns = np.array(ratings, dtype=np.float32)
        rtgs = np.cumsum(returns[::-1])[::-1]  # suma acumulada desde el final
        
        # Preparar tensores
        states = ratings.reshape(-1, 1).astype(np.float32)
        actions = np.array(items, dtype=np.int64)
        rtgs = rtgs.reshape(-1, 1).astype(np.float32)
        timesteps = np.arange(seq_len, dtype=np.int64)
        
        # Truncar si es muy largo
        if seq_len > max_len:
            states = states[-max_len:]
            actions = actions[-max_len:]
            rtgs = rtgs[-max_len:]
            timesteps = timesteps[-max_len:]
        
        data.append({
            'states': states,
            'actions': actions,
            'rtgs': rtgs,
            'timesteps': timesteps,
            'group': group
        })
    
    return data