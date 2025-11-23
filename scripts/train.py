import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# === CONFIGURACIÓN DEL DATASET ===
# Cambiar aquí para elegir Netflix o Goodreads
DATASET = 'netflix'  # o 'goodreads'

if DATASET == 'netflix':
    NUM_ITEMS = 752
    train_path = 'data/train/netflix8_train.df'
    test_path = 'data/test_users/netflix8_test.json'
else:
    NUM_ITEMS = 472
    train_path = 'data/train/goodreads8_train.df'
    test_path = 'data/test_users/goodreads8_test.json'

NUM_GROUPS = 8


from src.data.load_data import create_dt_dataset, DTDataset
from src.models.decision_transformer import DecisionTransformer
from src.training.trainer import train_decision_transformer


def main():
    """Script completo para entrenar y evaluar"""
    
    print("="*60)
    print("DECISION TRANSFORMER TRAINING")
    print("="*60)
    
    # === 1. CARGAR DATOS ===
    print("\n[1/5] Cargando datos...")
    
    with open(train_path, 'rb') as f:
        df_train = pickle.load(f)
    
    import json
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    print(f"Train users: {len(df_train)}")
    print(f"Test users: {len(test_data)}")
    print(f"Num items: {NUM_ITEMS}")
    
    # === 2. CREAR DATASET ===
    print("\n[2/5] Creando dataset...")
    
    max_len = 20  # longitud máxima de secuencia
    train_data = create_dt_dataset(df_train, max_len=max_len)
    train_dataset = DTDataset(train_data, max_len=max_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    
    # === 3. CREAR MODELO ===
    print("\n[3/5] Creando modelo...")
    
    model = DecisionTransformer(
        num_items=NUM_ITEMS,
        num_groups=NUM_GROUPS,
        hidden_dim=128,
        n_layers=3,
        n_heads=4,
        context_length=max_len,
        max_timestep=200,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # === 4. ENTRENAR ===
    print("\n[4/5] Entrenando modelo...")
    
    model = train_decision_transformer(
        model=model,
        train_loader=train_loader,
        device=device,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        num_epochs=50
    )
    
    # Guardar modelo
    torch.save(model.state_dict(), f'decision_transformer_{DATASET}.pt')
    print(f"Modelo guardado: decision_transformer_{DATASET}.pt")
    
    # === 5. EVALUAR ===
    # print("\n[5/5] Evaluando en cold-start...")
    
    # results = evaluate_cold_start(
    #     model=model,
    #     test_data=test_data,
    #     device=device,
    #     target_return=None,  # usar máximo posible
    #     k_list=[5, 10, 20]
    # )
    
    # print("\n" + "="*60)
    # print("RESULTADOS FINALES")
    # print("="*60)
    # for metric, value in results.items():
    #     print(f"{metric:12s}: {value:.4f}")
    # print("="*60)


if __name__ == '__main__':
    main()