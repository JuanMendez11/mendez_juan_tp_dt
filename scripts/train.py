import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import pickle
import os
import json

from src.data.load_data import load_training_data
from src.data.preprocessing import create_dt_dataset, validate_preprocessing
from src.data.dataset import RecommendationDataset
from src.models.decision_transformer import DecisionTransformer
from src.training.trainer import train_decision_transformer
from src.evaluation.evaluate import evaluate_model


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



def main():
    """Script completo para entrenar"""
    
    print("="*60)
    print("DECISION TRANSFORMER TRAINING")
    print("="*60)
    
    # === 1. CARGAR DATOS ===
    print("\n[1/5] Cargando datos...")
    
    with open(train_path, 'rb') as f:
        df_train = pickle.load(f)
    
    print(f"Train users: {len(df_train)}")
    print(f"Num items: {NUM_ITEMS}")
    
    # === 2. CREAR DATASET ===
    print("\n[2/5] Creando dataset...")
    
    trajectories = create_dt_dataset(df_train)
    validate_preprocessing(trajectories)
    print("Trayectorias procesadas correctamente.")
    
    dataset = RecommendationDataset(trajectories, context_length=20)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Verificar un batch
    batch = next(iter(train_loader))
    print(f"Keys: {batch.keys()}")
    print(f"States shape: {batch['states'].shape}")  # (64, 20)
    
    # === 3. CREAR MODELO ===
    print("\n[3/5] Creando modelo...")
    
    model = DecisionTransformer(
        num_items=NUM_ITEMS,
        num_groups=NUM_GROUPS,
        hidden_dim=128,
        n_layers=3,
        n_heads=4
    )

    print(f"Parámetros totales: {sum(p.numel() for p in model.parameters())}")
    # Debería ser ~10-20M parámetros
    
    
    # === 4. ENTRENAR ===
    print("\n[4/5] Entrenando modelo...")

    # Configuración del dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train, history = train_decision_transformer(
        model=model,
        train_loader=train_loader,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        device=device,
        num_epochs=50
    )
    

    # === 5. GUARDAR MODELO ===
    print("\n[5/5] Guardando modelo...")

    # Crear directorios si no existen
    checkpoint_dir = 'results/checkpoints'
    logs_dir = 'results/logs'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Guardar modelo
    model_path = os.path.join(checkpoint_dir, f'decision_transformer_{DATASET}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado: {model_path}")
    
    # Guardar history
    history_path = os.path.join(logs_dir, f'history_{DATASET}.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"History guardado: {history_path}")


    # === 6. EVALUAR ===
    print("\nEvaluando en cold-start...")
    
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    print(f"Test users: {len(test_data)}")

    results = evaluate_model(
        model=model,
        test_data=test_data,
        device=device,
        target_return=None,  # usar máximo posible
        k_list=[5, 10, 20]
    )
    
    print("\n" + "="*60)
    print("RESULTADOS FINALES")
    print("="*60)
    for metric, value in results.items():
        print(f"{metric:12s}: {value:.4f}")
    print("="*60)

    
    


if __name__ == '__main__':
    main()