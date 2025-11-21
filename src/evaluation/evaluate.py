# 🎯 IMPLEMENTAR: Evaluación del modelo
# Código muy detallado en el TP - seguir esa guía

@torch.no_grad()
def evaluate_model(model, test_data, device, target_return=None, k_list=[5, 10, 20]):
    """
    Evalúa el modelo en test set (cold-start users).
    
    Ver código completo en 03_REFERENCIA_COMPLETA.md
    """
    model.eval()
    
    # TODO: Seguir lógica del TP:
    # 1. Para cada usuario de test
    # 2. Simular sesión: empezar con history vacío
    # 3. Ir "recomendando" items y observando ratings
    # 4. Calcular métricas
    
    pass