# Trabajo práctico: Decision Transformers

## Juan Ignacio Méndez

Este es un trabajo para la materia optativa **Aprendizaje por refuerzos** de la _diplomatura en ciencias de datos, aprendizaje automático y sus aplicaciones_ del año 2025. 

Para poder ejecutar los scripts y las notebooks es necesario que después de clonar el repositorio se posicionen sobre `mendez_juan_tp_dt` y ejecutar los siguientes comandos:

- `pip install -r requirements.txt`

- `pip install -e .`

El primero es para instalar todas las librerias necesarias para poder correr el código, y el segundo es para no tener problemas con los caminos que se encuentran en algunos archivos.

Para entrenar en modelo de decision tranformer se puede ejecutar `python3 scripts/train.py`. Lo que hace este comando es entrenar este modelo y guardar los pesos en results/chekpoints/decision_transformer_netflix.pt y los resultados en results/logs/history_netflix.pkl