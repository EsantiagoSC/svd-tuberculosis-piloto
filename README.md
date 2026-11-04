# SVD y SVM lineal para clasificar tuberculosis en radiografías

Prueba piloto de la ponencia de Metodología de la Investigación (Universidad Distrital Francisco José de Caldas).
Compara un SVM lineal sobre los píxeles aplanados (referencia) contra un SVM sobre los primeros k valores singulares de cada imagen.

Autores: Edilson Santiago Sepúlveda Cortés, Samuel Alberto Leal Sáenz y Andrés Felipe García Vargas.

## Datos

TB Chest X-ray Database (Kaggle): https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset

Descomprimir en una carpeta `datos` que contenga las subcarpetas `Normal` y `Tuberculosis`. Las imágenes no se incluyen en este repositorio.

## Ejecución

```bash
pip install numpy pandas matplotlib pillow scikit-learn
python prueba_piloto.py --data datos --n_per_class 150
```

El piloto usa 150 imágenes por clase (300 en total), semilla 42, partición estratificada 70/15/15 y k ∈ {5, 10, 20, 50}. Corre solo en CPU.

## Resultados

La carpeta `resultados_piloto` contiene las métricas de validación (`metricas_validacion.csv`), el resumen de la prueba con los datos del equipo (`resumen.json`) y las figuras.
Son resultados de una prueba piloto con una muestra reducida, no del protocolo completo.
