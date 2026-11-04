# SVD y SVM lineal para clasificar tuberculosis en radiografías

Prueba piloto de la ponencia de Metodología de la Investigación (UDFJC).
Compara un SVM lineal sobre píxeles aplanados contra un SVM sobre los
primeros k valores singulares de cada imagen.

## Datos
TB Chest X-ray Database (Kaggle):
https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
Descomprimir en una carpeta `datos` con las subcarpetas Normal y Tuberculosis.
Las imágenes no se incluyen en este repositorio.

## Ejecución
pip install numpy pandas matplotlib pillow scikit-learn
python NOMBRE_DEL_SCRIPT.py --data datos --n_per_class 150

## Resultados
La carpeta `resultados_piloto` contiene las métricas de validación (CSV),
el resumen con los datos del equipo (JSON) y las figuras.
Son resultados de una prueba piloto con muestra reducida, no del protocolo completo.
