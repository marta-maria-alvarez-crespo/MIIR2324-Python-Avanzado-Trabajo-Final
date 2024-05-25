# Autora:  Marta María Álvarez Crespo
# Descripción:  Archivo de ejecución de un experimento de aprendizaje automático sobre imágenes utilizando modelos de redes neuronales convolucionales (CNN).
# Última modificación: 23 / 05 / 2024

import funciones_datos
import fnc
import json
import numpy as np
from timeit import timeit
import pandas as pd
import cProfile
import matplotlib.pyplot as plt

configuracion = json.load(open("scripts/configuracion.json", "r", encoding="UTF-8"))

if __name__ == "__main__":
    # Carga del dataset
    im_filtradas, et_filtradas = funciones_datos.cargar_dataset()

    im_filtradas = np.concatenate((im_filtradas[:50], im_filtradas[-50:]))
    et_filtradas = np.concatenate((et_filtradas[:50], et_filtradas[-50:]))

    # División de los datos de entrada en conjuntos de entrenamiento y validación
    (
        pred_entrenamiento_or,
        pred_test_or,
        target_entrenamiento,
        target_test,
    ) = fnc.division_preparacion_datos_entrada(im_filtradas=im_filtradas, et_filtradas=et_filtradas)

    tiempo = []
    mw = 4
    max_workers =[]
    for v in [[0, 1, 0], [1, 0, 0], [0, 0, 1]]:
        if v[1] == 1:
            while mw <= 4:
                tiempo.append(
                    timeit(
                        "fnc.ejecuta_experimentos_transfer_learning(et_filtradas, pred_entrenamiento_or, pred_test_or, target_entrenamiento, target_test, v, mw)",
                        number=1,
                        globals=globals(),
                    )
                )
                max_workers.append(mw)
                mw += 1
        else:
            tiempo.append(
                timeit(
                    "fnc.ejecuta_experimentos_transfer_learning(et_filtradas, pred_entrenamiento_or, pred_test_or, target_entrenamiento, target_test, v, mw)",
                    number=1,
                    globals=globals(),
                )
            )
            

    for t in tiempo:
        print("=" * 50)
        print("tardé ", t, " segundos")
        print("=" * 50)
    
    labels =[f"ThreadPoolExecutor (max workers={workers})" for workers in max_workers] + ["Clase Thread"] + ["Secuencial"]
    # Crea un DataFrame de pandas
    df = pd.DataFrame({
        'Prueba': labels,
        'Tiempo': tiempo
    })
    
    print(df)

    # Calcula la media, el máximo y el mínimo
    media = df['Tiempo'].mean()
    maximo = df['Tiempo'].max()
    minimo = df['Tiempo'].min()
    
    plt.bar(labels, tiempo)
    plt.xticks(rotation=45)
    plt.xlabel("Tipo de Ejecución")
    plt.ylabel("Tiempo de Ejecución (s)")
    plt.show()

    # # Ejecución de los experimentos de Transfer Learning configurados
    # configuraciones, df_or, df_norm, df_preprocesado = fnc.ejecuta_experimentos_transfer_learning(
    #     et_filtradas=et_filtradas,
    #     pred_entrenamiento_or=pred_entrenamiento_or,
    #     pred_test_or=pred_test_or,
    #     target_entrenamiento=target_entrenamiento,
    #     target_test=target_test,
    # )

    # # Compara los experimentos y devuelve la combinación de la mejor red y configuración según los resultados obtenidos
    # nombre_cnn, nombre_top, clave_cnn = fnc.selecciona_mejor_cnn(df_or, df_norm, df_preprocesado)

    # # Experimentación de Fine Tunning con los parámetros óptimos y almacenamiento de los resultados en un dataframe
    # fnc.ejecuta_fine_tunning_mejor_cnn(im_filtradas, et_filtradas, nombre_cnn, configuraciones, nombre_top, clave_cnn)
