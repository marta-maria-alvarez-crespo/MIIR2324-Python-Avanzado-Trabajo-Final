# Autora:  Marta María Álvarez Crespo
# Descripción:  Archivo de ejecución de un experimento de aprendizaje automático sobre imágenes utilizando modelos de redes neuronales convolucionales (CNN).
# Última modificación: 25 / 05 / 2024
# GitHub: www.github.com/marta-maria-alvarez-crespo/MIIR2324-Python-Avanzado-Trabajo-Final

import os
import fnc
import funciones_datos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import repeat


def medicion_de_tiempos(
    et_filtradas,
    pred_entrenamiento_or,
    pred_test_or,
    target_entrenamiento,
    target_test,
    min_w=2,
    max_w=8,
    repeticiones=10,
):
    """Función que permite la experimentación con varios hilos o procesos (por implementar) y la obtención de un estudio estadístico sencillo

    :param et_filtradas: Lista de etiquetas filtradas.
    :type et_filtradas: list
    :param pred_entrenamiento_or: Predicciones de entrenamiento originales.
    :type pred_entrenamiento_or: numpy.ndarray
    :param pred_test_or: Predicciones de prueba originales.
    :type pred_test_or: numpy.ndarray
    :param target_entrenamiento: Objetivos de entrenamiento.
    :type target_entrenamiento: numpy.ndarray
    :param target_test: Objetivos de prueba.
    :type target_test: numpy.ndarray
    :param min_w: Número mínimo de workers, por defecto es 2.
    :type min_w: int, opcional
    :param max_w: Número máximo de workers, por defecto es 8.
    :type max_w: int, opcional
    :param repeticiones: Número de repeticiones, por defecto es 10.
    :type repeticiones: int, opcional
    :return: Una tupla que contiene el tiempo y las etiquetas.
    :rtype: tuple
    """
    v = 0
    tiempo = []
    max_workers = []
    for v in [[0, 1, 0], [1, 0, 0], [0, 0, 1]]:
        if v[1] == 1:
            while min_w <= max_w:
                tiempos = repeat(
                    "fnc.ejecuta_experimentos_transfer_learning(et_filtradas, pred_entrenamiento_or, pred_test_or, target_entrenamiento, target_test, v, min_w)",
                    repeat=repeticiones,
                    number=1,
                    globals={
                        "et_filtradas": et_filtradas,
                        "pred_entrenamiento_or": pred_entrenamiento_or,
                        "pred_test_or": pred_test_or,
                        "target_entrenamiento": target_entrenamiento,
                        "target_test": target_test,
                        "v": v,
                        "min_w": min_w,
                        "fnc": fnc,
                    },
                )
                tiempo.extend(tiempos)
                max_workers.extend([min_w] * len(tiempos))
                min_w += 1
        else:
            tiempos = repeat(
                "fnc.ejecuta_experimentos_transfer_learning(et_filtradas, pred_entrenamiento_or, pred_test_or, target_entrenamiento, target_test, v, min_w)",
                repeat=repeticiones,
                number=1,
                globals={
                    "et_filtradas": et_filtradas,
                    "pred_entrenamiento_or": pred_entrenamiento_or,
                    "pred_test_or": pred_test_or,
                    "target_entrenamiento": target_entrenamiento,
                    "target_test": target_test,
                    "v": v,
                    "min_w": min_w,
                    "fnc": fnc,
                },
            )
            tiempo.extend(tiempos)

    labels = (
        [f"ThreadPoolExecutor (max workers={workers})" for workers in max_workers]
        + ["Clase Thread"] * repeticiones
        + ["Secuencial"] * repeticiones
    )
    # Crea un DataFrame de pandas
    df = pd.DataFrame({"Prueba": labels, "Tiempo": tiempo})

    # Calcula la media, el máximo, el mínimo y la desviación típica por grupo
    df_agregado = df.groupby("Prueba").agg(["mean", "max", "min", "std"])

    dir_path = os.path.dirname(os.path.abspath(__file__))
    resultados_path = os.path.join(dir_path, "../Medicion_de_tiempos")
    if not os.path.exists(resultados_path):
        os.makedirs(resultados_path)

    df.to_excel(os.path.join(resultados_path, "medicion_de_tiempos_2.xlsx"))
    df_agregado.to_excel(os.path.join(resultados_path, "estudio_estadistico_2.xlsx"))
    return tiempo, labels


def generacion_grafica_comparativa(tiempo, labels):
    """Recoge los resultados obtenidos tras la ejecución del estudio temporal y genera una gráfica comparativa

    :param tiempo: Lista de tiempos de ejecución para cada tipo de ejecución
    :type tiempo: list
    :param labels: Lista de etiquetas para cada tipo de ejecución
    :type labels: list
    """
    plt.figure(figsize=(30, 8))
    plt.bar(labels, tiempo)
    plt.xticks(rotation=25, fontsize=8)
    plt.xlabel("Tipo de Ejecución")
    plt.ylabel("Tiempo de Ejecución (s)")

    dir_path = os.path.dirname(os.path.abspath(__file__))
    resultados_path = os.path.join(dir_path, "../Medicion_de_tiempos")
    if not os.path.exists(resultados_path):
        os.makedirs(resultados_path)

    plt.savefig(os.path.join(resultados_path, "grafica_2.png"))


def main():
    """
    Función principal del programa.

    Carga el dataset, realiza la preparación de los datos de entrada, mide los tiempos de ejecución de los experimentos
    de Transfer Learning, genera una gráfica comparativa, ejecuta los experimentos de Transfer Learning, selecciona la mejor
    red y configuración, realiza la experimentación de Fine Tunning y almacena los resultados en un dataframe.
    """
    
    # Carga del dataset
    im_filtradas, et_filtradas = funciones_datos.cargar_dataset()

    im_filtradas = np.concatenate((im_filtradas[:150], im_filtradas[2000:2150], im_filtradas[-150:]))
    et_filtradas = np.concatenate((et_filtradas[:150], et_filtradas[2000:2150], et_filtradas[-150:]))

    # División de los datos de entrada en conjuntos de entrenamiento y validación
    (
        pred_entrenamiento_or,
        pred_test_or,
        target_entrenamiento,
        target_test,
    ) = fnc.division_preparacion_datos_entrada(im_filtradas=im_filtradas, et_filtradas=et_filtradas)

    # Medicion de tiempos de ejecución de los experimentos de Transfer Learning configurados
    tiempo, labels = medicion_de_tiempos(
        et_filtradas,
        pred_entrenamiento_or,
        pred_test_or,
        target_entrenamiento,
        target_test,
        min_w=2,
        max_w=16,
        repeticiones=10,
    )

    generacion_grafica_comparativa(tiempo, labels)

    # Ejecución de los experimentos de Transfer Learning configurados
    configuraciones, df_or, df_norm, df_preprocesado = fnc.ejecuta_experimentos_transfer_learning(
        et_filtradas=et_filtradas,
        pred_entrenamiento_or=pred_entrenamiento_or,
        pred_test_or=pred_test_or,
        target_entrenamiento=target_entrenamiento,
        target_test=target_test,
    )

    # Compara los experimentos y devuelve la combinación de la mejor red y configuración según los resultados obtenidos
    nombre_cnn, nombre_top, clave_cnn = fnc.selecciona_mejor_cnn(df_or, df_norm, df_preprocesado)

    # Experimentación de Fine Tunning con los parámetros óptimos y almacenamiento de los resultados en un dataframe
    fnc.ejecuta_fine_tunning_mejor_cnn(im_filtradas, et_filtradas, nombre_cnn, configuraciones, nombre_top, clave_cnn)


if __name__ == "__main__":
    main()
