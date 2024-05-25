# Autora:  Marta María Álvarez Crespo
# Descripción:  Archivo de ejecución de un experimento de aprendizaje automático sobre imágenes utilizando modelos de redes neuronales convolucionales (CNN).
# Última modificación: 25 / 05 / 2024
# GitHub: www.github.com/marta-maria-alvarez-crespo/MIIR2324-Python-Avanzado-Trabajo-Final

import os
import json
import cProfile
import fnc
import funciones_datos
import numpy as np
import pandas as pd
import pstats


def main():

    configuracion = json.load(open("scripts/configuracion.json", "r", encoding="UTF-8"))

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

    
    # Ejecución de los experimentos de Transfer Learning configurados
    configuraciones, df_or, df_norm, df_preprocesado = fnc.ejecuta_experimentos_transfer_learning(
        et_filtradas=et_filtradas,
        pred_entrenamiento_or=pred_entrenamiento_or,
        pred_test_or=pred_test_or,
        target_entrenamiento=target_entrenamiento,
        target_test=target_test,
        v=[0, 1, 0],
        mw=4
    )

    # Compara los experimentos y devuelve la combinación de la mejor red y configuración según los resultados obtenidos
    nombre_cnn, nombre_top, clave_cnn = fnc.selecciona_mejor_cnn(df_or, df_norm, df_preprocesado)

    # Experimentación de Fine Tunning con los parámetros óptimos y almacenamiento de los resultados en un dataframe
    fnc.ejecuta_fine_tunning_mejor_cnn(im_filtradas, et_filtradas, nombre_cnn, configuraciones, nombre_top, clave_cnn)


if __name__ == "__main__":
    cProfile.run("main()", filename="output.pstats", sort="cumulative")

    stats = pstats.Stats("output.pstats")

    stats_data = []
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, line, funcname = func
        stats_data.append(
            {"filename": filename, "line": line, "funcname": funcname, "cc": cc, "nc": nc, "tt": tt, "ct": ct}
        )

    df = pd.DataFrame(stats_data)

    dir_path = os.path.dirname(os.path.abspath(__file__))
    resultados_path = os.path.join(dir_path, "../Medicion_de_tiempos")
    if not os.path.exists(resultados_path):
        os.makedirs(resultados_path)

    df.to_excel(os.path.join(resultados_path, "profiling.xlsx"))
