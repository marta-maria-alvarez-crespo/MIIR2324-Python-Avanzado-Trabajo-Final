# Autora:  Marta María Álvarez Crespo
# Descripción: Funciones necesarias para la ejecución de los experimentos de Transfer Learning y Fine-Tuning.
# Última modificación: 25 / 05 / 2024
# GitHub: www.github.com/marta-maria-alvarez-crespo/MIIR2324-Python-Avanzado-Trabajo-Final


import os
import json
import deep_learning
import funciones_datos
import pandas as pd
from scripts.procesado_paralelo import MiHilo, MiProceso
from multiprocess import Queue
from tensorflow.keras import optimizers
from pathos.multiprocessing import ProcessPool
from preprocesado import imagenes_preprocesadas
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split


configuracion = json.load(open("scripts/configuracion.json", "r", encoding="UTF-8"))

#  Creación de un diccionario con las CNN pre-entrenadas que se deseen cargar
cnn_preentrenadas = {"mn": deep_learning.cargar_mn, "vgg": deep_learning.cargar_vgg}


def division_preparacion_datos_entrada(im_filtradas, et_filtradas):
    """Divide los datos de entrada en conjuntos de entrenamiento y prueba y aplica data augmentation a las imágenes de entrenamiento.

    :param im_filtradas: Imágenes filtradas.
    :type im_filtradas: list
    :param et_filtradas: Etiquetas filtradas.
    :type et_filtradas: list
    :return: Tupla que contiene los conjuntos de entrenamiento y prueba de las imágenes filtradas y las etiquetas filtradas.
    :rtype: tuple
    """
    # División de datos de entrenamiento y prueba
    (pred_entrenamiento_or, pred_test_or, target_entrenamiento, target_test) = train_test_split(
        im_filtradas, et_filtradas, test_size=0.2, shuffle=True, random_state=configuracion["parametros_top"]["seed"]
    )

    # Data Augmentation de las imágenes
    da = funciones_datos.data_augmentation(im_filtradas.shape[1:])

    # Realiza una predicción utilizando una red neuronal convolucional
    pred_entrenamiento_or = funciones_datos.cnn_predict(pred_entrenamiento_or, "pred_entrenamiento_", da, "or")

    return (
        pred_entrenamiento_or,
        pred_test_or,
        target_entrenamiento,
        target_test,
    )


# TODO quitar la v de la función para la entrega final
def ejecuta_experimentos_transfer_learning(
    et_filtradas, pred_entrenamiento_or, pred_test_or, target_entrenamiento, target_test, v, mw
):
    """Ejecuta experimentos de Transfer Learning utilizando los parámetros proporcionados y almacena los resultados en un dataframe.

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
    # :param v: Lista de valores booleanos para seleccionar el método de ejecución.
    # :type v: list
    :param mw: Valor para el número máximo de hilos de ejecución para multihilo_pool_executor.
    :type mw: int
    :return: Un diccionario de configuraciones, y tres dataframes con los resultados.
    :rtype: tuple
    """

    configuraciones = {
        "mn": {"im_or": {}, "im_norm": {}, "im_preprocesadas": {}},
        "vgg": {"im_or": {}, "im_norm": {}, "im_preprocesadas": {}},
    }

    # Experimentación de Transfer Learning con los parámetros establecidos y almacenamiento de los resultados en el dataframe creado
    if v[0]:
        df_tl_or, configuraciones = multihilo_clase_thread(
            et_filtradas, pred_entrenamiento_or, pred_test_or, target_entrenamiento, target_test, configuraciones
        )
    elif v[1]:
        df_tl_or, configuraciones = multihilo_pool_executor(
            et_filtradas, pred_entrenamiento_or, pred_test_or, target_entrenamiento, target_test, configuraciones, mw
        )
    elif v[2]:
        df_tl_or, configuraciones = secuencial(
            et_filtradas, pred_entrenamiento_or, pred_test_or, target_entrenamiento, target_test, configuraciones
        )

    # Guardado de los datos originales en Excel
    df_mini = df_tl_or.set_index("Modelo de entrenamiento utilizado")

    df_or = df_mini[df_mini["Tipo de imagen"] == "im_or"]
    df_norm = df_mini[df_mini["Tipo de imagen"] == "im_norm"]
    df_preprocesado = df_mini[df_mini["Tipo de imagen"] == "im_preprocesadas"]

    dir_path = os.path.dirname(os.path.abspath(__file__))
    resultados_path = os.path.join(dir_path, "../Resultados_Dataframes")
    if not os.path.exists(resultados_path):
        os.makedirs(resultados_path)

    df_or.to_excel(os.path.join(resultados_path, "resultados_transfer_learning_original.xlsx"))
    df_norm.to_excel(os.path.join(resultados_path, "resultados_transfer_learning_normalizado.xlsx"))
    df_preprocesado.to_excel(os.path.join(resultados_path, "resultados_transfer_learning_preprocesado.xlsx"))

    return configuraciones, df_or, df_norm, df_preprocesado


def multihilo_clase_thread(
    et_filtradas, pred_entrenamiento_or, pred_test_or, target_entrenamiento, target_test, configuraciones
):
    """Ejecuta el entrenamiento de una red neuronal en múltiples hilos.

    Este método recibe los datos de entrada necesarios para entrenar una red neuronal en múltiples configuraciones.
    Cada configuración consiste en una combinación de modelos de redes neuronales pre-entrenadas y parámetros de la capa superior.
    El método crea hilos para cada configuración y ejecuta el entrenamiento de la red neuronal en paralelo.
    Al finalizar, se obtienen los resultados de cada hilo y se devuelve un dataframe con los resultados obtenidos.

    :param et_filtradas: Lista de etiquetas filtradas.
    :type et_filtradas: list
    :param pred_entrenamiento_or: Predicciones de entrenamiento originales.
    :type pred_entrenamiento_or: numpy.ndarray
    :param pred_test_or: Predicciones de prueba originales.
    :type pred_test_or: numpy.ndarray
    :param target_entrenamiento: Etiquetas de entrenamiento.
    :type target_entrenamiento: numpy.ndarray
    :param target_test: Etiquetas de prueba.
    :type target_test: numpy.ndarray
    :param configuraciones: Configuraciones de modelos y parámetros de la capa superior.
    :type configuraciones: diccionario
    :return: Un dataframe con los resultados obtenidos en el entrenamiento de la red neuronal y las configuraciones utilizadas.
    :rtype: pandas.DataFrame, dict
    """
    hilos = []
    for n_cnn, pruebas in configuraciones.items():
        cnn = cnn_preentrenadas[n_cnn](pred_entrenamiento_or.shape[1:])
        for prueba in pruebas:
            prueba_mas_cnn = prueba + "_" + n_cnn
            predictores_train = imagenes_preprocesadas[prueba_mas_cnn](pred_entrenamiento_or, prueba_mas_cnn)
            predictores_train = funciones_datos.cnn_predict(predictores_train, "entrenamiento", cnn, n_cnn)
            predictores_test = imagenes_preprocesadas[prueba_mas_cnn](pred_test_or, prueba_mas_cnn)
            predictores_test = funciones_datos.cnn_predict(predictores_test, "validacion", cnn, n_cnn)

            for neurona in configuracion["parametros_top"]["neuronas"]:
                for dropout in configuracion["parametros_top"]["dropouts"]:
                    for activacion in configuracion["parametros_top"]["activaciones"]:
                        for capa in configuracion["parametros_top"]["capas"]:
                            hilo = MiHilo(
                                target=entrenar_red,
                                args=(
                                    et_filtradas,
                                    target_entrenamiento,
                                    target_test,
                                    n_cnn,
                                    predictores_train,
                                    predictores_test,
                                    neurona,
                                    dropout,
                                    activacion,
                                    capa,
                                    configuracion["parametros_top"]["transfer_learning"]["max_epoch"],
                                    prueba,
                                ),
                            )
                            hilo.start()
                            hilos.append(hilo)

    for hilo in hilos:
        hilo.join()

    # Creación de un dataframe con los resultados obtenidos en Transfer Learning
    df_tl_or = pd.DataFrame()
    for hilo in hilos:
        diccionario = hilo.get_result()
        df_tl_or, configuraciones = obtener_resultados(configuraciones, df_tl_or, diccionario)
    return df_tl_or, configuraciones


def multihilo_pool_executor(
    et_filtradas, pred_entrenamiento_or, pred_test_or, target_entrenamiento, target_test, configuraciones, mw
):
    """Ejecuta con múltiples hilos utilizando ThreadPoolExecutor.

    Este método ejecuta múltiples hilos utilizando ThreadPoolExecutor para entrenar redes neuronales convolucionales
    con diferentes configuraciones. Toma como entrada los datos de entrenamiento y prueba, las configuraciones de las
    redes neuronales, y el número máximo de trabajadores (hilos) a utilizar.

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
    :param configuraciones: Configuraciones de las redes neuronales.
    :type configuraciones: dict
    :param mw: Número máximo de trabajadores (hilos) a utilizar.
    :type mw: int
    :return: Un DataFrame con los resultados de las redes neuronales y las configuraciones actualizadas.
    :rtype: pandas.DataFrame, dict
    """
    futures = []
    with ThreadPoolExecutor(max_workers=mw) as pool:
        for n_cnn, pruebas in configuraciones.items():
            cnn = cnn_preentrenadas[n_cnn](pred_entrenamiento_or.shape[1:])
            for prueba in pruebas:
                prueba_mas_cnn = prueba + "_" + n_cnn
                predictores_train = imagenes_preprocesadas[prueba_mas_cnn](pred_entrenamiento_or, prueba_mas_cnn)
                predictores_train = funciones_datos.cnn_predict(predictores_train, "entrenamiento", cnn, n_cnn)
                predictores_test = imagenes_preprocesadas[prueba_mas_cnn](pred_test_or, prueba_mas_cnn)
                predictores_test = funciones_datos.cnn_predict(predictores_test, "validacion", cnn, n_cnn)
                for neurona in configuracion["parametros_top"]["neuronas"]:
                    for dropout in configuracion["parametros_top"]["dropouts"]:
                        for activacion in configuracion["parametros_top"]["activaciones"]:
                            for capa in configuracion["parametros_top"]["capas"]:
                                futures.append(
                                    pool.submit(
                                        entrenar_red,
                                        et_filtradas,
                                        target_entrenamiento,
                                        target_test,
                                        n_cnn,
                                        predictores_train,
                                        predictores_test,
                                        neurona,
                                        dropout,
                                        activacion,
                                        capa,
                                        configuracion["parametros_top"]["transfer_learning"]["max_epoch"],
                                        prueba,
                                    )
                                )
        results = [f.result() for f in futures]

    df_tl_or = pd.DataFrame()
    for result in results:
        df_tl_or, configuraciones = obtener_resultados(configuraciones, df_tl_or, result)
    return df_tl_or, configuraciones


def multiproceso_pool_executor(
    et_filtradas, pred_entrenamiento_or, pred_test_or, target_entrenamiento, target_test, configuraciones
):
    """Ejecuta el proceso de entrenamiento de una red neuronal convolucional utilizando multiprocessing.

    :param et_filtradas: Lista de etiquetas filtradas.
    :type et_filtradas: list
    :param pred_entrenamiento_or: Matriz de predictores de entrenamiento originales.
    :type pred_entrenamiento_or: numpy.ndarray
    :param pred_test_or: Matriz de predictores de prueba originales.
    :type pred_test_or: numpy.ndarray
    :param target_entrenamiento: Vector de etiquetas de entrenamiento.
    :type target_entrenamiento: numpy.ndarray
    :param target_test: Vector de etiquetas de prueba.
    :type target_test: numpy.ndarray
    :param configuraciones: Diccionario de configuraciones de la red neuronal.
    :type configuraciones: dict
    :return: DataFrame con los resultados de la red neuronal y el diccionario de configuraciones actualizado.
    :rtype: pandas.DataFrame, dict
    """
    with ProcessPool() as pool:
        results = []
        for n_cnn, pruebas in configuraciones.items():
            cnn = cnn_preentrenadas[n_cnn](pred_entrenamiento_or.shape[1:])
            for prueba in pruebas.keys():
                prueba_mas_cnn = prueba + "_" + n_cnn
                predictores_train = imagenes_preprocesadas[prueba_mas_cnn](pred_entrenamiento_or, prueba_mas_cnn)
                predictores_train = funciones_datos.cnn_predict(predictores_train, "entrenamiento", cnn, n_cnn)
                predictores_test = imagenes_preprocesadas[prueba_mas_cnn](pred_test_or, prueba_mas_cnn)
                predictores_test = funciones_datos.cnn_predict(predictores_test, "validacion", cnn, n_cnn)

                for neurona in configuracion["parametros_top"]["neuronas"]:
                    for dropout in configuracion["parametros_top"]["dropouts"]:
                        for activacion in configuracion["parametros_top"]["activaciones"]:
                            for capa in configuracion["parametros_top"]["capas"]:
                                result = pool.apipe(
                                    entrenar_red,
                                    et_filtradas,
                                    target_entrenamiento,
                                    target_test,
                                    n_cnn,
                                    predictores_train,
                                    predictores_test,
                                    neurona,
                                    dropout,
                                    activacion,
                                    capa,
                                    configuracion["parametros_top"]["transfer_learning"]["max_epoch"],
                                    prueba,
                                )
                                results.append(result)

        df_tl_or = pd.DataFrame()
        for result in results:
            diccionario = result.get()
            df_tl_or, configuraciones = obtener_resultados(configuraciones, df_tl_or, diccionario)
    return df_tl_or, configuraciones


def multiproceso_clase_process(
    et_filtradas, pred_entrenamiento_or, pred_test_or, target_entrenamiento, target_test, configuraciones
):
    """Ejecuta el proceso de entrenamiento de redes neuronales en paralelo utilizando multiprocessing.

    :param et_filtradas: Lista de etiquetas filtradas.
    :type et_filtradas: list
    :param pred_entrenamiento_or: Predicciones de entrenamiento originales.
    :type pred_entrenamiento_or: numpy.ndarray
    :param pred_test_or: Predicciones de prueba originales.
    :type pred_test_or: numpy.ndarray
    :param target_entrenamiento: Etiquetas de entrenamiento.
    :type target_entrenamiento: numpy.ndarray
    :param target_test: Etiquetas de prueba.
    :type target_test: numpy.ndarray
    :param configuraciones: Diccionario de configuraciones de las redes neuronales.
    :type configuraciones: dict
    :return: Un dataframe con los resultados obtenidos en Transfer Learning y el diccionario de configuraciones actualizado.
    :rtype: pandas.DataFrame, dict
    """
    queue = Queue()
    procesos = []
    for n_cnn, pruebas in configuraciones.items():
        cnn = cnn_preentrenadas[n_cnn](pred_entrenamiento_or.shape[1:])
        for prueba in pruebas.keys():
            prueba_mas_cnn = prueba + "_" + n_cnn
            predictores_train = imagenes_preprocesadas[prueba_mas_cnn](pred_entrenamiento_or, prueba_mas_cnn)
            predictores_train = funciones_datos.cnn_predict(predictores_train, "entrenamiento", cnn, n_cnn)
            predictores_test = imagenes_preprocesadas[prueba_mas_cnn](pred_test_or, prueba_mas_cnn)
            predictores_test = funciones_datos.cnn_predict(predictores_test, "validacion", cnn, n_cnn)

            for neurona in configuracion["parametros_top"]["neuronas"]:
                for dropout in configuracion["parametros_top"]["dropouts"]:
                    for activacion in configuracion["parametros_top"]["activaciones"]:
                        for capa in configuracion["parametros_top"]["capas"]:
                            proceso = MiProceso(
                                target=entrenar_red,
                                args=(
                                    et_filtradas,
                                    target_entrenamiento,
                                    target_test,
                                    n_cnn,
                                    predictores_train,
                                    predictores_test,
                                    neurona,
                                    dropout,
                                    activacion,
                                    capa,
                                    configuracion["parametros_top"]["transfer_learning"]["max_epoch"],
                                    prueba,
                                ),
                                queue=queue,
                            )
                            proceso.start()
                            procesos.append(proceso)

    for proceso in procesos:
        proceso.join()

    # Creación de un dataframe con los resultados obtenidos en Transfer Learning
    df_tl_or = pd.DataFrame()

    while not queue.empty():
        diccionario = queue.get()
        configuraciones[diccionario["nombre_dicc_cnn"]][diccionario["prueba"]][diccionario["config"]] = diccionario[
            "modelo"
        ]
        df_tl_or, configuraciones = obtener_resultados(configuraciones, df_tl_or, diccionario)

    return df_tl_or, configuraciones


def secuencial(et_filtradas, pred_entrenamiento_or, pred_test_or, target_entrenamiento, target_test, configuraciones):
    """Ejecuta el proceso secuencial para entrenar y evaluar una red neuronal convolucional.

    :param et_filtradas: Lista de etiquetas filtradas.
    :type et_filtradas: list
    :param pred_entrenamiento_or: Predicciones de entrenamiento originales.
    :type pred_entrenamiento_or: numpy.ndarray
    :param pred_test_or: Predicciones de prueba originales.
    :type pred_test_or: numpy.ndarray
    :param target_entrenamiento: Etiquetas de entrenamiento.
    :type target_entrenamiento: numpy.ndarray
    :param target_test: Etiquetas de prueba.
    :type target_test: numpy.ndarray
    :param configuraciones: Diccionario de configuraciones de la red neuronal.
    :type configuraciones: dict
    :return: DataFrame con los resultados de entrenamiento y evaluación de la red neuronal, y el diccionario de configuraciones actualizado.
    :rtype: pandas.DataFrame, dict
    """
    df_tl_or = pd.DataFrame()

    for n_cnn, pruebas in configuraciones.items():
        cnn = cnn_preentrenadas[n_cnn](pred_entrenamiento_or.shape[1:])
        for prueba in pruebas:
            prueba_mas_cnn = prueba + "_" + n_cnn
            predictores_train = imagenes_preprocesadas[prueba_mas_cnn](pred_entrenamiento_or, prueba_mas_cnn)
            predictores_train = funciones_datos.cnn_predict(predictores_train, "entrenamiento", cnn, n_cnn)
            predictores_test = imagenes_preprocesadas[prueba_mas_cnn](pred_test_or, prueba_mas_cnn)
            predictores_test = funciones_datos.cnn_predict(predictores_test, "validacion", cnn, n_cnn)

            for neurona in configuracion["parametros_top"]["neuronas"]:
                for dropout in configuracion["parametros_top"]["dropouts"]:
                    for activacion in configuracion["parametros_top"]["activaciones"]:
                        for capa in configuracion["parametros_top"]["capas"]:
                            diccionario = entrenar_red(
                                et_filtradas,
                                target_entrenamiento,
                                target_test,
                                n_cnn,
                                predictores_train,
                                predictores_test,
                                neurona,
                                dropout,
                                activacion,
                                capa,
                                configuracion["parametros_top"]["transfer_learning"]["max_epoch"],
                                prueba,
                            )
                            df_tl_or, configuraciones = obtener_resultados(configuraciones, df_tl_or, diccionario)
    return df_tl_or, configuraciones


def obtener_resultados(configuraciones, df_tl_or, diccionario):
    """Obtiene los resultados de las configuraciones y los agrega al diccionario y al dataframe.

    :param configuraciones: Diccionario de configuraciones.
    :type configuraciones: dict
    :param df_tl_or: DataFrame original.
    :type df_tl_or: pandas.DataFrame
    :param diccionario: Diccionario con los resultados.
    :type diccionario: dict
    :return: DataFrame actualizado y diccionario actualizado.
    :rtype: pandas.DataFrame, dict
    """
    configuraciones[diccionario["nombre_dicc_cnn"]][diccionario["prueba"]][diccionario["config"]] = diccionario[
        "modelo"
    ]
    # Si el dataframe está vacío, se asigna el mini_df_tl, si no, se concatenan ambos dataframes
    if len(df_tl_or) == 0:
        df_tl_or = diccionario["dataframe"]
    else:
        df_tl_or = pd.concat([df_tl_or, diccionario["dataframe"]], join="outer", axis=0, ignore_index=True)
    return df_tl_or, configuraciones


def entrenar_red(
    et_filtradas: list,
    target_entrenamiento: list,
    target_test: list,
    nombre_dicc_cnn: str,
    pred_entrenamiento: list,
    pred_test: list,
    neurona: int,
    dropout: float,
    activacion: str,
    capa: int,
    max_epoch_tl: int,
    prueba: str,
):
    """Entrena una red neuronal utilizando el proceso de Transfer Learning.

    :param et_filtradas: Lista de características filtradas.
    :type et_filtradas: list
    :param target_entrenamiento: Lista de etiquetas de entrenamiento.
    :type target_entrenamiento: list
    :param target_test: Lista de etiquetas de prueba.
    :type target_test: list
    :param nombre_dicc_cnn: Nombre del diccionario de la CNN.
    :type nombre_dicc_cnn: str
    :param pred_entrenamiento: Lista de predicciones de entrenamiento.
    :type pred_entrenamiento: list
    :param pred_test: Lista de predicciones de prueba.
    :type pred_test: list
    :param neurona: Número de neuronas en la capa oculta.
    :type neurona: int
    :param dropout: Valor de dropout para regularización.
    :type dropout: float
    :param activacion: Función de activación para la capa oculta.
    :type activacion: str
    :param capa: Número de capas ocultas.
    :type capa: int
    :param max_epoch_tl: Número máximo de épocas para el proceso de Transfer Learning.
    :type max_epoch_tl: int
    :param prueba: Nombre de la prueba.
    :type prueba: str
    :return: Un diccionario que contiene el dataframe resultante, el nombre del diccionario de la CNN,
             el nombre de la prueba, la configuración y el modelo.
    :rtype: dict
    """
    # Creación de un dataframe vacío
    df_tl = pd.DataFrame()

    # Realiza el proceso de Transfer Learning
    df_tl, config, modelo = deep_learning.transfer_learning(
        neurona=neurona,
        dropout=dropout,
        activacion=activacion,
        capa=capa,
        max_epoch_tl=max_epoch_tl,
        et_filtradas=et_filtradas,
        pred_entrenamiento=pred_entrenamiento,
        pred_test=pred_test,
        target_entrenamiento=target_entrenamiento,
        target_test=target_test,
        df=df_tl,
        nombre_dicc_cnn=nombre_dicc_cnn,
        tasa_aprendizaje=configuracion["parametros_top"]["transfer_learning"]["learning_rate"],
        preprocesado=prueba,
    )

    return {
        "dataframe": df_tl,
        "nombre_dicc_cnn": nombre_dicc_cnn,
        "prueba": prueba,
        "config": config,
        "modelo": modelo,
    }


def seleccion_mejor_configuracion(df_mini_pp):
    """Selecciona la mejor configuración obtenida en Transfer Learning.

    :param df_mini_pp: DataFrame que contiene los resultados de las configuraciones.
    :type df_mini_pp: pandas.DataFrame
    :return: El nombre de la CNN, el número de neuronas, el valor de dropout, la función de activación y el número de capas de la mejor configuración.
    :rtype: tuple
    """
    # Selección de la mejor configuración obtenida en Transfer Learning
    nombre_cnn, nombre_top = df_mini_pp["Accuracy"].idxmax().rsplit(" | ")
    _, n_neuronas, n_dropout, n_activacion, n_capas = nombre_top.rsplit("_")

    # Conversión de los valores a los tipos correctos
    n_neuronas = int(n_neuronas)
    n_dropout = float(n_dropout)
    n_capas = int(n_capas)

    return nombre_cnn, n_neuronas, n_dropout, n_activacion, n_capas


def ejecuta_preprocesado_red_elegida(
    cnn_preentrenadas: dict,
    im_filtradas: list,
    configuraciones: dict,
    et_filtradas: list,
    imagenes_preprocesadas: dict,
    nombre_cnn: str,
    n_neuronas: int,
    n_dropout: float,
    n_activacion: str,
    n_capas: int,
):
    """Ejecuta el preprocesado de la red neuronal elegida.

    :param cnn_preentrenadas: Diccionario que contiene las redes neuronales preentrenadas.
    :type cnn_preentrenadas: dict
    :param im_filtradas: Lista de imágenes filtradas.
    :type im_filtradas: list
    :param configuraciones: Diccionario que contiene las configuraciones de las redes neuronales.
    :type configuraciones: dict
    :param et_filtradas: Lista de etiquetas filtradas.
    :type et_filtradas: list
    :param imagenes_preprocesadas: Diccionario que contiene las imágenes preprocesadas.
    :type imagenes_preprocesadas: dict
    :param nombre_cnn: Nombre de la red neuronal.
    :type nombre_cnn: str
    :param n_neuronas: Número de neuronas.
    :type n_neuronas: int
    :param n_dropout: Valor de dropout.
    :type n_dropout: float
    :param n_activacion: Tipo de función de activación.
    :type n_activacion: str
    :param n_capas: Número de capas.
    :type n_capas: int
    :return: Un dataframe con los resultados del preprocesado y las configuraciones actualizadas.
    :rtype: tuple
    """

    # Creación de un dataframe vacío
    df_prepro = pd.DataFrame()

    # Realiza el preprocesado de la red neuronal elegida
    cnn_elegida = cnn_preentrenadas[nombre_cnn](im_filtradas.shape[1:])
    pred_entrenamiento, pred_test, target_entrenamiento, target_test = train_test_split(
        imagenes_preprocesadas["im_preprocesadas_" + nombre_cnn],
        et_filtradas,
        test_size=0.2,
        shuffle=True,
        random_state=configuracion["parametros_top"]["seed"],
    )

    # Realiza una predicción utilizando una red neuronal convolucional
    pred_entrenamiento = funciones_datos.cnn_predict(
        pred_entrenamiento, "pred_entrenamiento_prep_" + nombre_cnn, cnn_elegida, nombre_cnn
    )
    pred_test = funciones_datos.cnn_predict(pred_test, "pred_test_prep_" + nombre_cnn, cnn_elegida, nombre_cnn)

    # Realiza el proceso de Transfer Learning
    df_prepro, config, modelo = deep_learning.transfer_learning(
        neurona=n_neuronas,
        dropout=n_dropout,
        activacion=n_activacion,
        capa=n_capas,
        max_epoch_tl=configuracion["parametros_top"]["transfer_learning"]["max_epoch"],
        et_filtradas=et_filtradas,
        pred_entrenamiento=pred_entrenamiento,
        pred_test=pred_test,
        target_entrenamiento=target_entrenamiento,
        target_test=target_test,
        df=df_prepro,
        nombre_dicc_cnn=nombre_cnn,
        tasa_aprendizaje=configuracion["parametros_top"]["transfer_learning"]["learning_rate"],
        preprocesado="preprocesada",
    )
    configuraciones[nombre_cnn]["prep"][config] = modelo
    return df_prepro, configuraciones


def selecciona_mejor_cnn(df_mini_or, df_mini_pp, df_prepro):
    """Selecciona la mejor CNN basada en los resultados de precisión comparando los resultados de precisión de tres dataframes: df_mini_or, df_mini_pp y df_prepro.
    Luego, selecciona la CNN con la mayor precisión y devuelve su nombre, el nombre del modelo top y una clave identificadora.

    :param df_mini_or: Dataframe que contiene los resultados de precisión de las CNN sin preprocesamiento de imágenes.
    :type df_mini_or: pandas.DataFrame
    :param df_mini_pp: Dataframe que contiene los resultados de precisión de las CNN con normalización de imágenes.
    :type df_mini_pp: pandas.DataFrame
    :param df_prepro: Dataframe que contiene los resultados de precisión de las CNN con imágenes preprocesadas.
    :type df_prepro: pandas.DataFrame
    :return: El nombre de la CNN seleccionada, el nombre del modelo top y una clave identificadora.
    :rtype: tuple
    """

    # Encontrar el índice del valor máximo en el dataframe df_mini_or
    indice_or = df_mini_or["Accuracy"].idxmax()
    nombre_cnn_mini_or, nombre_top_mini_or = indice_or.rsplit(" | ")
    # Encontrar el índice del valor máximo en el dataframe df_mini_pp
    indice_pp = df_mini_pp["Accuracy"].idxmax()
    nombre_cnn_mini_pp, nombre_top_mini_pp = indice_pp.rsplit(" | ")
    # Encontrar el índice del valor máximo en el dataframe df_prep
    indice_prep = df_prepro["Accuracy"].idxmax()
    nombre_cnn_prep, nombre_top_prep = indice_prep.rsplit(" | ")

    # Obtener los valores máximos de precisión
    max_accuracy_mini_or = float(df_mini_or.loc[indice_or, "Accuracy"].iloc[0])
    max_accuracy_mini_pp = float(df_mini_pp.loc[indice_pp, "Accuracy"].iloc[0])
    max_accuracy_prep = float(df_prepro.loc[indice_prep, "Accuracy"].iloc[0])

    # Comparar los valores máximos y seleccionar el mayor
    if max_accuracy_mini_or > max_accuracy_mini_pp and max_accuracy_mini_or > max_accuracy_prep:
        nombre_cnn = nombre_cnn_mini_or
        nombre_top = nombre_top_mini_or
        clave_cnn = "im_or"
    elif max_accuracy_mini_pp > max_accuracy_prep:
        nombre_cnn = nombre_cnn_mini_pp
        nombre_top = nombre_top_mini_pp
        clave_cnn = "im_norm"
    else:
        nombre_cnn = nombre_cnn_prep
        nombre_top = nombre_top_prep
        clave_cnn = "im_preprocesadas"

    return nombre_cnn, nombre_top, clave_cnn


def crear_cnn_ft(im_filtradas, nombre_cnn, configuraciones, nombre_top, clave_cnn):
    """Crea un modelo de red neuronal convolucional (CNN) para Fine-Tuning.

    :param im_filtradas: Las imágenes filtradas para entrenar el modelo.
    :type im_filtradas: numpy.ndarray
    :param nombre_cnn: El nombre de la CNN pre-entrenada a utilizar.
    :type nombre_cnn: str
    :param configuraciones: Las configuraciones de la CNN pre-entrenada.
    :type configuraciones: dict
    :param nombre_top: El nombre del clasificador a utilizar.
    :type nombre_top: str
    :param clave_cnn: La clave para acceder a las configuraciones del clasificador.
    :type clave_cnn: str
    :return: El modelo completo para Fine-Tuning.
    :rtype: tensorflow.keras.Model
    """
    cnn_funcion = cnn_preentrenadas[nombre_cnn](im_filtradas.shape[1:])
    # cnn_funcion.trainable = False # Establecimiento de la red como no entrenable para acoplar el clasificador seleccionado

    # Construcción del modelo completo para Fine-Tuning
    top = configuraciones[nombre_cnn][clave_cnn][nombre_top]

    modelo_completo = deep_learning.reconstruccion_mejor_modelo_df(cnn_funcion, top)

    modelo_completo.trainable = True  # Establecimiento de la red como entrenable para realizar el fine-tunning
    modelo_completo.compile(
        optimizer=optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return modelo_completo


def ejecuta_fine_tunning_mejor_cnn(im_filtradas, et_filtradas, nombre_cnn, configuraciones, nombre_top, clave_cnn):
    """Ejecuta el proceso de Fine-Tuning de una red neuronal convolucional (CNN) tomando como entrada imágenes filtradas y etiquetas filtradas,
    el nombre de la CNN a utilizar, las configuraciones necesarias, el nombre del modelo top, y la clave de la CNN.

    :param im_filtradas: Imágenes filtradas para el entrenamiento y prueba.
    :type im_filtradas: list
    :param et_filtradas: Etiquetas filtradas correspondientes a las imágenes.
    :type et_filtradas: list
    :param nombre_cnn: Nombre de la CNN a utilizar.
    :type nombre_cnn: str
    :param configuraciones: Configuraciones necesarias para el proceso.
    :type configuraciones: dict
    :param nombre_top: Nombre del modelo top.
    :type nombre_top: str
    :param clave_cnn: Clave de la CNN.
    :type clave_cnn: str
    """

    # Creación del modelo completo para Fine-Tuning con la mejor configuración obtenida
    modelo_completo = crear_cnn_ft(im_filtradas, nombre_cnn, configuraciones, nombre_top, clave_cnn)

    # Creación de la clave para acceder a las imágenes preprocesadas
    clave = clave_cnn + "_" + nombre_cnn

    # Preprocesado de las imágenes para Fine-Tuning y división de los datos de entrenamiento y prueba
    imagenes = imagenes_preprocesadas[clave](im_filtradas, "temporal")
    pred_entrenamiento, pred_test, target_entrenamiento, target_test = train_test_split(
        imagenes, et_filtradas, test_size=0.2, shuffle=True, random_state=configuracion["parametros_top"]["seed"]
    )

    # Entrenamiento y evaluación del modelo completo para Fine-Tuning
    target, predict_II = deep_learning.evaluar_modelo(
        configuracion["parametros_top"]["fine_tunning"]["max_epoch"],
        modelo_completo,
        pred_entrenamiento,
        pred_test,
        target_entrenamiento,
        target_test,
        nombre_cnn + "_finetunning",
        nombre_top,
        "finetunning",
    )

    # Creación del DataFrame final con los resultados del Fine-Tuning
    df_finetunning = deep_learning.crear_dataframe(predict_II, target, nombre_cnn + " | " + nombre_top, clave_cnn)

    # Guardado de los resultados finales del Fine-Tuning
    df_finetunning.to_excel("resultados_finetunning.xlsx")
