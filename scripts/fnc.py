from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
import pandas as pd
import os
import deep_learning
import funciones_datos
from mi_hilo import MiHilo, MiProceso
from multiprocess import Queue

from preprocesado import imagenes_preprocesadas
import json

configuracion = json.load(open("./configuracion.json", "r", encoding= 'UTF-8'))

#  Creación de un diccionario con las CNN pre-entrenadas que se deseen cargar
cnn_preentrenadas = {"mn": deep_learning.cargar_mn, "vgg": deep_learning.cargar_vgg}

def division_preparacion_datos_entrada(im_filtradas, et_filtradas):
    """Divide los datos de entrada en conjuntos de entrenamiento y prueba y aplica data augmentation a las imágenes de entrenamiento.

    :param im_filtradas: Imágenes filtradas.
    :type im_filtradas: numpy.ndarray
    :param et_filtradas: Etiquetas filtradas.
    :type et_filtradas: numpy.ndarray
    :return: Tupla que contiene los conjuntos de entrenamiento y prueba de las imágenes filtradas y las etiquetas filtradas.
    :rtype: tuple
    """    
    # División de datos de entrenamiento y prueba
    (
        pred_entrenamiento_or,
        pred_test_or,
        target_entrenamiento,
        target_test
    ) = train_test_split(im_filtradas, et_filtradas, test_size=0.2, shuffle=True, random_state= configuracion["parametros_top"]["seed"])

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


def ejecuta_experimentos_transfer_learning(
    et_filtradas,
    pred_entrenamiento_or,
    pred_test_or,
    target_entrenamiento,
    target_test,
):
    """Realiza experimentos de Transfer Learning utilizando los parámetros establecidos y almacena los resultados en un dataframe.

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
    :return: Un diccionario con las configuraciones y tres dataframes con los resultados.
    :rtype: tuple
    """

    dicc_base = {"im_or": {}, "im_norm": {}, "im_preprocesadas": {}}
    configuraciones = {"mn": {"im_or": {}, "im_norm": {}, "im_preprocesadas": {}}, "vgg": {"im_or": {}, "im_norm": {}, "im_preprocesadas": {}}}

    multis = []
    queue = Queue()

    # Experimentación de Transfer Learning con los parámetros establecidos y almacenamiento de los resultados en el dataframe creado

    for n_cnn, pruebas in configuraciones.items():
        cnn = cnn_preentrenadas[n_cnn](pred_entrenamiento_or.shape[1:])
        for prueba in pruebas.keys():
            prueba_mas_cnn = prueba + '_' + n_cnn
            predictores_train = imagenes_preprocesadas[prueba_mas_cnn](pred_entrenamiento_or, prueba_mas_cnn)
            predictores_train = funciones_datos.cnn_predict(
                predictores_train, "entrenamiento", cnn, n_cnn
            )
            predictores_test = imagenes_preprocesadas[prueba_mas_cnn](pred_test_or, prueba_mas_cnn)
            predictores_test = funciones_datos.cnn_predict(
                predictores_test, "validacion", cnn, n_cnn
            )
            
            for neurona in configuracion["parametros_top"]["neuronas"]:
                for dropout in configuracion["parametros_top"]["dropouts"]:
                    for activacion in configuracion["parametros_top"]["activaciones"]:
                        for capa in configuracion["parametros_top"]["capas"]:
                            multi = MiHilo( # MiProceso para PROCESS 
                                target=multi_tl,
                                args=(
                                    et_filtradas,
                                    target_entrenamiento,
                                    target_test,
                                    # configuraciones,
                                    n_cnn,
                                    predictores_train,
                                    predictores_test,
                                    neurona,
                                    dropout,
                                    activacion,
                                    capa,
                                    configuracion["parametros_top"]["transfer_learning"]["max_epoch"],
                                    prueba
                                ),
                                #queue= queue # DESCOMENTA PARA MULTIPROCESO PROCESS
                            )
                            multi.start()
                            multis.append(multi)

    for multi in multis:
        multi.join()
    
    # Creación de un dataframe con los resultados obtenidos en Transfer Learning
    df_tl_or = pd.DataFrame()
    
    # DESCOMENTA PARA MULTIHILO
    for multi in multis:
        diccionario = multi.get_result()
        configuraciones[diccionario["nombre_dicc_cnn"]][diccionario["prueba"]][diccionario["config"]] = diccionario["modelo"]
        # Si el dataframe está vacío, se asigna el mini_df_tl, si no, se concatenan ambos dataframes
        if not len(df_tl_or): df_tl_or = diccionario["dataframe"]
        else: df_tl_or = pd.concat([df_tl_or, diccionario["dataframe"]], join= "outer", axis=0, ignore_index=True)      
    
    print("HOLA")
    # while not queue.empty():
    #     diccionario = queue.get()
    #     configuraciones[diccionario["nombre_dicc_cnn"]][diccionario["prueba"]][diccionario["config"]] = diccionario["modelo"]
    #     # Si el dataframe está vacío, se asigna el mini_df_tl, si no, se concatenan ambos dataframes
    #     if not len(df_tl_or): df_tl_or = diccionario["dataframe"]
    #     else: df_tl_or = pd.concat([df_tl_or, diccionario["dataframe"]], join= "outer", axis=0, ignore_index=True)      
    #     print("HOLA")
    # print(df_tl_or)
    
    # Guardado de los datos originales en Excel
    df_mini = df_tl_or.set_index("Modelo de entrenamiento utilizado")

    df_or = df_mini[df_mini["Tipo de imagen"] == "im_or"]
    df_norm = df_mini[df_mini["Tipo de imagen"] == "im_norm"]
    df_preprocesado = df_mini[df_mini["Tipo de imagen"] == "im_preprocesadas"]

    if not os.path.exists("Resultados_Dataframes"):
        os.makedirs("Resultados_Dataframes")

    df_or.to_excel(os.path.join("Resultados_Dataframes", "resultados_transfer_learning_original.xlsx"))
    df_norm.to_excel(os.path.join("Resultados_Dataframes", "resultados_transfer_learning_normalizado.xlsx"))
    df_preprocesado.to_excel(os.path.join("Resultados_Dataframes", "resultados_transfer_learning_preprocesado.xlsx"))

    return configuraciones, df_or, df_norm, df_preprocesado



def multi_tl(
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
    prueba: str
):
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

    return {'dataframe':df_tl, 'nombre_dicc_cnn': nombre_dicc_cnn, "prueba": prueba, "config": config, "modelo": modelo} 

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


def ejecuta_fine_tunning_mejor_cnn(
    im_filtradas, et_filtradas, nombre_cnn, configuraciones, nombre_top, clave_cnn
):
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
    clave = clave_cnn + '_' + nombre_cnn

    # Preprocesado de las imágenes para Fine-Tuning y división de los datos de entrenamiento y prueba
    imagenes = imagenes_preprocesadas[clave](im_filtradas, "temporal")
    pred_entrenamiento, pred_test, target_entrenamiento, target_test = train_test_split(imagenes, et_filtradas, test_size=0.2, shuffle=True, random_state= configuracion["parametros_top"]["seed"])

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