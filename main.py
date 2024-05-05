# Autora:  Marta María Álvarez Crespo
# Descripción:  Archivo de ejecución de un experimento de aprendizaje automático sobre imágenes utilizando modelos de redes neuronales convolucionales (CNN).
# Última modificación: 05 / 05 / 2024


"""
Práctica Python:

- Paralelizar praáctica de VA-I
-  df_prepro, ccn_elegida, configuraciones, config = deep_learning.transfer_learning(neuronas, dropouts, activaciones, capas, max_epoch_tl, im_red, et_filtradas, pred_entrenamiento_vgg, pred_test_vgg, target_entrenamiento, target_test, df_prepro, nombre_cnn, cnn_preentrenadas["vgg16"], 0.05, "preprocesada") 

En estas funciones, tengo que pasarle un solo parámetro en vez de una lista. 

La idea es hacer un pool de varios procesos donde se prueban de X en X configuraciones para cada red. 

- Algo dijo de sacar esto de aqui, pero no se (funcion transfer learning)

    # Experimentación con diferentes configuraciones
    for config, modelo in configuraciones.items():
        print(f"\n\n\n============ Se está probando {nombre_dicc_cnn} con la config {config} ==============\n") 
        target, predict_II = evaluar_modelo(max_epoch_tl, modelo, pred_entrenamiento_cnn, pred_test_cnn, target_entrenamiento, target_test, nombre_dicc_cnn, config, tasa_aprendizaje, preprocesado)
        # Concatenación de resultados al DataFrame
        df= pd.concat([df, crear_dataframe(predict_II, target, ccn_elegida.name + " | " + config)], join= "outer", axis= 0, ignore_index= True)


- Todo esto para el lunes (quiere que saque gráficas de rendimiento etc y me lo explicará cuando acabe esto)
"""

from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers  # type: ignore

import numpy as np
import pandas as pd
import deep_learning
import funciones_datos
import preprocesado
from mi_hilo import MiHilo


# ToDo: Fichero config
#  Creación de un diccionario con las CNN pre-entrenadas que se deseen cargar
cnn_preentrenadas = {"mn": deep_learning.cargar_mn, "vgg": deep_learning.cargar_vgg}

#  Selección de los parámetros con los que se ejecutarán los experimentos
neuronas = [5]
dropouts = [0]
capas = [2]
activaciones = ["linear"]

max_epoch_tl = 1
max_epoch_ft = 1

seed = 26

# #  Creación de un diccionario con las CNN pre-entrenadas que se deseen cargar
# cnn_preentrenadas = {"mn": deep_learning.cargar_mn, "vgg": deep_learning.cargar_vgg}

# #  Selección de los parámetros con los que se ejecutarán los experimentos
# neuronas = np.arange(5, 55, 5)
# dropouts = np.arange(0.0, 0.6, 0.1)
# capas = np.arange(1, 3, 1)
# activaciones = ["linear, sigmoid, relu, tanh"]

# max_epoch_tl = 30
# max_epoch_ft = 50

# seed = 26


def dividir_conjunto_datos(seed, im_filtradas, et_filtradas, imagenes_preprocesadas):
    """Divide el conjuntos de datos (imágenes filtradas y preprocesadas) en conjuntos de entrenamiento y prueba.

    :param seed: asignación de un valor para aplicar una randomización de los datos reproducible
    :type seed: int
    :param im_filtradas: conjunto de imágenes a dividir
    :type im_filtradas: _type_
    :param et_filtradas: conjunto de etiquetas
    :type et_filtradas: _type_
    :param imagenes_preprocesadas: _description_
    :type imagenes_preprocesadas: _type_
    :return: _description_
    :rtype: _type_
    """
    pred_entrenamiento_or, pred_test_or, target_entrenamiento, target_test = train_test_split(
        im_filtradas, et_filtradas, test_size=0.2, shuffle=True, random_state=seed
    )
    pred_entrenamiento_mn, pred_test_mn, target_entrenamiento, target_test = train_test_split(
        imagenes_preprocesadas["im_norm_mn"], et_filtradas, test_size=0.2, shuffle=True, random_state=seed
    )
    pred_entrenamiento_vgg, pred_test_vgg, target_entrenamiento, target_test = train_test_split(
        imagenes_preprocesadas["im_norm_vgg"], et_filtradas, test_size=0.2, shuffle=True, random_state=seed
    )

    return (
        pred_entrenamiento_or,
        pred_test_or,
        target_entrenamiento,
        target_test,
        pred_entrenamiento_mn,
        pred_test_mn,
        pred_entrenamiento_vgg,
        pred_test_vgg,
    )


def division_preparacion_datos_entrada(im_filtradas, imagenes_preprocesadas, seed, et_filtradas):
    # División de datos de entrenamiento y prueba en funcion de la red
    (
        pred_entrenamiento_or,
        pred_test_or,
        target_entrenamiento,
        target_test,
        pred_entrenamiento_mn,
        pred_test_mn,
        pred_entrenamiento_vgg,
        pred_test_vgg,
    ) = dividir_conjunto_datos(seed, im_filtradas, et_filtradas, imagenes_preprocesadas)

    da_or = funciones_datos.data_augmentation(im_filtradas.shape[1:])
    da_mn = funciones_datos.data_augmentation(imagenes_preprocesadas["im_norm_mn"].shape[1:])
    da_vgg = funciones_datos.data_augmentation(imagenes_preprocesadas["im_norm_vgg"].shape[1:])

    pred_entrenamiento_or = funciones_datos.cnn_predict(pred_entrenamiento_or, "pred_entrenamiento_", da_or, "or")
    pred_entrenamiento_mn = funciones_datos.cnn_predict(pred_entrenamiento_mn, "pred_entrenamiento_", da_mn, "mn")
    pred_entrenamiento_vgg = funciones_datos.cnn_predict(pred_entrenamiento_vgg, "pred_entrenamiento_", da_vgg, "vgg")

    return (
        pred_entrenamiento_or,
        pred_test_or,
        target_entrenamiento,
        target_test,
        pred_entrenamiento_mn,
        pred_test_mn,
        pred_entrenamiento_vgg,
        pred_test_vgg,
    )


def ejecuta_experimentos_transfer_learning(
    im_filtradas,
    et_filtradas,
    pred_entrenamiento_or,
    pred_test_or,
    target_entrenamiento,
    target_test,
    pred_entrenamiento_mn,
    pred_test_mn,
    pred_entrenamiento_vgg,
    pred_test_vgg,
):
    df_tl_or = pd.DataFrame()
    df_tl_pp = pd.DataFrame()

    # Experimentación de Transfer Learning con los parámetros establecidos y almacenamiento de los resultados en el dataframe creado
    print("\n\n\n==================== TRANSFER LEARNING ====================\n")
    dicc_base = {"or": {}, "pp": {}, "prep": {}}
    configuraciones = {"mn": dicc_base, "vgg": dicc_base}
    for nombre_dicc_cnn, cnn_funcion in cnn_preentrenadas.items():
        # Creación de la CNN elegida
        cnn_elegida = cnn_funcion(im_filtradas.shape[1:])

        # Predicción de los datos de entrenamiento y prueba con la CNN
        pred_entrenamiento_cnn = funciones_datos.cnn_predict(
            pred_entrenamiento_or, "entrenamiento", cnn_elegida, nombre_dicc_cnn
        )
        pred_test_cnn = funciones_datos.cnn_predict(pred_test_or, "validacion", cnn_elegida, nombre_dicc_cnn)

        if nombre_dicc_cnn == "mn":
            pred_train = pred_entrenamiento_mn
            pred_val = pred_test_mn

        elif nombre_dicc_cnn == "vgg":
            pred_train = pred_entrenamiento_vgg
            pred_val = pred_test_vgg

        pred_entrenamiento_cnn_pp = funciones_datos.cnn_predict(
            pred_train, "entrenamiento", cnn_elegida, nombre_dicc_cnn
        )
        pred_test_cnn_pp = funciones_datos.cnn_predict(pred_val, "validacion", cnn_elegida, nombre_dicc_cnn)

        hilos = []
        for neurona in neuronas:
            for dropout in dropouts:
                for activacion in activaciones:
                    for capa in capas:
                        hilo = MiHilo(
                            target=hilo_tl,
                            args=(
                                et_filtradas,
                                target_entrenamiento,
                                target_test,
                                configuraciones,
                                nombre_dicc_cnn,
                                pred_entrenamiento_cnn,
                                pred_test_cnn,
                                pred_entrenamiento_cnn_pp,
                                pred_test_cnn_pp,
                                neurona,
                                dropout,
                                activacion,
                                capa,
                            ),
                        )
                        hilo.start()
                        hilos.append(hilo)

        for hilo in hilos:
            hilo.join()

        for hilo in hilos:
            mini_df_tl_or, mini_df_tl_pp, configuracion = hilo.result
            if not (len(df_tl_or)):
                df_tl_or = mini_df_tl_or
            else:
                df_tl_or = pd.concat([df_tl_or, mini_df_tl_or], axis=0, ignore_index=True)
            if not (len(df_tl_pp)):
                df_tl_pp = mini_df_tl_pp
            else:
                df_tl_pp = pd.concat([df_tl_pp, mini_df_tl_pp], axis=0, ignore_index=True)
            configuraciones.update(configuracion)

    # Guardado de los datos originales en Excel
    df_mini_or = df_tl_or.set_index("Modelo de entrenamiento utilizado")
    df_mini_or.to_excel("resultados_transfer_learning_original.xlsx")

    # Guardado de los datos preprocesados en Excel
    df_mini_pp = df_tl_pp.set_index("Modelo de entrenamiento utilizado")
    df_mini_pp.to_excel("resultados_transfer_learning_input_procesado.xlsx")

    return configuraciones, df_mini_or, df_mini_pp


def hilo_tl(
    et_filtradas,
    target_entrenamiento,
    target_test,
    configuraciones,
    nombre_dicc_cnn,
    pred_entrenamiento_cnn,
    pred_test_cnn,
    pred_entrenamiento_cnn_pp,
    pred_test_cnn_pp,
    neurona,
    dropout,
    activacion,
    capa,
):
    df_tl_or = pd.DataFrame()
    df_tl_or, config, modelo = deep_learning.transfer_learning(
        neurona=neurona,
        dropout=dropout,
        activacion=activacion,
        capa=capa,
        max_epoch_tl=max_epoch_tl,
        et_filtradas=et_filtradas,
        pred_entrenamiento=pred_entrenamiento_cnn,
        pred_test=pred_test_cnn,
        target_entrenamiento=target_entrenamiento,
        target_test=target_test,
        df=df_tl_or,
        nombre_dicc_cnn=nombre_dicc_cnn,
        tasa_aprendizaje=0.1,
        preprocesado="original",
    )
    configuraciones[nombre_dicc_cnn]["or"][config] = modelo

    df_tl_pp = pd.DataFrame()
    df_tl_pp, config, modelo = deep_learning.transfer_learning(
        neurona=neurona,
        dropout=dropout,
        activacion=activacion,
        capa=capa,
        max_epoch_tl=max_epoch_tl,
        et_filtradas=et_filtradas,
        pred_entrenamiento=pred_entrenamiento_cnn_pp,
        pred_test=pred_test_cnn_pp,
        target_entrenamiento=target_entrenamiento,
        target_test=target_test,
        df=df_tl_pp,
        nombre_dicc_cnn=nombre_dicc_cnn,
        tasa_aprendizaje=0.1,
        preprocesado="normalizado",
    )
    configuraciones[nombre_dicc_cnn]["pp"][config] = modelo

    return df_tl_or, df_tl_pp, configuraciones


def seleccion_mejor_configuracion(df_mini_pp):
    # Selección de la mejor configuración obtenida en Transfer Learning
    nombre_cnn, nombre_top = df_mini_pp["Accuracy"].idxmax().rsplit(" | ")
    _, n_neuronas, n_dropout, n_activacion, n_capas = nombre_top.rsplit("_")

    n_neuronas = int(n_neuronas)
    n_dropout = float(n_dropout)
    n_capas = int(n_capas)
    return nombre_cnn, n_neuronas, n_dropout, n_activacion, n_capas


def ejecuta_preprocesado_red_elegida(
    configuraciones, et_filtradas, imagenes_preprocesadas, nombre_cnn, n_neuronas, n_dropout, n_activacion, n_capas
):
    df_prepro = pd.DataFrame()

    cnn_elegida = cnn_preentrenadas[nombre_cnn](im_filtradas.shape[1:])
    pred_entrenamiento, pred_test, target_entrenamiento, target_test = train_test_split(
        imagenes_preprocesadas["im_preprocesadas_" + nombre_cnn],
        et_filtradas,
        test_size=0.2,
        shuffle=True,
        random_state=seed,
    )

    pred_entrenamiento = funciones_datos.cnn_predict(
        pred_entrenamiento, "pred_entrenamiento_prep_" + nombre_cnn, cnn_elegida, nombre_cnn
    )
    pred_test = funciones_datos.cnn_predict(pred_test, "pred_test_prep_" + nombre_cnn, cnn_elegida, nombre_cnn)

    df_prepro, config, modelo = deep_learning.transfer_learning(
        neurona=n_neuronas,
        dropout=n_dropout,
        activacion=n_activacion,
        capa=n_capas,
        max_epoch_tl=max_epoch_tl,
        et_filtradas=et_filtradas,
        pred_entrenamiento=pred_entrenamiento,
        pred_test=pred_test,
        target_entrenamiento=target_entrenamiento,
        target_test=target_test,
        df=df_prepro,
        nombre_dicc_cnn=nombre_cnn,
        tasa_aprendizaje=0.05,
        preprocesado="preprocesada",
    )
    configuraciones[nombre_cnn]["prep"][config] = modelo
    return df_prepro, configuraciones


def preprocesar(et_filtradas, imagenes_preprocesadas, nombre_cnn):
    im_red = imagenes_preprocesadas["im_preprocesadas_" + nombre_cnn]
    pred_entrenamiento, pred_test, target_entrenamiento, target_test = train_test_split(
        imagenes_preprocesadas["im_preprocesadas_" + nombre_cnn],
        et_filtradas,
        test_size=0.2,
        shuffle=True,
        random_state=seed,
    )

    return im_red, pred_entrenamiento, pred_test, target_entrenamiento, target_test


def selecciona_mejor_cnn(df_mini_or, df_mini_pp, df_prepro):

    df_mini_prepro = df_prepro.set_index("Modelo de entrenamiento utilizado")
    df_mini_prepro.to_excel("resultados_preprocesado.xlsx")

    # Fine Tunning con la red y parámetros que han ofrecido mejores resultados

    # Encontrar el índice del valor máximo en el dataframe df_mini_or
    indice_or = df_mini_or["Accuracy"].idxmax()
    nombre_cnn_mini_or, nombre_top_mini_or = indice_or.rsplit(" | ")
    # Encontrar el índice del valor máximo en el dataframe df_mini_pp
    indice_pp = df_mini_pp["Accuracy"].idxmax()
    nombre_cnn_mini_pp, nombre_top_mini_pp = indice_pp.rsplit(" | ")
    # Encontrar el índice del valor máximo en el dataframe df_prep
    indice_prep = df_mini_prepro["Accuracy"].idxmax()
    nombre_cnn_prep, nombre_top_prep = indice_prep.rsplit(" | ")

    # Obtener los valores máximos de precisión
    max_accuracy_mini_or = float(df_mini_or.loc[indice_or, "Accuracy"].iloc[0])
    max_accuracy_mini_pp = float(df_mini_pp.loc[indice_pp, "Accuracy"].iloc[0])
    max_accuracy_prep = float(df_mini_prepro.loc[indice_prep, "Accuracy"].iloc[0])

    # Comparar los valores máximos y seleccionar el mayor
    if max_accuracy_mini_or > max_accuracy_mini_pp and max_accuracy_mini_or > max_accuracy_prep:
        nombre_cnn = nombre_cnn_mini_or
        nombre_top = nombre_top_mini_or
        clave_cnn = "or"
    elif max_accuracy_mini_pp > max_accuracy_prep:
        nombre_cnn = nombre_cnn_mini_pp
        nombre_top = nombre_top_mini_pp
        clave_cnn = "pp"
    else:
        nombre_cnn = nombre_cnn_prep
        nombre_top = nombre_top_prep
        clave_cnn = "prep"

    return nombre_cnn, nombre_top, clave_cnn


def crearcnnft(im_filtradas, nombre_cnn, configuraciones, nombre_top, clave_cnn):
    cnn_funcion = cnn_preentrenadas[nombre_cnn](im_filtradas.shape[1:])
    # cnn_funcion.trainable = False # Establecimiento de la red como no entrenable para acoplar el clasificador seleccionado
    # cnn_funcion.summary()

    # Construcción del modelo completo para Fine-Tuning
    top = configuraciones[nombre_cnn][clave_cnn][nombre_top]
    modelo_completo = deep_learning.reconstruccion_mejor_modelo_df(cnn_funcion, top)

    modelo_completo.trainable = True  # Establecimiento de la red como entrenable para realizar el fine-tunning
    modelo_completo.summary()
    modelo_completo.compile(
        optimizer=optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return modelo_completo


def ejecuta_fine_tunning_mejor_cnn(
    im_filtradas, et_filtradas, imagenes_preprocesadas, nombre_cnn, configuraciones, nombre_top, clave_cnn
):
    print("\n\n\n==================== FINE TUNNING ====================\n")

    # Selección de la función de la red neuronal pre-entrenada
    modelo_completo = crearcnnft(im_filtradas, nombre_cnn, configuraciones, nombre_top, clave_cnn)

    # División de los datos en conjuntos de entrenamiento y prueba según la CNN seleccionada
    _, pred_entrenamiento, pred_test, target_entrenamiento, target_test = preprocesar(
        et_filtradas, imagenes_preprocesadas, nombre_cnn
    )

    # Entrenamiento y evaluación del modelo completo para Fine-Tuning
    target, predict_II = deep_learning.evaluar_modelo(
        max_epoch_ft,
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
    df_finetunning = deep_learning.crear_dataframe(predict_II, target, nombre_cnn + " | " + nombre_top)

    # Guardado de los resultados finales del Fine-Tuning
    df_finetunning.to_excel("resultados_finetunning.xlsx")


if __name__ == "__main__":
    # Carga del dataset
    im_filtradas, et_filtradas = funciones_datos.cargar_dataset()

    im_filtradas = np.concatenate((im_filtradas[:50], im_filtradas[-50:]))
    et_filtradas = np.concatenate((et_filtradas[:50], et_filtradas[-50:]))

    # Preprocesamiento de las imágenes
    imagenes_preprocesadas = {
        "im_norm_mn": preprocesado.normalizacion_mn(im_filtradas, "im_norm_mn"),
        "im_norm_vgg": preprocesado.normalizacion_vgg(im_filtradas, "im_norm_vgg"),
        "im_preprocesadas_mn": preprocesado.preprocesado_mn(im_filtradas, 4, "im_prep_mn"),
        "im_preprocesadas_vgg": preprocesado.preprocesado_vgg(im_filtradas, 4, "im_prep_vgg"),
    }


    # División de los datos de entrada en conjuntos de entrenamiento y validación
    (
        pred_entrenamiento_or,
        pred_test_or,
        target_entrenamiento,
        target_test,
        pred_entrenamiento_mn,
        pred_test_mn,
        pred_entrenamiento_vgg,
        pred_test_vgg,
    ) = division_preparacion_datos_entrada(im_filtradas, imagenes_preprocesadas, seed, et_filtradas)

    # Ejecución de los experimentos de Transfer Learning configurados
    configuraciones, df_mini_or, df_mini_pp = ejecuta_experimentos_transfer_learning(
        im_filtradas,
        et_filtradas,
        pred_entrenamiento_or,
        pred_test_or,
        target_entrenamiento,
        target_test,
        pred_entrenamiento_mn,
        pred_test_mn,
        pred_entrenamiento_vgg,
        pred_test_vgg,
    )

    # Selección de la mejor configuración obtenida en Transfer Learning
    nombre_cnn, n_neuronas, n_dropout, n_activacion, n_capas = seleccion_mejor_configuracion(df_mini_pp)

    # Procesamiento del dataset según la CNN seleccionada
    df_prepro, configuraciones = ejecuta_preprocesado_red_elegida(
        configuraciones, et_filtradas, imagenes_preprocesadas, nombre_cnn, n_neuronas, n_dropout, n_activacion, n_capas
    )

    # Compara los experimentos y devuelve la combinación de la mejor red y configuración según los resultados obtenidos
    nombre_cnn, nombre_top, clave_cnn = selecciona_mejor_cnn(df_mini_or, df_mini_pp, df_prepro)

    # Experimentación de Fine Tunning con los parámetros óptimos y almacenamiento de los resultados en un dataframe
    ejecuta_fine_tunning_mejor_cnn(
        im_filtradas, et_filtradas, imagenes_preprocesadas, nombre_cnn, configuraciones, nombre_top, clave_cnn
    )
