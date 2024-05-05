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

import numpy as np
import deep_learning
import funciones_datos
import preprocesado
import fnc


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
    ) = fnc.division_preparacion_datos_entrada(im_filtradas, imagenes_preprocesadas, seed, et_filtradas)

    # Ejecución de los experimentos de Transfer Learning configurados
    configuraciones, df_mini_or, df_mini_pp = fnc.ejecuta_experimentos_transfer_learning(
        im_filtradas=im_filtradas,
        et_filtradas=et_filtradas,
        cnn_preentrenadas=cnn_preentrenadas,
        neuronas=neuronas,
        dropouts=dropouts,
        activaciones=activaciones,
        capas=capas,
        max_epoch_tl=max_epoch_tl,
        pred_entrenamiento_or=pred_entrenamiento_or,
        pred_test_or=pred_test_or,
        target_entrenamiento=target_entrenamiento,
        target_test=target_test,
        pred_entrenamiento_mn=pred_entrenamiento_mn,
        pred_test_mn=pred_test_mn,
        pred_entrenamiento_vgg=pred_entrenamiento_vgg,
        pred_test_vgg=pred_test_vgg,
    )

    # Selección de la mejor configuración obtenida en Transfer Learning
    nombre_cnn, n_neuronas, n_dropout, n_activacion, n_capas = fnc.seleccion_mejor_configuracion(df_mini_pp)

    # Procesamiento del dataset según la CNN seleccionada
    df_prepro, configuraciones = fnc.ejecuta_preprocesado_red_elegida(
        cnn_preentrenadas=cnn_preentrenadas,
        im_filtradas=im_filtradas,
        configuraciones=configuraciones,
        et_filtradas=et_filtradas,
        imagenes_preprocesadas=imagenes_preprocesadas,
        nombre_cnn=nombre_cnn,
        n_neuronas=n_neuronas,
        n_dropout=n_dropout,
        n_activacion=n_activacion,
        n_capas=n_capas,
        max_epoch_tl=max_epoch_tl,
        seed=seed,
    )

    # Compara los experimentos y devuelve la combinación de la mejor red y configuración según los resultados obtenidos
    nombre_cnn, nombre_top, clave_cnn = fnc.selecciona_mejor_cnn(df_mini_or, df_mini_pp, df_prepro)

    # Experimentación de Fine Tunning con los parámetros óptimos y almacenamiento de los resultados en un dataframe
    fnc.ejecuta_fine_tunning_mejor_cnn(
        im_filtradas, et_filtradas, imagenes_preprocesadas, nombre_cnn, configuraciones, nombre_top, clave_cnn, max_epoch_ft, seed
    )
