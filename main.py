
# Autores:  Marta María Álvarez Crespo y Juan Manuel Ramos Pérez
# Descripción:  Archivo de ejecución de un experimento de aprendizaje automático sobre imágenes utilizando modelos de redes neuronales convolucionales (CNN). 
# Última modificación: 23 / 03 / 2024


'''
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
'''



import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import deep_learning
import funciones_datos
import preprocesado
from tensorflow.keras import optimizers


if __name__ == "__main__":
    
    #  Creación de un diccionario con las CNN pre-entrenadas que se deseen cargar
    cnn_preentrenadas= {"Mobilenet": deep_learning.cargar_mn, "vgg16": deep_learning.cargar_vgg}
    
    #  Selección de los parámetros con los que se ejecutarán los experimentos
    neuronas= np.arange(5, 55, 5)
    dropouts= np.arange(0.0, 0.6, 0.1)
    capas= np.arange(1,3,1)
    activaciones= ["linear, sigmoid, relu, tanh"]
    
    max_epoch_tl = 30
    max_epoch_ft = 50

    
    seed = 26
    
    # Carga del dataset y organización de los datos para el entrenamiento
    im_filtradas, et_filtradas = funciones_datos.cargar_dataset()
    
    # Preprocesamiento de imágenes
    imagenes_preprocesadas = {
        "im_norm_mn": preprocesado.normalizacion_mn(im_filtradas, "im_norm_mn"),
        "im_norm_vgg": preprocesado.normalizacion_vgg(im_filtradas, "im_norm_vgg"),
        "im_preprocesadas_mn": preprocesado.preprocesado_mn(im_filtradas, 4, "im_prep_mn"),
        "im_preprocesadas_vgg": preprocesado.preprocesado_vgg(im_filtradas, 4, "im_prep_vgg")
    }
    
    # División de datos de entrenamiento y prueba en funcion de la red
    pred_entrenamiento_or, pred_test_or, target_entrenamiento, target_test= train_test_split(im_filtradas, et_filtradas, test_size= 0.2, shuffle= True, random_state= seed)
    pred_entrenamiento_mn, pred_test_mn, target_entrenamiento, target_test= train_test_split(imagenes_preprocesadas["im_norm_mn"], et_filtradas, test_size= 0.2, shuffle= True, random_state= seed)
    pred_entrenamiento_vgg, pred_test_vgg, target_entrenamiento, target_test= train_test_split(imagenes_preprocesadas["im_norm_vgg"], et_filtradas, test_size= 0.2, shuffle= True, random_state= seed)
    
    # Data Augmentation
    da_or= funciones_datos.data_augmentation(im_filtradas.shape[1:])
    da_mn= funciones_datos.data_augmentation(imagenes_preprocesadas["im_norm_mn"].shape[1:])
    da_vgg= funciones_datos.data_augmentation(imagenes_preprocesadas["im_norm_vgg"].shape[1:])
    pred_entrenamiento_or= funciones_datos.cnn_predict(pred_entrenamiento_or, "pred_entrenamiento_or", da_or)
    pred_entrenamiento_mn= funciones_datos.cnn_predict(pred_entrenamiento_mn, "pred_entrenamiento_mn", da_mn)
    pred_entrenamiento_vgg= funciones_datos.cnn_predict(pred_entrenamiento_vgg, "pred_entrenamiento_vgg", da_vgg)
    
    
    # Creación de los dataframe donde se almacenarán los resultados del estudio
    df_tl_or= pd.DataFrame()
    df_tl_pp= pd.DataFrame()
    
    # Experimentación de Transfer Learning con los parámetros establecidos y almacenamiento de los resultados en el dataframe creado
    print("\n\n\n==================== TRANSFER LEARNING ====================\n")
    
    for nombre_dicc_cnn, cnn_funcion in cnn_preentrenadas.items():
        df_tl_or, ccn_elegida, configuraciones, config = deep_learning.transfer_learning(neuronas, dropouts, activaciones, capas, max_epoch_tl, im_filtradas, et_filtradas, pred_entrenamiento_or, pred_test_or, target_entrenamiento, target_test, df_tl_or, nombre_dicc_cnn, cnn_funcion, 0.1, "original")   
        
        if nombre_dicc_cnn == "Mobilenet":
            pred_train = pred_entrenamiento_mn
            pred_val = pred_test_mn
        elif nombre_dicc_cnn == "vgg16":
            pred_train = pred_entrenamiento_vgg
            pred_val = pred_test_vgg
        df_tl_pp, ccn_elegida, configuraciones, config = deep_learning.transfer_learning(neuronas, dropouts, activaciones, capas, max_epoch_tl, pred_train, et_filtradas, pred_train, pred_val, target_entrenamiento, target_test, df_tl_pp, nombre_dicc_cnn, cnn_funcion, 0.1, "normalizado")   

    # Guardado de los datos originales en Excel
    df_mini_or= df_tl_or.set_index("Modelo de entrenamiento utilizado") 
    df_mini_or.to_excel("resultados_transfer_learning_original.xlsx") 
        
    # Guardado de los datos preprocesados en Excel
    df_mini_pp= df_tl_pp.set_index("Modelo de entrenamiento utilizado") 
    df_mini_pp.to_excel("resultados_transfer_learning_input_procesado.xlsx") 
    
    # Creación de DataFrame para almacenar resultados del Transfer Learning con las imágenes pre-procesadas
    df_prepro = pd.DataFrame() 

    # Selección de la mejor configuración obtenida en Transfer Learning
    nombre_cnn, nombre_top = df_mini_pp["Accuracy"].idxmax().rsplit(" | ")
    _, n_neuronas, n_dropout, n_activacion, n_capas = nombre_top.rsplit("_")
    
    n_neuronas = int(n_neuronas)
    n_dropout = float(n_dropout)
    n_activacion = n_activacion
    n_capas = int(n_capas)
    
    # Procesamiento según la CNN seleccionada
    if nombre_cnn == "vgg16":
        # Preprocesamiento específico para VGG16
        im_red = imagenes_preprocesadas["im_preprocesadas_vgg"]
        pred_entrenamiento_vgg, pred_test_vgg, target_entrenamiento, target_test= train_test_split(imagenes_preprocesadas["im_preprocesadas_vgg"], et_filtradas, test_size= 0.2, shuffle= True, random_state= seed)
        da_vgg= funciones_datos.data_augmentation(imagenes_preprocesadas["im_preprocesadas_vgg"].shape[1:])
        pred_entrenamiento_vgg= da_vgg.predict(pred_entrenamiento_vgg)
        df_prepro, ccn_elegida, configuraciones, config = deep_learning.transfer_learning(neuronas, dropouts, activaciones, capas, max_epoch_tl, im_red, et_filtradas, pred_entrenamiento_vgg, pred_test_vgg, target_entrenamiento, target_test, df_prepro, nombre_cnn, cnn_preentrenadas["vgg16"], 0.05, "preprocesada")
    
    elif nombre_cnn == "Mobilenet":
        # Preprocesamiento específico para MobileNet
        im_red = imagenes_preprocesadas["im_preprocesadas_mn"]
        pred_entrenamiento_mn, pred_test_mn, target_entrenamiento, target_test= train_test_split(imagenes_preprocesadas["im_preprocesadas_vmn"], et_filtradas, test_size= 0.2, shuffle= True, random_state= seed)
        da_mn = funciones_datos.data_augmentation(imagenes_preprocesadas["im_preprocesadas_mn"].shape[1:])
        pred_entrenamiento_mn= da_mn.predict(pred_entrenamiento_mn)
        df_prepro, ccn_elegida, configuraciones, config = deep_learning.transfer_learning(neuronas, dropouts, activaciones, capas, max_epoch_tl, im_red, et_filtradas, pred_entrenamiento_mn, pred_test_mn, target_entrenamiento, target_test, df_prepro, nombre_cnn, cnn_preentrenadas["Mobilenet"], 0.05, "preprocesada")

    # Guardado de los datos de preprocesamiento en Excel
    df_mini_prepro = df_prepro.set_index("Modelo de entrenamiento utilizado") 
    df_mini_prepro.to_excel("resultados_preprocesado.xlsx")
    
    
    # Fine Tunning con la red y parámetros que han ofrecido mejores resultados
    
    # Encontrar el índice del valor máximo en el dataframe df_mini_or
    indice_or= df_mini_or["Accuracy"].idxmax()
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
    elif max_accuracy_mini_pp > max_accuracy_prep:
        nombre_cnn = nombre_cnn_mini_pp
        nombre_top = nombre_top_mini_pp
    else:
        nombre_cnn = nombre_cnn_prep
        nombre_top = nombre_top_prep
        
    # Experimentación de Fine Tunning con los parámetros óptimos y almacenamiento de los resultados en el dataframe correspondiente
    print("\n\n\n==================== FINE TUNNING ====================\n")
    
    # Selección de la función de la red neuronal pre-entrenada
    cnn_funcion = cnn_preentrenadas[nombre_cnn](im_filtradas.shape[1:])
    # cnn_funcion.trainable = False # Establecimiento de la red como no entrenable para acoplar el clasificador seleccionado
    # cnn_funcion.summary()
    
    # Construcción del modelo completo para Fine-Tuning
    top = configuraciones[nombre_top]
    modelo_completo = deep_learning.reconstruccion_mejor_modelo_df(cnn_funcion, top)
    
    modelo_completo.trainable = True # Establecimiento de la red como entrenable para realizar el fine-tunning
    modelo_completo.summary()
    modelo_completo.compile(optimizer= optimizers.Adam(learning_rate=0.001), loss= "categorical_crossentropy", metrics=['accuracy'])
    
    # División de los datos en conjuntos de entrenamiento y prueba según la CNN seleccionada
    if nombre_cnn == "vgg16":
        # Preprocesamiento específico para VGG16
        im_red = imagenes_preprocesadas["im_preprocesadas_vgg"]
        pred_entrenamiento, pred_test, target_entrenamiento, target_test= train_test_split(imagenes_preprocesadas["im_preprocesadas_vgg"], et_filtradas, test_size= 0.2, shuffle= True, random_state= seed)
    
    elif nombre_cnn == "Mobilenet":
        # Preprocesamiento específico para MobileNet
        im_red = imagenes_preprocesadas["im_preprocesadas_mn"]
        pred_entrenamiento, pred_test, target_entrenamiento, target_test= train_test_split(imagenes_preprocesadas["im_preprocesadas_vmn"], et_filtradas, test_size= 0.2, shuffle= True, random_state= seed)
    
    # Entrenamiento y evaluación del modelo completo para Fine-Tuning
    target, predict_II = deep_learning.evaluar_modelo(max_epoch_ft, modelo_completo, pred_entrenamiento, pred_test, target_entrenamiento, target_test, nombre_cnn + "_finetunning", nombre_top, "finetunning")

    # Creación del DataFrame final con los resultados del Fine-Tuning
    df_finetunning = deep_learning.crear_dataframe(predict_II, target, ccn_elegida.name + " | " + config)
    
    # Guardado de los resultados finales del Fine-Tuning
    df_finetunning.to_excel("resultados_finetunning.xlsx")
