
# Autores:  Marta María Álvarez Crespo y Juan Manuel Ramos Pérez
# Descripción: Funciones y utilidades para entrenar, evaluar y visualizar CNN para tareas de clasificación utilizando transfer-learning
# Última modificación: 20 / 03 / 2024


from itertools import cycle
import numpy as np
from tensorflow import keras
from tensorflow.keras import callbacks, layers, optimizers
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import utilidades
import funciones_datos

def cargar_mn(input_shape):
    """Carga el modelo pre-entrenado MobileNet

    Args:
        input_shape (tuple): tamaño de las imágenes a procesar

    Returns:
        keras.src.models.functional.Functional: MobileNet para el tamaño deseado
    """
    modeloCNN = keras.applications.MobileNet(include_top= False, input_shape= input_shape)
    modeloCNN.summary()
    return modeloCNN


def cargar_vgg(input_shape):
    """Carga el modelo pre-entrenado VGG16

    Args:
        input_shape (tuple): tamaño de las imágenes a procesar

    Returns:
        keras.src.models.functional.Functional: VGG16 para el tamaño deseado
    """
    modeloCNN = keras.applications.VGG16(include_top= False, input_shape= input_shape)
    modeloCNN.summary()
    
    return modeloCNN


def crear_clasificador(entrada: tuple, neuronas : int, dropout : float, activation : str, numero_de_clases: int, numero_de_capas : int = 1):
    """_summary_

    Args:
        entrada (tuple): Dimensiones de la entrada de datos
        neuronas (int): Número de neuronas en la capa densa
        dropout (float): Tasa de dropout para reducir el sobreajuste del modelo
        activation (str): Función de activación para las capas densas
        numero_de_clases (int): Número de clases en la tarea de clasificación
        numero_de_capas (int): Número de capas ocultas a añadir al modelo

    Returns:
        keras.models.Model: Modelo de red neuronal completamente definido
    """
    model_input= keras.Input(shape= entrada)
    model_output= layers.Flatten()(model_input)
    
    for _ in range(numero_de_capas):
        model_output = layers.Dense(neuronas, activation=activation)(model_output)
        model_output = layers.Dropout(dropout)(model_output)
    model_output= layers.Dense(numero_de_clases, activation='softmax')(model_output)
    
    model= keras.models.Model(inputs= model_input, outputs= model_output)

    # Mostrar resumen del modelo
    # model.summary()
    
    return model


def entrenar_modelo(max_epochs, model, pred_entrenamiento, target_entrenamiento, tasa_aprendizaje):
    """Entrena el modelo de red neuronal especificado.

    Args:
        max_epochs (int): Número máximo de iteraciones para el entrenamiento
        model (keras.models.Model): Modelo de red neuronal a entrenar
        pred_entrenamiento (numpy.ndarray): Datos de entrenamiento (características)
        target_entrenamiento (numpy.ndarray): Etiquetas de entrenamiento (objetivos)
        tasa_aprendizaje (float): Tamaño de los ajustes realizados a los pesos durante el entrenamiento

    Returns:
        resumen: Historia del entrenamiento del modelo
    """
    n_iter_no_change = 5
    
    # Detecta el estancamiento de la red y la detiene para evitar el sobreentrenamiento
    earlystop_callback = callbacks.EarlyStopping(monitor='val_loss', verbose = 1, restore_best_weights = True, patience=n_iter_no_change) 
    
    # Compilar el modelo
    model.compile(
        optimizer= optimizers.Adam(learning_rate=tasa_aprendizaje),
        loss= "categorical_crossentropy",
        metrics=['accuracy']
    )
    
    # Entrenar el modelo
    resumen= model.fit(
        x= pred_entrenamiento, 
        y= target_entrenamiento,
        validation_split= 0.2, 
        batch_size= 32,
        callbacks= earlystop_callback,
        epochs= max_epochs
        )

    return resumen


def crear_configuraciones(entrada, neuronas, dropouts, activaciones, capa_salida, capas):
    """Crea diferentes configuraciones de clasificadores basadas en los parámetros proporcionados

    Args:
        entrada (tuple): Dimensiones de la entrada de datos
        neuronas (list): Lista de números de neuronas para las capas densas
        dropouts (list): Lista de tasas de dropout para reducir el sobreajuste del modelo
        activaciones (list): Lista de funciones de activación para las capas densas
        capa_salida (int): Número de clases en la tarea de clasificación

    Returns:
        dict: Diccionario que contiene las diferentes configuraciones de clasificadores. Las claves son cadenas que representan las configuraciones, 
              y los valores son modelos de red neuronal completamente definidos
    """
    dict = {}
    for neurona in neuronas:
        for dropout in dropouts:
            for activacion in activaciones:
                for capa in capas:
                    dict[f"TOP_{str(neurona)}_{str(dropout)}_{str(activacion)}_{str(capa)}"] = crear_clasificador(entrada, neurona, dropout, activacion, capa_salida, capa)
                
    return dict
    
    
def metricas_entrenamiento(history, nombre, config, preprocesado):
    """Grafica las métricas de entrenamiento del modelo a lo largo de los epochs

    Args:
        history (keras.callbacks.History): Objeto que contiene el historial de métricas y pérdidas durante el entrenamiento del modelo.
        nombre (str): Nombre del modelo
        config (str): Configuración específica del modelo

    Returns:
        None
    """
    # Gráfica de pérdida y exactitud
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Precisión de entrenamiento')
    plt.plot(epochs, val_acc, 'r', label='Precisión de validación')
    plt.title(f'Precisión de entrenamiento y validación para\n{config}_{nombre}')
    plt.legend()
    
    plt.figure()
    
    plt.plot(epochs, loss, 'b', label='Pérdida de entrenamiento')
    plt.plot(epochs, val_loss, 'r', label='Pérdida de validación')
    plt.title(f'Pérdida de entrenamiento y validación para\n{config}_{nombre}')
    plt.legend()

    utilidades.crear_carpeta("metricas_entrenamiento/" + nombre + "/" + preprocesado)
    plt.savefig("metricas_entrenamiento/" + nombre + "/" + preprocesado + "/perdida_exactitud_"+ nombre + "_" + config + ".png")
    plt.close()
    
    
def metricas_evaluacion(pred, target_test, nombre, config, preprocesado):
    """Genera métricas de evaluación del modelo y visualizaciones, como curvas ROC y matrices de confusión.

    Args:
        pred (numpy.ndarray): Predicciones del modelo.
        target_test (numpy.ndarray): Etiquetas verdaderas del conjunto de prueba.
        nombre (str): Nombre del modelo.
        config (str): Configuración específica del modelo.

    Returns:
        None
    """
    _, ax = plt.subplots(figsize=(6, 6))
    
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])

    for id_clase, color in zip(range(3), colors):
        RocCurveDisplay.from_predictions(
        target_test[:, id_clase],
        pred[:, id_clase],
        name=f"Curva ROC para la clase {id_clase}",
        color=color,
        ax=ax,
        plot_chance_level=(id_clase == 2),
    )
    # plt.show()
    
    _ = ax.set(
        xlabel="Tasa de Falsos Positivos (FP)",
        ylabel="Tasa de Verdaderos Positivos (TP)",
        title="Extensión de la Característica Operativa del Receptor\na Multiclase Uno-contra-Resto",
    )
    # plt.show()
    utilidades.crear_carpeta("metricas_evaluacion/" + nombre + "/" + preprocesado)
    plt.savefig("metricas_evaluacion/" + nombre + "/" + preprocesado + "/roc_curve_"+ nombre + "_" + config + ".png")
    plt.close()
    
    target_test= np.argmax(target_test, axis= 1)
    pred= np.argmax(pred, axis= 1)

    cm_display = ConfusionMatrixDisplay.from_predictions(target_test, pred, cmap='Blues')
    plt.title('Matriz de Confusión')
    # plt.show()
    plt.savefig("metricas_evaluacion/" + nombre + "/" + preprocesado + "/confusion_matrix_" + nombre + "_" + config + ".png")
    plt.close()
    
    
def evaluar_modelo(max_epoch, modelo, pred_entrenamiento, pred_test, target_entrenamiento, target_test, nombre_dicc_cnn, config, tasa_aprendizaje, tipo_ejecucion):
    """Evalúa el modelo utilizando los datos de entrenamiento y prueba, y genera métricas de evaluación

    Args:
        max_epoch (int): Número máximo de epochs de entrenamiento
        modelo (keras.Model): Modelo de red neuronal a evaluar
        pred_entrenamiento (numpy.ndarray): Datos de entrada de entrenamiento
        pred_test (numpy.ndarray): Datos de entrada de prueba
        target_entrenamiento (numpy.ndarray): Etiquetas de entrenamiento
        target_test (numpy.ndarray): Etiquetas de prueba
        nombre_dicc_cnn (str): Nombre del modelo de red neuronal
        config (str): Configuración específica del modelo
        tasa_aprendizaje (float): Tamaño de los ajustes realizados a los pesos durante el entrenamiento 
        tipo_ejecucion (str): Nombre de la ejecución a realizar

    Returns:
        tuple: Un par de arrays numpy que representan las predicciones del modelo y las etiquetas de prueba
    """
    history = entrenar_modelo(max_epoch, modelo, pred_entrenamiento, target_entrenamiento, tasa_aprendizaje)
    metricas_entrenamiento(history, nombre_dicc_cnn, config, tipo_ejecucion)
    resultados_predict= modelo.predict(pred_test)
    metricas_evaluacion(resultados_predict, target_test, nombre_dicc_cnn, config, tipo_ejecucion)
    
    return resultados_predict, target_test
  
       
def crear_dataframe(pred, target_test, nombre):
    """Crea un DataFrame a partir de las predicciones y etiquetas de prueba, y calcula métricas de evaluación

    Args:
        pred (numpy.ndarray): Predicciones del modelo
        target_test (numpy.ndarray): Etiquetas de prueba
        nombre (str): Nombre del modelo de entrenamiento utilizado

    Returns:
        pandas.DataFrame: DataFrame que contiene métricas de evaluación para cada clase.
    """
    target_test= np.argmax(target_test, axis= 1)
    pred= np.argmax(pred, axis= 1)
     
    reporte = classification_report(target_test, pred, digits= 4, output_dict=True, zero_division= np.nan)
    
    # Crear una lista de diccionarios que contienen las métricas de interés
    data = []
    for clase, metrics in reporte.items():
        if clase in ['0', '1', '2']: 
            data.append({
                'Modelo de entrenamiento utilizado': nombre,
                'Clase a predecir': clase,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1-score'],
                'Accuracy': reporte["accuracy"]
            })

    # Crear el DataFrame
    df = pd.DataFrame(data)

    # Ordenar el DataFrame por la columna 'Clase a predecir'
    df = df.sort_values(by='Clase a predecir', ignore_index=True)
    df['Clase a predecir']= df['Clase a predecir'].map({'0': "Galaxia A", '1': "Galaxia B", "2": "Galaxia C"})

    return df


def transfer_learning(neuronas, dropouts, activaciones, capas, max_epoch_tl, im_filtradas, et_filtradas, pred_entrenamiento, pred_test, target_entrenamiento, target_test, df, nombre_dicc_cnn, cnn, tasa_aprendizaje, preprocesado):
    """ Realiza transfer learning utilizando modelos de redes neuronales preentrenadas según los parámetros preestablecidos

    Args:
        neuronas (list): Lista de enteros que representa el número de neuronas en cada capa oculta del modelo
        dropouts (list): Lista de floats que representa la tasa de dropout en cada capa del modelo
        activaciones (list): Lista de cadenas que representa las funciones de activación en cada capa del modelo
        capas (int): Número de capas ocultas que tendrá el modelo
        max_epoch_tl (int): Número máximo de epoch de entrenamiento
        im_filtradas (array): Matriz de datos de entrada filtrados
        et_filtradas (array): Matriz de etiquetas filtradas
        pred_entrenamiento (array): Datos de predicción de entrenamiento
        pred_test (array): Datos de predicción de prueba
        target_entrenamiento (array): Etiquetas de entrenamiento
        target_test (array): Etiquetas de prueba
        df (DataFrame): DataFrame donde se almacenan los resultados
        nombre_dicc_cnn (str): Nombre de la CNN a ejecutar 
        cnn (keras.src.models.functional.Functional): el modelo preentrenado de Keras
        tasa_aprendizaje (float): Tamaño de los ajustes realizados a los pesos durante el entrenamiento
        preprocesado (str):
        
        

    Returns:
    - df (DataFrame): DataFrame actualizado con los nuevos resultados
    - ccn_elegida: CNN elegida para el entrenamiento
    - configuraciones (dict): Diccionario que contiene las configuraciones de modelos probadas
    - config (str): Configuración actual del modelo
    """
    # Creación de la CNN elegida
    ccn_elegida = cnn(im_filtradas.shape[1:])
    
    # Predicción de los datos de entrenamiento y prueba con la CNN
    pred_entrenamiento_cnn = funciones_datos.cnn_predict(pred_entrenamiento, "entrenamiento", ccn_elegida)
    pred_test_cnn = funciones_datos.cnn_predict(pred_test, "validacion", ccn_elegida)
    
    # Creación de las configuraciones para experimentar
    configuraciones = crear_configuraciones(pred_entrenamiento_cnn.shape[1:], neuronas, dropouts, activaciones, et_filtradas[0].shape[0], capas)
    
    # Experimentación con diferentes configuraciones
    for config, modelo in configuraciones.items():
        print(f"\n\n\n============ Se está probando {nombre_dicc_cnn} con la config {config} ==============\n") 
        target, predict_II = evaluar_modelo(max_epoch_tl, modelo, pred_entrenamiento_cnn, pred_test_cnn, target_entrenamiento, target_test, nombre_dicc_cnn, config, tasa_aprendizaje, preprocesado)
        # Concatenación de resultados al DataFrame
        df= pd.concat([df, crear_dataframe(predict_II, target, ccn_elegida.name + " | " + config)], join= "outer", axis= 0, ignore_index= True)
        
    return df,ccn_elegida,configuraciones,config
