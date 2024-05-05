# Autores:  Marta María Álvarez Crespo y Juan Manuel Ramos Pérez
# Descripción:  Funciones y utilidades para entrenar, evaluar y visualizar CNN para tareas de clasificación utilizando fine-tunning
# Última modificación: 23 / 03 / 2024

from tensorflow import keras
import transfer_learning
 

def cargar_modelo_y_configuraciones(dicc, configuraciones, df_mini, im_filtradas):
    """Carga un modelo y sus configuraciones óptimas basadas en un DataFrame dado

    Args:
        dicc (dict): Diccionario que contiene los modelos de red neuronal disponibles
        configuraciones (dict): Diccionario que contiene las diferentes configuraciones de modelos
        df_mini (pandas.DataFrame): DataFrame que contiene los resultados de la evaluación de los modelos
        im_filtradas (numpy.ndarray): Datos de entrada (imágenes) utilizados para la evaluación del modelo

    Returns:
        tuple: Una tupla que contiene el modelo de red neuronal cargado y su configuración óptima
    """
    nombre_cnn, nombre_top = df_mini["Accuracy"].idxmax().rsplit(" | ")
    cnn = dicc[nombre_cnn](im_filtradas.shape[1:])
    cnn.trainable = False
    top = configuraciones[nombre_top]
    return cnn, top


def construir_modelo_completo(cnn, top):
    """Construye un modelo de red neuronal completo a partir de una CNN y una capa densa superior

    Args:
        cnn (keras.models.Model): Modelo de red neuronal convolucional (CNN)
        top (keras.models.Model): Capa densa superior para la clasificación

    Returns:
        keras.models.Model: Modelo de red neuronal completo
    """
    
    full_output = top(cnn.output)
    full_model = keras.models.Model(inputs=cnn.input, outputs=full_output)
    return full_model


def entrenar_y_evaluar_modelo(max_epoch, full_model, pred_entrenamiento, pred_test, target_entrenamiento, target_test, nombre_cnn, nombre_top, tasa_aprendizaje, preprocesado):
    """Entrena y evalúa un modelo de red neuronal completo

    Args:
        max_epoch (int): Número máximo de epochs de entrenamiento
        full_model (keras.models.Model): Modelo de red neuronal completo a entrenar y evaluar
        pred_entrenamiento (numpy.ndarray): Datos de entrada de entrenamiento
        pred_test (numpy.ndarray): Datos de entrada de prueba
        target_entrenamiento (numpy.ndarray): Etiquetas de entrenamiento
        target_test (numpy.ndarray): Etiquetas de prueba
        nombre_cnn (str): Nombre del modelo de red neuronal convolucional (CNN) utilizado
        nombre_top (str): Nombre de la capa densa superior utilizada
        tasa_aprendizaje (float): 
        preprocesado ()

    Returns:
        tuple: Un par de arrays numpy que representan las etiquetas de prueba y las predicciones del modelo
    """
    target, predict_II = transfer_learning.evaluar_modelo(max_epoch, full_model, pred_entrenamiento, pred_test, target_entrenamiento, target_test, nombre_cnn + "_finetunning", nombre_top, tasa_aprendizaje, preprocesado)
    return target, predict_II


def guardar_resultados(df_final):
    """Guarda los resultados finales en archivos Excel

    Args:
        df_final (pandas.DataFrame): DataFrame que contiene los resultados finales a guardar

    Returns:
        None
    """
    df_final.to_excel("resultados_finetunning.xlsx")


def fine_tunning(dicc, max_epoch, im_filtradas, pred_entrenamiento, pred_test, target_entrenamiento, target_test, ccn_elegida, configuraciones, config, df_mini):
    """Carga el modelo de red neuronal óptimo y sus configuraciones, realiza el ajuste fino del modelo utilizando los datos de entrenamiento y evalúa su desempeño 
    utilizando los datos de prueba

    Args:
        dicc (dict): Diccionario que contiene los modelos de red neuronal disponibles
        max_epoch (int): Número máximo de epochs de entrenamiento
        im_filtradas (numpy.ndarray): Imágenes utilizadas para la evaluación del modelo
        pred_entrenamiento (numpy.ndarray): Datos de entrada de entrenamiento
        pred_test (numpy.ndarray): Datos de entrada de prueba
        target_entrenamiento (numpy.ndarray): Etiquetas de entrenamiento
        target_test (numpy.ndarray): Etiquetas de prueba
        ccn_elegida (objeto): Modelo de red neuronal convolucional (CNN) seleccionado
        configuraciones (dict): Diccionario que contiene las diferentes configuraciones de modelos
        config (str): Configuración específica del modelo
        df_mini (pandas.DataFrame): DataFrame que contiene los resultados de la evaluación de los modelos

    Returns:
        None
    """
    
    nombre_cnn, nombre_top = df_mini["Accuracy"].idxmax().rsplit(" | ")

    cnn = dicc[nombre_cnn](im_filtradas.shape[1:])
    cnn.trainable = False
    top = configuraciones[nombre_top]

    full_model = construir_modelo_completo(cnn, top)
    full_model.summary()

    target, predict_II = entrenar_y_evaluar_modelo(max_epoch, full_model, pred_entrenamiento, pred_test, target_entrenamiento, target_test, nombre_cnn, nombre_top)
    df_final = transfer_learning.crear_dataframe(predict_II, target, ccn_elegida.name + " | " + config)

    guardar_resultados(df_final)
    