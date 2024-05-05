
# Autores:  Marta María Álvarez Crespo y Juan Manuel Ramos Pérez
# Descripción: Carga y tratamiento de los datos seleccionados para una red neuronal
# Última modificación: 20 / 03 / 2024 

import numpy as np
import h5py
from tensorflow import keras
from tensorflow.keras import layers
import os

def cargar_dataset():
    """Carga el conjunto de datos Galaxy10 y filtra las clases específicas de interés

    Returns:
        tuple: Una tupla que contiene las imágenes filtradas y las etiquetas correspondientes
    """
    with h5py.File('dataset/Galaxy10_DECals.h5', 'r') as F:
        imagenes = np.array(F['images'])
        etiquetas = np.array(F['ans'])
        
    # unique, counts = np.unique(etiquetas, return_counts=True)
    # print(dict(zip(unique, counts)))
    
    # Clases que vamos a utilizar (1, 6 y 9)
    clases_utilizadas = [1, 6, 9]

    # Índices de las muestras a las que pertenecen estas clases
    indices_clases_utilizadas = np.isin(etiquetas, clases_utilizadas)

    # Selección de las filas
    imagenes_filtradas = imagenes[indices_clases_utilizadas]
    etiquetas_filtradas = etiquetas[indices_clases_utilizadas]

    # unique, counts = np.unique(etiquetas_filtradas, return_counts=True)
    # print(dict(zip(unique, counts)))
    
    target= []
    for etiqueta in etiquetas_filtradas:
        if etiqueta==1:
            target.append([1,0,0])
        elif etiqueta==9:
            target.append([0,1,0])
        else:
            target.append([0,0,1])
            
    etiquetas_filtradas= np.array(target)
    
    return imagenes_filtradas, etiquetas_filtradas


def data_augmentation(input_shape):
    """Aplica técnicas de aumento de datos a las imágenes

    Args:
        input_shape (tuple): Dimensiones de la entrada de datos

    Returns:
        keras.models.Model: Modelo de aumento de datos
    """
    model_input= keras.Input(shape = input_shape)
    model_output= layers.RandomFlip("horizontal_and_vertical")(model_input)
    model_output= layers.RandomRotation(0.2)(model_output)
    
    model= keras.models.Model(inputs= model_input, outputs= model_output)
    return model


def cnn_predict(img, tipo, modeloCNN, nombreCNN):
    """Realiza predicciones utilizando un modelo de red neuronal convolucional pre-entrenado

    Args:
        img (numpy.ndarray): Imagen a predecir
        tipo (str): Tipo de modelo de red neuronal convolucional
        modeloCNN (keras.models.Model): Modelo de red neuronal convolucional pre-entrenado

    Returns:
        numpy.ndarray: Predicciones del modelo.
    """
    file_path = f"{tipo + nombreCNN}.npy"
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        predict = modeloCNN.predict(img)
        predict = np.array(predict)
        np.save(file_path, predict)
        return predict