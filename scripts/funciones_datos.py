# Autores:  Marta María Álvarez Crespo
# Descripción: Carga y tratamiento de los datos seleccionados para una red neuronal
# Última modificación: 20 / 05 / 2024

import numpy as np
import h5py
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder
import os

import json

configuracion = json.load(open("./configuracion.json", "r", encoding="UTF-8"))


def cargar_dataset():
    """
    Carga y filtra el dataset según las clases elegidas.

    :return: Una tupla con las imágenes y las etiquetas filtradas.
    :rtype: tuple
    """
    with h5py.File(configuracion["cargar_dataset"]["dataset"], "r") as F:
        imagenes = np.array(F["images"])
        etiquetas = np.array(F["ans"])

    # Clases que se van a utilizar
    clases_utilizadas = configuracion["cargar_dataset"]["clases"]

    # Índices de las muestras a las que pertenecen estas clases
    indices_clases_utilizadas = np.isin(etiquetas, clases_utilizadas)

    # Selección de las filas
    imagenes_filtradas = imagenes[indices_clases_utilizadas]
    etiquetas_filtradas = etiquetas[indices_clases_utilizadas]

    # # Codificación One Hot de las clases
    # OHE = OneHotEncoder()
    # etiquetas_filtradas = OHE.fit_transform(etiquetas_filtradas)
    # # Codificación One Hot de las clases
    # OHE = OneHotEncoder()
    # etiquetas_filtradas = OHE.fit_transform(etiquetas_filtradas)

    target = []
    for etiqueta in etiquetas_filtradas:
        if etiqueta == 1:
            target.append([1, 0, 0])
        elif etiqueta == 9:
            target.append([0, 1, 0])
        else:
            target.append([0, 0, 1])

    etiquetas_filtradas = np.array(target)

    return imagenes_filtradas, etiquetas_filtradas


def data_augmentation(input_shape):
    """Realiza la ampliación de datos mediante transformaciones aleatorias.

    :param input_shape: Forma de entrada de los datos.
    :type input_shape: tuple
    :return: Modelo con las transformaciones aleatorias aplicadas.
    :rtype: keras.models.Model
    """
    model_input = keras.Input(shape=input_shape)
    model_output = layers.RandomFlip("horizontal_and_vertical")(model_input)
    model_output = layers.RandomRotation(0.2)(model_output)

    model = keras.models.Model(inputs=model_input, outputs=model_output)
    return model


def cnn_predict(img, tipo, modeloCNN, nombreCNN):
    """Realiza una predicción utilizando una red neuronal convolucional (CNN).

    :param img: La imagen de entrada para realizar la predicción.
    :type img: numpy.ndarray
    :param tipo: El tipo de imagen.
    :type tipo: str
    :param modeloCNN: El modelo de la red neuronal convolucional.
    :type modeloCNN: keras.models.Model
    :param nombreCNN: El nombre de la red neuronal convolucional.
    :type nombreCNN: str
    :return: La predicción realizada por la CNN.
    :rtype: numpy.ndarray
    """
    file_path = f"{tipo + nombreCNN}.npy"
    if os.path.exists(os.path.join("predictores", file_path)):
        return np.load(os.path.join("predictores", file_path))
    else:
        predict = modeloCNN.predict(img)
        predict = np.array(predict)
        if not os.path.exists("predictores"):
            os.makedirs("predictores")
        np.save(os.path.join("predictores", file_path), predict)
        return predict
