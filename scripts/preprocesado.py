# Autora:  Marta María Álvarez Crespo 
# Descripción:  Archivo con las funciones principales de tratamiento de las imágenes para realizar la experimentación
# Última modificación: 25 / 03 / 2024
# GitHub: www.github.com/marta-maria-alvarez-crespo/MIIR2324-Python-Avanzado-Trabajo-Final


import os
import json
import numpy as np
from skimage.filters import gaussian
from tensorflow import keras
from skimage.color import rgb2gray
from skimage import filters


configuracion = json.load(open("scripts/configuracion.json", "r", encoding="UTF-8"))


def escala_grises(imagenes_filtradas):
    """Convierte las imágenes en escala de grises.

    :param imagenes_filtradas: Lista de imágenes a convertir.
    :type imagenes_filtradas: list
    :return: Lista de imágenes en escala de grises.
    :rtype: list
    """
    imagenes_filtradas = rgb2gray(imagenes_filtradas)
    return imagenes_filtradas


def disminucion_ruido(imagenes_filtradas):
    """Disminuye el ruido en las imágenes utilizando un filtro Gaussiano.

    :param imagenes_filtradas: Lista de imágenes a convertir.
    :type imagenes_filtradas: list
    :return: Lista de imágenes en escala de grises.
    :rtype: list
    """
    imagenes_filtradas = gaussian(imagenes_filtradas, sigma=configuracion["preprocesamiento"]["sigma"])
    return imagenes_filtradas


def realzar_bordes(imagenes_filtradas):
    """Realza los bordes en las imágenes utilizando el operador Sobel

    :param imagenes_filtradas: Lista de imágenes a convertir.
    :type imagenes_filtradas: list
    :return: Lista de imágenes en escala de grises.
    :rtype: list
    """
    imagenes_filtradas = filters.sobel(imagenes_filtradas)
    return imagenes_filtradas


def originales(imagenes_filtradas, *args):
    """Función que devuelve las imágenes originales sin filtrar.

    :param imagenes_filtradas: Lista de imágenes filtradas.
    :type imagenes_filtradas: list
    :param args: Argumentos adicionales (opcional).
    :return: Lista de imágenes filtradas originales.
    :rtype: list
    """
    return imagenes_filtradas


def normalizacion_mn(imagenes_filtradas, nombre):
    """Normaliza las imágenes filtradas según el input requerido por Mobilenet y guarda los resultados en un archivo numpy.

    :param imagenes_filtradas: Las imágenes filtradas que se van a normalizar.
    :type imagenes_filtradas: numpy.ndarray
    :param nombre: El nombre del archivo numpy donde se guardarán los resultados.
    :type nombre: str
    :return: Las imágenes filtradas normalizadas.
    :rtype: numpy.ndarray
    """
    file_path = f"./predictores/{nombre}.npy"
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        imagenes_filtradas = keras.applications.mobilenet.preprocess_input(imagenes_filtradas)
        if nombre != "temporal":
            np.save(file_path, imagenes_filtradas)
        return imagenes_filtradas


def normalizacion_vgg(imagenes_filtradas, nombre):
    """Normaliza las imágenes filtradas según el input requerido por VGG16 y guarda los resultados en un archivo numpy.

    :param imagenes_filtradas: Las imágenes filtradas que se van a normalizar.
    :type imagenes_filtradas: numpy.ndarray
    :param nombre: El nombre del archivo numpy donde se guardarán los resultados.
    :type nombre: str
    :return: Las imágenes filtradas normalizadas.
    :rtype: numpy.ndarray
    """
    file_path = f"./predictores/{nombre}.npy"
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        imagenes_filtradas = keras.applications.vgg16.preprocess_input(imagenes_filtradas)
        np.save(file_path, imagenes_filtradas)
        return imagenes_filtradas


def preprocesado(imagenes_filtradas):
    """Realiza el preprocesamiento elegido de las imágenes filtradas.

    :param imagenes_filtradas: Lista de imágenes filtradas.
    :type imagenes_filtradas: list
    :return: Lista de imágenes preprocesadas.
    :rtype: list
    """
    imagenes_filtradas = disminucion_ruido(imagenes_filtradas)
    imagenes_filtradas = realzar_bordes(imagenes_filtradas)

    return imagenes_filtradas


def preprocesado_mn(imagenes_filtradas, nombre):
    """Realiza el preprocesamiento de las imágenes filtradas con la normalización requerida por Mobilenet.

    :param imagenes_filtradas: Lista de imágenes filtradas.
    :type imagenes_filtradas: list
    :return: Lista de imágenes preprocesadas.
    :rtype: list
    """
    file_path = f"./predictores/{nombre}.npy"
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        imagenes_filtradas = preprocesado(imagenes_filtradas)
        imagenes_filtradas = keras.applications.mobilenet.preprocess_input(imagenes_filtradas)
        np.save(file_path, imagenes_filtradas)
        return imagenes_filtradas


def preprocesado_vgg(imagenes_filtradas, nombre):
    """Realiza el preprocesamiento de las imágenes filtradas con la normalización requerida por VGG16.

    :param imagenes_filtradas: Lista de imágenes filtradas.
    :type imagenes_filtradas: list
    :return: Lista de imágenes preprocesadas.
    :rtype: list
    """
    file_path = f"./predictores/{nombre}.npy"
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        imagenes_filtradas = preprocesado(imagenes_filtradas)
        imagenes_filtradas = keras.applications.vgg16.preprocess_input(imagenes_filtradas)
        np.save(file_path, imagenes_filtradas)
        return imagenes_filtradas


# Preprocesamiento de las imágenes
imagenes_preprocesadas = {
    "im_or_mn": originales,
    "im_or_vgg": originales,
    "im_norm_mn": normalizacion_mn,
    "im_norm_vgg": normalizacion_vgg,
    "im_preprocesadas_mn": preprocesado_mn,
    "im_preprocesadas_vgg": preprocesado_vgg,
}
