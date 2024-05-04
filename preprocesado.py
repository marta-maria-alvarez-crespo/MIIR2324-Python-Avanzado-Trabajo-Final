# Autores:  Marta María Álvarez Crespo y Juan Manuel Ramos Pérez
# Descripción:  Archivo con las funciones principales de tratamiento de las imágenes para realizar la experimentación
# Última modificación: 20 / 03 / 2024

from skimage.filters import gaussian
import numpy as np
from tensorflow import keras
from skimage.color import rgb2gray
from skimage import filters
import os

def escala_grises(imagenes_filtradas):
    """Convierte las imágenes a escala de grises

    Args:
        imagenes_filtradas (numpy.ndarray): Matriz de imágenes RGB

    Returns:
        numpy.ndarray: Matriz de imágenes en escala de grises
    """
    imagenes_filtradas = rgb2gray(imagenes_filtradas)  
    return imagenes_filtradas


def disminucion_ruido(imagenes_filtradas, sigma):
    """Aplica un filtro Gaussiano para disminuir el ruido en las imágenes

    Args:
        imagenes_filtradas (numpy.ndarray): Matriz de imágenes.
        sigma (float): Valor del parámetro sigma para el filtro Gaussiano

    Returns:
        numpy.ndarray: Matriz de imágenes con el ruido reducido
    """
    imagenes_filtradas = gaussian(imagenes_filtradas, sigma=sigma)
    return imagenes_filtradas


def realzar_bordes(imagenes_filtradas):
    """Realza los bordes en las imágenes utilizando el operador Sobel

    Args:
        imagenes_filtradas (numpy.ndarray): Matriz de imágenes

    Returns:
        numpy.ndarray: Matriz de imágenes con bordes realzados
    """
    imagenes_filtradas = filters.sobel(imagenes_filtradas)
    return imagenes_filtradas


def normalizacion_mn(imagenes_filtradas, nombre):
    """Normaliza las imágenes para el modelo MobileNet

    Args:
        imagenes_filtradas (numpy.ndarray): Matriz de imágenes
        nombre (str): Nombre para guardar el archivo de normalización

    Returns:
        numpy.ndarray: Matriz de imágenes normalizadas para MobileNet
    """
    file_path = f"{nombre}.npy"
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        imagenes_filtradas = keras.applications.mobilenet.preprocess_input(imagenes_filtradas)
        np.save(file_path, imagenes_filtradas)
        return imagenes_filtradas


def normalizacion_vgg(imagenes_filtradas, nombre):
    """Normaliza las imágenes para el modelo VGG16

    Args:
        imagenes_filtradas (numpy.ndarray): Matriz de imágenes
        nombre (str): Nombre para guardar el archivo de normalización

    Returns:
        numpy.ndarray: Matriz de imágenes normalizadas para VGG16
    """
    file_path = f"{nombre}.npy"
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        imagenes_filtradas = keras.applications.vgg16.preprocess_input(imagenes_filtradas)
        np.save(file_path, imagenes_filtradas)
        return imagenes_filtradas
    
    
def preprocesado(imagenes_filtradas, sigma):
    """Realiza el preprocesamiento de las imágenes

    Args:
        imagenes_filtradas (numpy.ndarray): Matriz de imágenes
        sigma (float): Valor del parámetro sigma para el filtro Gaussiano
        
    Returns:
        numpy.ndarray: Matriz de imágenes preprocesadas
    """
    # imagenes_filtradas = escala_grises(imagenes_filtradas)
    imagenes_filtradas = disminucion_ruido(imagenes_filtradas, sigma)
    imagenes_filtradas = realzar_bordes(imagenes_filtradas)
    return imagenes_filtradas


def preprocesado_mn(imagenes_filtradas, sigma, nombre):
    """Realiza el preprocesamiento específico para MobileNet

    Args:
        imagenes_filtradas (numpy.ndarray): Matriz de imágenes
        sigma (float): Valor del parámetro sigma para el filtro Gaussiano
        nombre (str): Nombre para guardar el archivo de normalización

    Returns:
        numpy.ndarray: Matriz de imágenes preprocesadas para MobileNet
    """
    file_path = f"{nombre}.npy"
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        imagenes_filtradas = preprocesado(imagenes_filtradas, sigma)
        imagenes_filtradas = keras.applications.mobilenet.preprocess_input(imagenes_filtradas)
        np.save(file_path, imagenes_filtradas)
        return imagenes_filtradas


def preprocesado_vgg(imagenes_filtradas, sigma, nombre):
    """Realiza el preprocesamiento específico para VGG16

    Args:
        imagenes_filtradas (numpy.ndarray): Matriz de imágenes
        sigma (float): Valor del parámetro sigma para el filtro Gaussiano
        nombre (str): Nombre para guardar el archivo de normalización

    Returns:
        numpy.ndarray: Matriz de imágenes preprocesadas para VGG16
    """
    file_path = f"{nombre}.npy"
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        imagenes_filtradas = preprocesado(imagenes_filtradas, sigma)
        imagenes_filtradas = keras.applications.vgg16.preprocess_input(imagenes_filtradas)
        np.save(file_path, imagenes_filtradas)
        return imagenes_filtradas
