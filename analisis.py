# Autores:  Marta María Álvarez Crespo y Juan Manuel Ramos Pérez
# Descripción:  Archivo de funciones y utilidades para realizar un estudio de las imágenes de un dataset
# Última modificación: 23 / 03 / 2024

import matplotlib.pyplot as plt
import random
from skimage import exposure
from skimage.color import rgb2gray
from skimage import filters
from funciones_datos import cargar_dataset
import utilidades


def imagen_original(imagen, etiqueta, valor):
    """
    Genera una imagen original con su respectivo título y la guarda como un archivo.

    Args:
        imagen (array): La imagen original
        etiqueta (str): La etiqueta de clasificación de la imagen
        valor (int): El número de la imagen

    Returns:
        None
    """
    generar_imagen(imagen, etiqueta, valor)
    guardar_figuras(valor, etiqueta, "imagen_original")
    

def visualizar_canales_color(imagen, etiqueta, valor):
    """
    Obtiene los canales de color (rojo, verde y azul) de una imagen y genera imágenes para cada canal.

    Args:
        imagen (array): La imagen original en formato RGB.
        etiqueta (str): La etiqueta de clasificación de la imagen.
        valor (int): El número de la imagen.

    Returns:
        tuple: Una tupla que contiene los tres canales de color por separado (canal_g, canal_r, canal_z).
    """
    # Extraer los tres canales de color RGB
    canales = {"canal_g" : imagen[:, :, 0],
               "canal_r" : imagen[:, :, 1],
               "canal_z" : imagen[:, :, 2],
               }
    for clave, canal in canales.items():
        generar_imagen(canal, etiqueta, valor, cmap='gray')  
        guardar_figuras(valor, etiqueta, "extraccion_de_canales_" + clave)

    # Mostrar los canales de color por separado
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title(f"Imagen número {valor}, clasificada como {etiqueta}")
    plt.imshow(canales["canal_g"], cmap='Greens' if len(imagen.shape) == 3 else 'gray')
    plt.title('Canal Verde')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(canales["canal_r"], cmap='Reds' if len(imagen.shape) == 3 else 'gray')
    plt.title('Canal Rojo')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(canales["canal_z"], cmap='Blues' if len(imagen.shape) == 3 else 'gray')
    plt.title('Banda Z')
    plt.axis('off')
    
    guardar_figuras(valor, etiqueta, "visualizar_canales_color_por_color")
    
    return canales["canal_g"], canales["canal_r"], canales["canal_z"]


def blanco_y_negro(imagen, etiqueta, valor):
    """
    Convierte una imagen RGB en una imagen en blanco y negro y la guarda como un archivo

    Args:
        imagen (array): La imagen original en formato RGB
        etiqueta (str): La etiqueta de clasificación de la imagen
        valor (int): El número de la imagen

    Returns:
        array: La imagen en blanco y negro
    """
    imagen_bn = rgb2gray(imagen)
    generar_imagen(imagen_bn, etiqueta, valor, 'gray')
    guardar_figuras(valor, etiqueta, "imagen_bn")
    
    return imagen_bn


def analizar_histograma(imagen, etiqueta, valor):
    """
    Genera un histograma de la imagen en blanco y negro y lo guarda como un archivo.

    Args:
        imagen (array): La imagen original en formato RGB.
        etiqueta (str): La etiqueta de clasificación de la imagen.
        valor (int): El número de la imagen.

    Returns:
        None
    """
    imagen_bn = rgb2gray(imagen)
    hist, bins_center = exposure.histogram(imagen_bn)
    plt.figure(figsize=(32, 8))

    # Primer subplot: Imagen original
    plt.subplot(1, 2, 1)
    plt.title(f"Imagen número {valor}, clasificada como {etiqueta}")
    plt.imshow(imagen_bn, cmap='gray')
    plt.axis('off')

    # Segundo subplot: Histograma de la imagen
    plt.subplot(1, 2, 2)
    plt.plot(bins_center, hist, color='black')
    plt.title('Histograma de la imagen')
    plt.xlabel('Intensidad de píxeles')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.title(f"Imagen número {valor}, clasificada como {etiqueta}")
    guardar_figuras(valor, etiqueta, "histograma")


def reduccion_ruido(imagen, etiqueta, valor, sigma):
    """
    Aplica filtros de reducción de ruido a la imagen y la guarda como un archivo

    Args:
        imagen (array): La imagen original en formato RGB
        etiqueta (str): La etiqueta de clasificación de la imagen
        valor (int): El número de la imagen
        sigma (float): El valor de sigma para el filtro gaussiano

    Returns:
        tuple: Una tupla que contiene las imágenes tratadas con filtro gaussiano y filtro de mediana
    """
    filtros = {"gaussian": filters.gaussian, "median": filters.median}

    imagen_gaussian = filtros["gaussian"](imagen.copy(), sigma)
    generar_imagen(imagen_gaussian, etiqueta, valor)
    guardar_figuras(valor, etiqueta, "reduccion_ruido_gaussian_"+ str(sigma))

    imagen_median = filtros["median"](imagen.copy())
    generar_imagen(imagen_median, etiqueta, valor)
    guardar_figuras(valor, etiqueta, "reduccion_ruido_median")
    
    return imagen_gaussian, imagen_median


def umbralizacion(imagen, etiqueta, valor):
    """
    Aplica técnicas de umbralización a la imagen y la guarda como un archivo

    Args:
        imagen (array): La imagen original en blanco y negro
        etiqueta (str): La etiqueta de clasificación de la imagen
        valor (int): El número de la imagen

    Returns:
        array: La imagen binarizada
    """
    imagenes_umbralizadas = []
    
    filtros = {"otsu": filters.threshold_otsu, "mean": filters.threshold_mean, "local": filters.threshold_local}
    for filtro, funcion in filtros.items():
        thresh = funcion(imagen)
        imagen_umbralizada = imagen > thresh
        imagenes_umbralizadas.append(imagen_umbralizada)
        
        generar_imagen(imagen_umbralizada, etiqueta, valor, 'gray')
        guardar_figuras(valor, etiqueta, "umbralizacion_" + filtro)
        
    return imagenes_umbralizadas


def ecualizacion_histograma(imagen, etiqueta, valor, nbins = 256):
    """
    Aplica la ecualización del histograma a la imagen y la guarda como un archivo

    Args:
        imagen (array): La imagen original en blanco y negro
        etiqueta (str): La etiqueta de clasificación de la imagen
        valor (int): El número de la imagen
        nbins (int): El número de bins para la ecualización del histograma

    Returns:
        array: La imagen con histograma ecualizado
    """
    imagen = exposure.equalize_hist(imagen, nbins = nbins)
    generar_imagen(imagen, etiqueta, valor, 'gray')
    guardar_figuras(valor, etiqueta, "ecualizacion_histograma_" + str(nbins))
    
    return imagen


def realce_bordes(imagen, etiqueta, valor):
    """
    Realza los bordes de la imagen y la guarda como un archivo

    Args:
        imagen (array): La imagen original en blanco y negro
        etiqueta (str): La etiqueta de clasificación de la imagen
        valor (int): El número de la imagen

    Returns:
        array: La imagen con los bordes realzados
    """
    filtros = {"laplace": filters.laplace, "sobel": filters.sobel}
    imagenes_realzadas = []
    for filtro, funcion in filtros.items():
        imagen_realzada = funcion(imagen)
        imagenes_realzadas.append(imagen_realzada)
        generar_imagen(imagen_realzada, etiqueta, valor)
        guardar_figuras(valor, etiqueta, "realce_bordes_" + filtro)
        
    return imagenes_realzadas


def guardar_figuras(valor, etiqueta, nombre_archivo):
    """
    Guarda la figura generada como un archivo de imagen

    Args:
        valor (int): El número de la imagen
        etiqueta (str): La etiqueta de clasificación de la imagen
        nombre_archivo (str): El nombre del archivo de imagen
    """

    utilidades.crear_carpeta("analisis")
    plt.savefig("analisis/" + str(nombre_archivo) + "_" + str(valor) + "_" + str(etiqueta) + ".png")
    plt.close()


def generar_imagen(imagen, etiqueta, valor, cmap=None):
    """
    Genera una imagen con su respectivo título

    Args:
        imagen (array): La imagen a mostrar
        etiqueta (str): La etiqueta de clasificación de la imagen
        valor (int): El número de la imagen
        cmap (str, opcional): El mapa de colores a utilizar. Por defecto es None

    Returns:
        None
    """
    plt.figure()
    plt.imshow(imagen, cmap=cmap)
    plt.title(f"Imagen número {valor}, clasificada como {etiqueta}")
    plt.axis('off')


if __name__ == "__main__":
    numero_iteraciones = 9
    # Cargar las imágenes y etiquetas del dataset
    imagenes, etiquetas = cargar_dataset()

    # Iterar el número de veces elegido para obtener una muestra representativa para el análisis (P.Ej: 5)
    for i in range(numero_iteraciones):
        # Seleccionar un número de imagen al azar
        num_imagen = int(random.random() * 5555)
        
        # Mostrar la imagen original y guardarla
        imagen_original(imagenes[num_imagen], etiquetas[num_imagen], num_imagen)
        
        # Visualizar los canales de color por separado y guardarlos como imágenes individuales
        canal_g, canal_r, canal_z = visualizar_canales_color(imagenes[num_imagen], etiquetas[num_imagen], num_imagen)
        
        # Analizar el histograma de la imagen original
        analizar_histograma(imagenes[num_imagen], etiquetas[num_imagen], num_imagen)

        # Convertir la imagen original a blanco y negro, reducir el ruido, realizar umbralización,
        # ecualizar el histograma y realzar los bordes. Guardar las imágenes generadas.
        imagen_bn = blanco_y_negro(imagenes[num_imagen], etiquetas[num_imagen], num_imagen)
        imagen_sin_ruido = reduccion_ruido(imagen_bn, etiquetas[num_imagen], num_imagen, int(1+i))
        imagen_umbralizacion = umbralizacion(imagen_bn, etiquetas[num_imagen], num_imagen)
        imagen_ecualizada = ecualizacion_histograma(imagen_bn, etiquetas[num_imagen], num_imagen, int(random.random()*256))
        imagen_realzada = realce_bordes(imagen_bn, etiquetas[num_imagen], num_imagen)
    
num_imagen = [400, 3000, 5000]
for imagen in num_imagen:
    # imagen_bn = blanco_y_negro(imagenes[imagen], etiquetas[imagen], imagen)
    imagen_gaussiana, _ = reduccion_ruido(imagenes[imagen], etiquetas[imagen], imagen, 2)
    # imagen_ecualizada = ecualizacion_histograma(imagen_gaussiana, etiquetas[imagen], imagen, 2)
    # imagenes_realzadas = realce_bordes(imagen_gaussiana, etiquetas[imagen], imagen)
    # imagenes_umbralizadas= umbralizacion(imagenes_realzadas[1], etiquetas[imagen], imagen)