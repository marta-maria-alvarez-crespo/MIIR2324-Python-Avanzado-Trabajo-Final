# Autora:  Marta María Álvarez Crespo
# Descripción:  Archivo de utilidades para el manejo de datos
# Última modificación: 25 / 03 / 2024
# GitHub: www.github.com/marta-maria-alvarez-crespo/MIIR2324-Python-Avanzado-Trabajo-Final

import os

def crear_carpeta(nombre_carpeta: str = "nombre_carpeta"):
    """Crea una carpeta con el nombre especificado.

    :param nombre_carpeta: El nombre de la carpeta a crear, por defecto es "nombre_carpeta"
    :type nombre_carpeta: str, opcional
    """
    if not os.path.exists(nombre_carpeta):
        os.makedirs(nombre_carpeta)
