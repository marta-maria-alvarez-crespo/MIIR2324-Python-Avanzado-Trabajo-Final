# Autores:  Marta María Álvarez Crespo y Juan Manuel Ramos Pérez
# Descripción:  Archivo de utilidades para el manejo de datos
# Última modificación: 23 / 03 / 2024

import os


def crear_carpeta(nombre_carpeta: str = "nombre_carpeta"):
    """Crea una carpeta con el nombre especificado.

    :param nombre_carpeta: El nombre de la carpeta a crear, por defecto es "nombre_carpeta"
    :type nombre_carpeta: str, opcional
    """
    if not os.path.exists(nombre_carpeta):
        os.makedirs(nombre_carpeta)
