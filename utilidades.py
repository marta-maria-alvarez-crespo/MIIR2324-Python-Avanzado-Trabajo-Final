# Autores:  Marta María Álvarez Crespo y Juan Manuel Ramos Pérez
# Descripción:  Archivo de utilidades para el manejo de datos
# Última modificación: 23 / 03 / 2024

import os

def crear_carpeta(nombre_carpeta: str = "nombre_carpeta"):
    """
    Crea una nueva carpeta en el sistema de archivos si no existe.

    Args:
        nombre_carpeta (str, opcional): Nombre de la carpeta a crear. 
                                         El valor por defecto es "nombre_carpeta".

    Returns:
        None
    """
    if not os.path.exists(nombre_carpeta):
        os.makedirs(nombre_carpeta)
    
