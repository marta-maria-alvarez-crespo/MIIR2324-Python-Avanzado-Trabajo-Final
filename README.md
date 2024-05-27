# Entrenamiento y Validación de RN para la clasificación de Galaxias
## Estudio de rendimiento y profiling con Timeit y cProfile

En este proyecto, se incluye un estudio de medición de tiempos para evaluar el rendimiento del algoritmo de clasificación de galaxias implementado. El objetivo de este estudio es analizar el tiempo que tarda el algoritmo en procesar diferentes conjuntos de datos y comparar su eficiencia en diferentes configuraciones.

Este estudio de medición de tiempos es útil para identificar posibles cuellos de botella en el algoritmo, optimizar su rendimiento y tomar decisiones informadas sobre la configuración más adecuada para diferentes escenarios de uso.

## Requisitos

- Se recomienda la creación de un entorno virtual personalizado para cargar el fichero de requirements.txt
- Es necesario descargar el dataset de ejemplo desde [AstroNN](https://astronn.readthedocs.io/en/latest/galaxy10.html). Otros datasets de tipo .h5 son compatibles con esta implementación, pero se recomienda utilizar este.

## Instalación

1. Clona el repositorio.
2. Navega a la carpeta [scripts](./scripts/)
3. Instala las librerías necesarias (fichero [requirements.txt](./requirements.txt))

## Uso

1. Configura el experimento acorde a tus necesidades modificando los valores del fichero [configuracion](./scripts/configuracion.json)
2. Ejecuta el programa desde el [main](./scripts/main.py)
3. En caso de querer realizar un estudio de profiling, ejecuta el fichero profiling.py, configurando previamente la ejecución deseada a través del fichero json.


## Contribución

Si deseas contribuir a este proyecto, sigue los siguientes pasos:

1. Haz un fork del repositorio.
2. Crea una rama nueva para tu contribución.
3. Realiza los cambios necesarios.
4. Envía un pull request.
