# Autora:  Marta María Álvarez Crespo
# Descripción: Funciones y utilidades para entrenar, evaluar y visualizar CNN para tareas de clasificación utilizando transfer-learning
# Última modificación: 25 / 05 / 2024
# GitHub: www.github.com/marta-maria-alvarez-crespo/MIIR2324-Python-Avanzado-Trabajo-Final


from itertools import cycle
import numpy as np
from tensorflow import keras
from tensorflow.keras import callbacks, layers, optimizers
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import utilidades
import funciones_datos


def cargar_mn(input_shape):
    """Carga el modelo MobileNet pre-entrenado.

    :param input_shape: Forma de la entrada de la imagen.
    :type input_shape: tuple
    :return: El modelo MobileNet pre-entrenado.
    :rtype: keras.models.Model
    """
    modeloCNN = keras.applications.MobileNet(include_top=False, input_shape=input_shape)
    modeloCNN.summary()
    return modeloCNN


def cargar_vgg(input_shape):
    """Carga el modelo VGG16 pre-entrenado.

    :param input_shape: Forma de la entrada de la imagen.
    :type input_shape: tuple
    :return: El modelo VGG16 pre-entrenado.
    :rtype: keras.models.Model
    """
    modeloCNN = keras.applications.VGG16(include_top=False, input_shape=input_shape)
    modeloCNN.summary()

    return modeloCNN


def crear_clasificador(
    entrada: tuple, neuronas: int, dropout: float, activation: str, numero_de_clases: int, numero_de_capas: int = 1
):
    """Crea un clasificador de redes neuronales.

    :param entrada: Tupla que representa la forma de los datos de entrada.
    :type entrada: tuple
    :param neuronas: Número de neuronas en cada capa oculta.
    :type neuronas: int
    :param dropout: Valor de dropout para regularización.
    :type dropout: float
    :param activation: Función de activación para las capas ocultas.
    :type activation: str
    :param numero_de_clases: Número de clases en el problema de clasificación.
    :type numero_de_clases: int
    :param numero_de_capas: Número de capas ocultas en el modelo, por defecto es 1.
    :type numero_de_capas: int, opcional
    :return: Modelo de clasificador de redes neuronales.
    :rtype: keras.models.Model
    """
    input_modelo = keras.Input(shape=entrada)
    output_modelo = layers.Flatten()(input_modelo)

    for _ in range(numero_de_capas):
        output_modelo = layers.Dense(neuronas, activation=activation)(output_modelo)
        output_modelo = layers.Dropout(dropout)(output_modelo)
    output_modelo = layers.Dense(numero_de_clases, activation="softmax")(output_modelo)

    modelo = keras.models.Model(
        inputs=input_modelo, outputs=output_modelo, name="TOP_" + str(round(np.random.rand() * 1000, 4))
    )

    return modelo


# Funciones relacionadas con el proceso de Transfer-Learning


def entrenar_modelo(max_epochs, modelo, pred_entrenamiento, target_entrenamiento):
    """Entrena el modelo de transfer learning con los datos de entrenamiento.

    :param max_epochs: Número máximo de epochs de entrenamiento.
    :type max_epochs: int
    :param modelo: Modelo a entrenar.
    :type modelo: keras.Model
    :param pred_entrenamiento: Datos de entrenamiento para las características de entrada.
    :type pred_entrenamiento: numpy.ndarray
    :param target_entrenamiento: Datos de entrenamiento para las etiquetas objetivo.
    :type target_entrenamiento: numpy.ndarray
    :return: Resumen del entrenamiento del modelo.
    :rtype: keras.callbacks.History
    """
    n_iter_no_change = 5

    earlystop_callback = callbacks.EarlyStopping(
        monitor="val_loss", verbose=1, restore_best_weights=True, patience=n_iter_no_change
    )

    resumen = modelo.fit(
        x=pred_entrenamiento,
        y=target_entrenamiento,
        validation_split=0.2,
        batch_size=32,
        callbacks=earlystop_callback,
        epochs=max_epochs,
    )

    return resumen


def metricas_entrenamiento(history, nombre, config, preprocesado):
    """Genera y guarda las métricas de entrenamiento para un modelo de aprendizaje profundo.

    :param history: Historia del entrenamiento del modelo.
    :type history: keras.callbacks.History
    :param nombre: Nombre del modelo.
    :type nombre: str
    :param config: Configuración del modelo.
    :type config: str
    :param preprocesado: Tipo de preprocesamiento aplicado a los datos.
    :type preprocesado: str
    """

    # Gráfica de pérdida y exactitud
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(acc))

    # Gráfica de la precisión del modelo
    plt.figure()
    plt.plot(epochs, acc, "b", label="Precisión de entrenamiento")
    plt.plot(epochs, val_acc, "r", label="Precisión de validación")
    plt.title(f"Precisión de entrenamiento y validación para\n{config}_{nombre}")
    plt.legend()

    # Gráfica de la pérdida del modelo
    plt.figure()
    plt.plot(epochs, loss, "b", label="Pérdida de entrenamiento")
    plt.plot(epochs, val_loss, "r", label="Pérdida de validación")
    plt.title(f"Pérdida de entrenamiento y validación para\n{config}_{nombre}")
    plt.legend()

    # Guardar las gráficas
    utilidades.crear_carpeta("../metricas_entrenamiento/" + nombre + "/" + preprocesado)
    plt.savefig(
        "metricas_entrenamiento/" + nombre + "/" + preprocesado + "/perdida_exactitud_" + nombre + "_" + config + ".png"
    )
    plt.close()


def metricas_evaluacion(pred, target_test, nombre, config, preprocesado):
    """Calcula y guarda las métricas de evaluación para un modelo de aprendizaje profundo.

    :param pred: Predicciones del modelo.
    :type pred: numpy.ndarray
    :param target_test: Etiquetas verdaderas del conjunto de prueba.
    :type target_test: numpy.ndarray
    :param nombre: Nombre del modelo.
    :type nombre: str
    :param config: Configuración del modelo.
    :type config: str
    :param preprocesado: Tipo de preprocesamiento aplicado a los datos.
    :type preprocesado: str
    """

    # Gráficas de la curva ROC
    _, ax = plt.subplots(figsize=(6, 6))

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])

    for id_clase, color in zip(range(3), colors):
        RocCurveDisplay.from_predictions(
            target_test[:, id_clase],
            pred[:, id_clase],
            name=f"Curva ROC para la clase {id_clase}",
            color=color,
            ax=ax,
            plot_chance_level=(id_clase == 2),
        )

    _ = ax.set(
        xlabel="Tasa de Falsos Positivos (FP)",
        ylabel="Tasa de Verdaderos Positivos (TP)",
        title="Extensión de la Característica Operativa del Receptor\na Multiclase Uno-contra-Resto",
    )

    # Guardar las gráficas
    utilidades.crear_carpeta("../metricas_evaluacion/" + nombre + "/" + preprocesado)
    plt.savefig("metricas_evaluacion/" + nombre + "/" + preprocesado + "/roc_curve_" + nombre + "_" + config + ".png")
    plt.close()

    # Matriz de confusión
    target_test = np.argmax(target_test, axis=1)
    pred = np.argmax(pred, axis=1)

    ConfusionMatrixDisplay.from_predictions(target_test, pred, cmap="Blues")
    plt.title("Matriz de Confusión")

    # Guardar las gráficas
    plt.savefig(
        "metricas_evaluacion/" + nombre + "/" + preprocesado + "/confusion_matrix_" + nombre + "_" + config + ".png"
    )
    plt.close()


def evaluar_modelo(
    max_epoch,
    modelo,
    pred_entrenamiento,
    pred_test,
    target_entrenamiento,
    target_test,
    nombre_dicc_cnn,
    config,
    tipo_ejecucion,
):
    """Entrena y evalúa un modelo de red neuronal utilizando transfer-learning.

    :param max_epoch: Número máximo de iteraciones para el entrenamiento.
    :type max_epoch: int
    :param modelo: Modelo de red neuronal a entrenar.
    :type modelo: keras.models.Model
    :param pred_entrenamiento: Datos de entrenamiento (características).
    :type pred_entrenamiento: numpy.ndarray
    :param pred_test: Datos de prueba (características).
    :type pred_test: numpy.ndarray
    :param target_entrenamiento: Etiquetas de entrenamiento (objetivos).
    :type target_entrenamiento: numpy.ndarray
    :param target_test: Etiquetas de prueba (objetivos).
    :type target_test: numpy.ndarray
    :param nombre_dicc_cnn: Nombre del modelo.
    :type nombre_dicc_cnn: str
    :param config: Configuración específica del modelo.
    :type config: str
    :param tipo_ejecucion: Tipo de ejecución (entrenamiento o evaluación).
    :type tipo_ejecucion: str
    :return: Predicciones del modelo y etiquetas verdaderas del conjunto de prueba.
    :rtype: tuple
    """
    resumen = entrenar_modelo(max_epoch, modelo, pred_entrenamiento, target_entrenamiento)
    # metricas_entrenamiento(resumen, nombre_dicc_cnn, config, tipo_ejecucion)
    resultados_predict = modelo.predict(pred_test)
    # metricas_evaluacion(resultados_predict, target_test, nombre_dicc_cnn, config, tipo_ejecucion)

    return resultados_predict, target_test


def crear_dataframe(pred, target_test, nombre, preprocesado):
    """Calcula las métricas de evaluación del modelo y genera un DataFrame.

    :param pred: Predicciones del modelo.
    :type pred: numpy.ndarray
    :param target_test: Etiquetas verdaderas del conjunto de prueba.
    :type target_test: numpy.ndarray
    :param nombre: Nombre del modelo.
    :type nombre: str
    :param preprocesado: Nombre del procesado previo de imagen a clasificar.
    :type preprocesado: str
    :return: DataFrame con las métricas de evaluación.
    :rtype: pandas.DataFrame
    """
    target_test = np.argmax(target_test, axis=1)
    pred = np.argmax(pred, axis=1)

    reporte = classification_report(target_test, pred, digits=4, output_dict=True, zero_division=np.nan)

    # Crear una lista de diccionarios que contienen las métricas de interés
    data = []
    for clase, metrics in reporte.items():
        if clase in ["0", "1", "2"]:
            data.append(
                {
                    "Modelo de entrenamiento utilizado": nombre,
                    "Tipo de imagen": preprocesado,
                    "Clase a predecir": clase,
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1-Score": metrics["f1-score"],
                    "Accuracy": reporte["accuracy"],
                }
            )

    # Crear el DataFrame
    df = pd.DataFrame(data)

    # Ordenar el DataFrame por la columna 'Clase a predecir'
    df = df.sort_values(by="Clase a predecir", ignore_index=True)
    df["Clase a predecir"] = df["Clase a predecir"].map({"0": "Galaxia A", "1": "Galaxia B", "2": "Galaxia C"})

    return df


def transfer_learning(
    neurona: int,
    dropout: float,
    activacion: str,
    capa: int,
    max_epoch_tl: int,
    et_filtradas: np.ndarray,
    pred_entrenamiento: np.ndarray,
    pred_test: np.ndarray,
    target_entrenamiento: np.ndarray,
    target_test: np.ndarray,
    df: pd.DataFrame,
    nombre_dicc_cnn: str,
    tasa_aprendizaje: float,
    preprocesado: str,
):
    """Entrenamiento y evaluación de un modelo utilizando transfer-learning.

    :param neurona: Número de neuronas en la capa oculta.
    :type neurona: int
    :param dropout: Valor de dropout para regularización.
    :type dropout: float
    :param activacion: Función de activación de la capa oculta.
    :type activacion: str
    :param capa: Número de capas ocultas.
    :type capa: int
    :param max_epoch_tl: Número máximo de epochs de entrenamiento.
    :type max_epoch_tl: int
    :param et_filtradas: Etiquetas filtradas para el entrenamiento.
    :type et_filtradas: numpy.ndarray
    :param pred_entrenamiento: Datos de entrada de entrenamiento.
    :type pred_entrenamiento: numpy.ndarray
    :param pred_test: Datos de entrada de prueba.
    :type pred_test: numpy.ndarray
    :param target_entrenamiento: Etiquetas de entrenamiento.
    :type target_entrenamiento: numpy.ndarray
    :param target_test: Etiquetas de prueba.
    :type target_test: numpy.ndarray
    :param df: DataFrame que contiene métricas de evaluación.
    :type df: pandas.DataFrame
    :param nombre_dicc_cnn: Nombre del modelo de red neuronal.
    :type nombre_dicc_cnn: str
    :param tasa_aprendizaje: Tamaño de los ajustes realizados a los pesos durante el entrenamiento.
    :type tasa_aprendizaje: float
    :param preprocesado: Nombre del procesado previo de imagen a clasificar.
    :type preprocesado: str
    :return: DataFrame actualizado con métricas de evaluación, configuración del modelo y el modelo entrenado.
    :rtype: tuple
    """

    # Creación de las configuraciones para experimentar
    modelo = crear_clasificador(
        pred_entrenamiento.shape[1:], neurona, dropout, activacion, et_filtradas[0].shape[0], capa
    )
    config = f"TOP_{str(neurona)}_{str(dropout)}_{str(activacion)}_{str(capa)}"

    # Compilar el modelo
    modelo.compile(
        optimizer=optimizers.Adam(learning_rate=tasa_aprendizaje), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    target, predict_II = evaluar_modelo(
        max_epoch=max_epoch_tl,
        modelo=modelo,
        pred_entrenamiento=pred_entrenamiento,
        pred_test=pred_test,
        target_entrenamiento=target_entrenamiento,
        target_test=target_test,
        nombre_dicc_cnn=nombre_dicc_cnn,
        config=config,
        tipo_ejecucion=preprocesado,
    )

    # Concatenación de resultados al DataFrame
    minidf = crear_dataframe(predict_II, target, nombre_dicc_cnn + " | " + config, preprocesado)

    # Actualización del DataFrame
    if not (len(df)):
        df = minidf
    else:
        df = pd.concat([df, minidf], axis=0, ignore_index=True)

    return df, config, modelo


# Funciones relacionadas con el proceso de Fine-Tunning


def mejor_modelo_df(dicc, configuraciones, df_mini, im_filtradas):
    """Devuelve el mejor modelo de una lista de modelos y configuraciones.

    :param dicc: Un diccionario que contiene los modelos disponibles.
    :type dicc: dict
    :param configuraciones: Un diccionario que contiene las configuraciones disponibles.
    :type configuraciones: dict
    :param df_mini: Un DataFrame que contiene los resultados de los modelos.
    :type df_mini: pandas.DataFrame
    :param im_filtradas: Un array que contiene las imágenes filtradas.
    :type im_filtradas: numpy.ndarray
    :return: El mejor modelo y su configuración correspondiente.
    :rtype: tuple
    """
    nombre_cnn, nombre_top = df_mini["Accuracy"].idxmax().rsplit(" | ")
    cnn = dicc[nombre_cnn](im_filtradas.shape[1:])
    cnn.trainable = False
    top = configuraciones[nombre_top]

    return cnn, top


def reconstruccion_mejor_modelo_df(cnn, top):
    """Reconstruye el mejor modelo de un clasificador de redes neuronales convolucionales.

    :param cnn: El modelo de la red neuronal convolucional.
    :type cnn: keras.models.Model
    :param top: La capa superior del modelo.
    :type top: keras.layers.Layer
    :return: El modelo completo reconstruido.
    :rtype: keras.models.Model
    """
    full_output = top(cnn.output)
    modelo_completo = keras.models.Model(inputs=cnn.input, outputs=full_output)
    modelo_completo.summary()

    return modelo_completo
