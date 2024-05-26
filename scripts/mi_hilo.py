# Autora:  Marta María Álvarez Crespo
# Descripción:  Archivo de implementación para la ejecución multihilo y multiproceso (wip)
# Última modificación: 25 / 05 / 2024
# GitHub: www.github.com/marta-maria-alvarez-crespo/MIIR2324-Python-Avanzado-Trabajo-Final


import pandas as pd
from threading import Thread
from multiprocessing import Process


class MiHilo(Thread):
    """Representa un hilo personalizado que hereda de la clase Thread.

    :param target: La función objetivo que se ejecutará en el hilo.
    :type target: function
    :param args: Los argumentos que se pasarán a la función objetivo.
    :type args: tuple
    """

    def __init__(self, target, args):
        super().__init__(target=target, args=args)

    def run(self):
        """Ejecuta el hilo y guarda el resultado en la variable result.

        Este método se encarga de ejecutar el hilo y guardar el resultado en la variable result.
        El resultado se obtiene al llamar al método objetivo (_target) pasando los argumentos (_args) correspondientes.
        """
        self.result = self._target(*self._args)

    def get_result(self):
        """Obtiene el resultado del hilo.

        :return: El resultado del hilo.
        :rtype: object
        """
        return self.result


class MiProceso(Process):
    """Representa un proceso personalizado que hereda de la clase Process."""

    def __init__(self, target, args, queue):
        """Inicializa una instancia de la clase MiProceso.

        :param target: El objetivo de la función que se ejecutará en el proceso.
        :type target: function
        :param args: Los argumentos que se pasarán a la función objetivo.
        :type args: tuple
        :param queue: La cola en la que se almacenarán los resultados de la función objetivo.
        :type queue: Queue
        """
        super().__init__(target=target, args=args)
        self.queue = queue

    def run(self):
        """Ejecuta el proceso y coloca el resultado en la cola.

        Este método se ejecuta cuando se inicia el proceso y se encarga de llamar al objetivo del proceso (_target)
        pasándole los argumentos (_args). Luego, coloca el resultado en la cola (queue) para que pueda ser
        recuperado por el proceso principal.
        """
        result = self._target(*self._args)
        self.queue.put(result)


# def f(X, Y):
#     return pd.DataFrame({"a": [X], "b": [Y]}), "hola"

# def m():
#     hilos = []
#     for i in range(4):
#         hilo = MiHilo(target=f, args=([i]))
#         hilo.start()
#         hilos.append(hilo)

#     for hilo in hilos:
#         hilo.join()

#     df = pd.DataFrame()
#     for hilo in hilos:
#         result, hola = hilo.get_result()
#         df = pd.concat([df, pd.DataFrame(result)], join="outer", axis=0, ignore_index=True)
#     print(df)


# def m2():
#     neuronas = [1, 2, 3, 4]
#     capas = [5, 6, 7, 8]
#     futures = []

#     with ThreadPoolExecutor(max_workers=4) as pool:
#         for i in range(2):
#             for n in neuronas:
#                 for c in capas:
#                     futures.append(pool.submit(f, n, c))
#         results = [f.result() for f in futures]

#     df = pd.DataFrame()
#     for result in results:
#         df = pd.concat([df, pd.DataFrame(result[0])], join="outer", axis=0, ignore_index=True)
#         print(result[1])
#     print(df)


# def m3():
#     neuronas = [1, 2, 3, 4]
#     capas = [1, 2, 3, 4]

#     with ThreadPoolExecutor(max_workers=4) as pool:
#         results = list(pool.map(f, neuronas, capas))

#     df = pd.DataFrame()
#     for result in results:
#         df = pd.concat([df, pd.DataFrame(result[0])], join="outer", axis=0, ignore_index=True)
#     print(df)


# if __name__ == "__main__":
#     m2()
