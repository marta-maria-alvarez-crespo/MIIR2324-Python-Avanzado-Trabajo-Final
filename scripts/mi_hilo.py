# Autora:  Marta María Álvarez Crespo
# Descripción:  Archivo de implementación para la ejecución multihilo y multiproceso (wip)
# Última modificación: 25 / 05 / 2024
# GitHub: www.github.com/marta-maria-alvarez-crespo/MIIR2324-Python-Avanzado-Trabajo-Final


import pandas as pd
from threading import Thread
from multiprocessing import Process
from concurrent.futures import ThreadPoolExecutor


def f(X, Y):
    return pd.DataFrame({"a": [X], "b": [Y]}), "hola"


class MiHilo(Thread):
    def __init__(self, target, args):
        super().__init__(target=target, args=args)

    def run(self):
        self.result = self._target(*self._args)

    def get_result(self):
        return self.result


class MiProceso(Process):
    def __init__(self, target, args, queue):
        super().__init__(target=target, args=args)
        self.queue = queue

    def run(self):
        result = self._target(*self._args)
        self.queue.put(result)


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
