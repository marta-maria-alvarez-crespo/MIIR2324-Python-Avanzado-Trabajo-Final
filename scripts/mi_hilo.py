from threading import Thread
from multiprocess import Process, Queue
from pathos.multiprocessing import ProcessPool
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

def f(X, Y):
    return pd.DataFrame({"a": [X, X, X], "b": [Y, Y, Y]}), "hola"

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


def mp():
    hilos = []
    cola = Queue()
    for i in range(4):
        hilo = MiProceso(target=f, args=(i, i+1), queue = cola)
        hilo.start()
        hilos.append(hilo)

    for hilo in hilos:
        hilo.join()

    df = pd.DataFrame()
    while not cola.empty():
        mini_df_tl, hola = cola.get()
        if not len(df): df = mini_df_tl
        else: df = pd.concat([df, mini_df_tl], join= "outer", axis=0, ignore_index=True)      
        print("HOLA")
    print(df)

def m():
    hilos = []
    for i in range(4):
        hilo = MiHilo(target=f, args=([i]))
        hilo.start()
        hilos.append(hilo)

    for hilo in hilos:
        hilo.join()

    df = pd.DataFrame()
    for hilo in hilos:
        result, hola = hilo.get_result()
        df = pd.concat([df, pd.DataFrame(result)], join= "outer", axis= 0, ignore_index= True)
    print(df)

def m2():
    neuronas = [1,2,3,4]
    capas = [5, 6, 7, 8]
    futures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n in neuronas:
            for c in capas:
                futures.append(pool.submit(f, n, c))
        results = [f.result() for f in futures]

    df = pd.DataFrame()
    for result in results:
        df = pd.concat([df, pd.DataFrame(result[0])], join= "outer", axis= 0, ignore_index= True)
        print(result[1])
    print(df)
    
def mp2():
    neuronas = [1,2,3,4]
    capas = [5, 6, 7, 8]
    with ProcessPool(nodes=4) as pool:
        results = []
        for n in neuronas:
            for c in capas:
                result = pool.apipe(f, n, c)
                results.append(result)

        df = pd.DataFrame()
        for result in results:
            r = result.get()
            df = pd.concat([df, pd.DataFrame(r[0])], join= "outer", axis= 0, ignore_index= True)
            print(r[1])
        print(df)

def m3():
    neuronas = [1,2,3,4]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(f, neuronas, [i+1 for i in neuronas]))

    df = pd.DataFrame()
    for result in results:
        df = pd.concat([df, pd.DataFrame(result)], join= "outer", axis= 0, ignore_index= True)
    print(df)

if __name__ == "__main__":
    mp2()