from multiprocessing import Process, Queue

import pandas as pd

class MiHilo(Process):
    def __init__(self, target, args, queue):
        super().__init__(target=target, args=args)
        self.queue = queue

    def run(self):
        result = self._target(*self._args)
        self.queue.put(result)

def f(X):
    return {"a": [X, X, X], "b": [X, X, X]}

def m():
    hilos = []
    queue = Queue()
    for i in range(4):
        hilo = MiHilo(target=f, args=([i],), queue=queue)
        hilo.start()
        hilos.append(hilo)

    for hilo in hilos:
        hilo.join()

    df = pd.DataFrame()
    while not queue.empty():
        result = queue.get()
        df = pd.concat([df, pd.DataFrame(result)], join= "outer", axis= 0, ignore_index= True)
    print(df)

if __name__ == '__main__':
    m()
