from threading import Thread


class MiHilo(Thread):
    def __init__(self, target, args):
        super().__init__(target=target, args=args)

    def run(self):
        self.result = self._target(*self._args)

    def result(self):
        return self.result
