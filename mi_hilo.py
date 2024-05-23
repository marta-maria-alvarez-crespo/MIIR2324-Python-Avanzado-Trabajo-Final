from threading import Thread
from main import configuracion


class MiHilo(Thread):
    def __init__(self, target, args):
        super().__init__(target=target, args=args)

    def run(self):
        self.result = self._target(*self._args)

    def get_result(self):
        return self.result
