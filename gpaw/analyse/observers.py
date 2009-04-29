
class Observer:
    def __init__(self, interval=1):
        self.niter = 0
        self.interval = interval

    def __call__(self, *args, **kwargs):
        self.niter += self.interval
        self.update(*args, **kwargs)

    def update(self):
        raise RuntimeError('Virtual member function called.')
