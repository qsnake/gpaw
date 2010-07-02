import sys


class ProgressBar:
    def __init__(self, x0=0.0, x1=1.0, n=70, fd=sys.stdout):
        """Progress bar from x0 to x1 in n steps."""
        self.x0 = x0
        self.x1 = x1
        self.n = n
        self.fd = fd
        self.b0 = 0
        
    def __call__(self, x):
        """Update progress bar."""
        b = int(round((x - self.x0) / (self.x1 - self.x0) * self.n))
        if b > self.b0:
            self.fd.write('=' * (b - self.b0))
            self.fd.flush()
            self.b0 = b
