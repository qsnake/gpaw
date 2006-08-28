# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""A replacement for the ``time.clock()`` function.

From the clock man page::

       Note that the time can wrap around. On a 32bit system
       where CLOCKS_PER_SEC equals 1000000 this function will
       return the same value approximately every 72 minutes.

The ``clock()`` function defined below tries to fix this problem.
However, if the ``clock()`` function is not called often enough (more
than 72 minutes between two calls), then there is no way of knowing
how many times the ``time.clock()`` function has wrapped arround! - in
this case a huge number is returned (1.0e100).  This problem can be
avoided by calling the ``update()`` function at intervals smaller than
72 minutes."""

import time
import sys


wrap = 1e-6 * 2**32

# Global variables:
c0 = time.clock()
t0 = time.time()
cputime = 0.0
trouble = False


def clock():
    """clock() -> floating point number

    Return the CPU time in seconds since the start of the process."""

    update()
    if trouble:
        return 1.0e100
    return cputime

def update():
    global trouble, t0, c0, cputime
    if trouble:
        return
    t = time.time()
    c = time.clock()
    if t - t0 >= wrap:
        trouble = True
        return
    dc = c - c0
    if dc < 0.0:
        dc += wrap
    cputime += dc
    t0 = t
    c0 = c


class Timer:
    def __init__(self):
        self.timers = {}
        self.t0 = time.time()
        
    def start(self, name):
        self.timers[name] = self.timers.get(name, 0.0) - time.time()
        
    def stop(self, name):
        self.timers[name] += time.time()
            
    def gettime(self, name):
        t = self.timers[name]
        assert t > 0.0
        return t

    def write(self, out=sys.stdout):
        if len(self.timers) == 0:
            return
        print >> out, 'Timing:'
        t0 = time.time()
        tot = t0 - self.t0
        n = max([len(name) for name in self.timers]) + 1
        names_and_times = self.timers.items()
        names_and_times.sort()
        for name, t in names_and_times:
            if t < 0.0:
                t += t0
            r = t / tot
            p = 100 * r
            i = int(50 * r + 0.5)
            if i == 0:
                bar = '|'
            else:
                bar = '|%s|' % ('=' * (i - 1))
            print >> out, '%-*s%9.3f %5.1f%% %s' % (n, name + ':', t, p, bar)
                
    def add(self, timer):
        for name, t in timer.timers.items():
            self.timers[name] = self.timers.get(name, 0.0) + t

