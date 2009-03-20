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
try:
    import pytau
except ImportError:
    pass

import gpaw.mpi as mpi
MASTER = 0

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
        self.running = []
        
    def start(self, name):
        self.timers[name] = self.timers.get(name, 0.0) - time.time()
        self.running.append(name)
        
    def stop(self, name=None):
        if name is None: name = self.running[-1]
        if name != self.running.pop():
            raise RuntimeError
        self.timers[name] += time.time()
            
    def gettime(self, name):
        t = self.timers[name]
        assert t > 0.0
        return t

    def reset(self):
        """Reset all timers"""
        for name in self.timers:
            if self.timers[name] < 0.0:
                self.timers[name] = -time.time()
            else:
                self.timers[name] = 0.0
                
    def write(self, out=sys.stdout):
        while self.running:
            self.stop()
        if len(self.timers) == 0:
            return
        print >> out
        print >> out, 'Timing:'
        print >> out, '-' * 60
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
        print >> out, '-' * 60
        print >> out, '%-*s%9.3f' % (n, 'Total' + ':', tot)
        print >> out
        print >> out, 'date:', time.asctime()
                
    def add(self, timer):
        for name, t in timer.timers.items():
            self.timers[name] = self.timers.get(name, 0.0) + t


class NullTimer(Timer):
    """Compatible with Timer and StepTimer interfaces.  Does nothing."""
    def __init__(self): pass
    def start(self, name): pass
    def stop(self, name=None): pass
    def gettime(self, name):
        return 0.0
    def reset(self): pass
    def write(self, out=sys.stdout): pass
    def add(self, timer): pass
    def write_now(self, mark=''): pass


nulltimer = NullTimer()


class StepTimer(Timer):
    """Step timer to print out timing used in computation steps.
    
    Use it like this::

      from gpaw.utilities.timing import StepTimer
      st = StepTimer()
      ...
      st.write_now('step 1')
      ...
      st.write_now('step 2')

    The parameter write_as_master_only can be used to force the timer to
    print from processess that are not the mpi master process.
    """
    
    def __init__(self,out=sys.stdout,name=None,write_as_master_only=True):
        Timer.__init__(self)
        if name is None:
            name = '<'+sys._getframe(1).f_code.co_name+'>'
        self.name = name
        self.out = out
        self.alwaysprint = not write_as_master_only
        self.now = 'temporary now'
        self.start(self.now)


    def write_now(self, mark=''):
        self.stop(self.now)
        if self.alwaysprint or mpi.rank == MASTER:
            print >> self.out, self.name, mark, self.gettime(self.now)
        self.out.flush()
        del self.timers[self.now]
        self.start(self.now)

class TauTimer(Timer):
    """TauTimers require installation of the TAU Performance System
    http://www.cs.uoregon.edu/research/tau/home.php

    The TAU Python API will not output any data if there are any
    unmatched starts/stops in the code."""
    
    def __init__(self):
        self.timers = {}
        self.running = []
        pytau.setNode(mpi.rank)
        self.start('PAW_calc') 

    def start(self, name):
        self.timers[name] = pytau.profileTimer(name)
        self.running.append(name)
        pytau.start(self.timers[name])
        
    def stop(self, name=None):
        if name is None: name = self.running[-1]
        if name != self.running.pop():
            raise RuntimeError
        pytau.stop(self.timers[name])

    def write(self, out=sys.stdout):
        self.stop('PAW_calc')

