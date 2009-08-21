"""Utilities to measure and estimate memory"""

# The functions  _VmB, memory, resident, and stacksize are based on
# Python Cookbook, recipe number 286222
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/286222

import os
import resource
import numpy as np

_proc_status = '/proc/%d/status' % os.getpid()

_scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
          'KB': 1024.0, 'MB': 1024.0*1024.0}


def _VmB(VmKey):
    """Private."""
    global _proc_status, _scale
     # get pseudo file  /proc/<pid>/status
    try:
        t = open(_proc_status)
        v = t.read()
        t.close()
        # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
        i = v.index(VmKey)
    except:
        return 0.0  # non-Linux?

    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
     # convert Vm value to bytes
    return float(v[1]) * _scale[v[2]]


def maxrss():
    """Return maximal resident memory size in bytes."""
    # see http://www.kernel.org/doc/man-pages/online/pages/man5/proc.5.html

    # try to get it from rusage
    mm = resource.getrusage(resource.RUSAGE_SELF)[2]*resource.getpagesize()
    if mm > 0:
        return mm

    # try to get it from /proc/id/status
    for name in ('VmHWM:',  # Peak resident set size ("high water mark")
                 'VmRss:',  # Resident set size
                 'VmPeak:', # Peak virtual memory size
                 'VmSize:', # Virtual memory size
                 ):
        mm = _VmB(name)
        if mm > 0:
            return mm
    return 0.0 # no more ideas


class MemNode:
    """Represents the estimated memory use of an object and its components.

    Can be used on any object which implements estimate_memory().
    Example::

      from sys import stdout
      from gpaw.utilities.memory import MemNode
      node = MemNode('Root') # any name will do
      some_object.estimate_memory(node)
      nbytes = node.calculate_size()
      print 'Bytes', nbytes
      node.write(stdout) # write details
      
    Note that calculate_size() must be called before write().  Some
    objects must be explicitly initialized before they can estimate
    their memory use.
    """    
    floatsize = np.array(1, float).itemsize
    complexsize = np.array(1, complex).itemsize
    itemsize = {float : floatsize, complex : complexsize}
    
    def __init__(self, name, basesize=0):
        """Create node with specified name and intrinsic size without
        subcomponents."""
        self.name = name
        self.basesize = float(basesize)
        self.totalsize = np.nan # Size including sub-objects
        self.nodes = []
        self.indent = '    '

    def write(self, txt, maxdepth=-1, depth=0):
        """Write representation of this node and its subnodes, recursively.

        The depth parameter determines indentation.  maxdepth of -1 means
        infinity."""
        print >> txt, ''.join([depth * self.indent, self.name, '  ',
                               self.memformat(self.totalsize)])
        if depth == maxdepth:
            return
        for node in self.nodes:
            node.write(txt, maxdepth, depth + 1)
        
    def memformat(self, bytes):
        # One MiB is 1024*1024 bytes, as opposed to one MB which is ambiguous
        return '%.2f MiB' % (bytes / float(1 << 20))

    def calculate_size(self):
        self.totalsize = self.basesize
        for node in self.nodes:
            self.totalsize += node.calculate_size()
        # Datatype must not be fixed-size np integer
        return self.totalsize

    def subnode(self, name, basesize=0):
        """Create subcomponent with given name and intrinsic size.  Use this 
        to build component tree."""
        mem = MemNode(name, basesize)
        self.nodes.append(mem)
        return mem
    
    def setsize(self, basesize):
        self.basesize = float(basesize)
