"""Utilities to measure and estimate memory"""

# The functions  _VmB, memory, resident, and stacksize are based on
# Python Cookbook, recipe number 286222
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/286222

import os
import resource
import numpy as npy

_proc_status = '/proc/%d/status' % os.getpid()

_scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
          'KB': 1024.0, 'MB': 1024.0*1024.0}

def _VmB(VmKey):
    '''Private.
    '''
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

def memory(since=0.0):
    '''Return memory usage in bytes.
    '''
    return _VmB('VmSize:') - since


def resident(since=0.0):
    '''Return resident memory usage in bytes.
    '''
    return _VmB('VmRSS:') - since

def stacksize(since=0.0):
    '''Return stack size in bytes.
    '''
    return _VmB('VmStk:') - since

def maxrss():
    '''Return maximal resident memory size in bytes.
    '''
    # see http://www.kernel.org/doc/man-pages/online/pages/man5/proc.5.html

    # try to get it from rusage
    mm = resource.getrusage(resource.RUSAGE_SELF)[2]*resource.getpagesize()
    if mm > 0: return mm

    # try to get it from /proc/id/status
    mm = _VmB('VmHWM:') # Peak resident set size ("high water mark")
    if mm > 0: return mm

    # try to get it from /proc/id/status
    mm = _VmB('VmRss:') # Resident set size
    if mm > 0: return mm

    # try to get it from /proc/id/status
    mm = _VmB('VmPeak:') # Peak virtual memory size
    if mm > 0: return mm

    # Finally, try to get the current usage from /proc/id/status
    mm = _VmB('VmSize:') # Virtual memory size
    if mm > 0: return mm

    # no more idea
    return 0.0


def estimate_memory(paw):

    scales = {'GB': 1024.0**3,
              'MB': 1024.0**2}

    n_c = paw.gd.n_c
    h_c = paw.gd.h_c
    nbands = paw.wfs.mynbands
    nspins = paw.wfs.nspins
    #nkpts = paw.wfs.nkpts
    nmyu = len(paw.wfs.kpt_u)
    #nuclei = paw.nuclei
    out = paw.txt

    float_size = npy.array([1], float).itemsize
    type_size = npy.array([1],paw.wfs.dtype).itemsize

    # coarse grid size
    grid_size = long(n_c[0] * n_c[1] * n_c[2])

    mem = long(0)
    # initial overhead (correct probably only in linux)
    mem_init= memory()
    mem += mem_init

    # density object
    mem_density = grid_size * 8 * (1 + nspins)     # rhot_g + nt_sg
    mem_density += (grid_size * (1 + nspins))      # nct_G + nt_sG
    mem_density += (grid_size * 8 * (nspins - 1))  # nt_g
    # interpolator
    mem_density += (grid_size * 17) # ~buf + buf2
    mem_density *= float_size
    mem += mem_density

    # hamiltonian
    # potentials
    mem_hamilt = grid_size * nspins
    mem_hamilt += grid_size * 8 * (1 + nspins)
    # xc (not GGA contributions)
    mem_hamilt += (grid_size * 8 * nspins)
    # restrictor
    mem_hamilt += (grid_size * 12)
    # Poisson (rhos, phis, residuals, interpolators, restrictors)
    # Multigrid adds a factor of 1.14
    mem_hamilt += (grid_size * 8 * 4 * 1.14)
    mem_hamilt += (grid_size * 6 * 1.14)
    mem_hamilt += (grid_size * 12 * 1.14)
    mem_hamilt *= float_size
    # Laplacian
    mem_hamilt += grid_size * type_size
    mem += mem_hamilt

    """
    # Localized functions. Normally 1 value + 3 derivatives + 1 work array
    # are stored
    mem_nuclei = npy.zeros(paw.domain.comm.size, dtype=long)
    for nucleus in nuclei:
        ni = nucleus.get_number_of_partial_waves()
        np = ni * (ni + 1) // 2
        # D_sp and H_sp
        mem_nuclei[nucleus.rank] += (2 * nspins * np * float_size)
        # P_uni
        mem_nuclei[nucleus.rank] += (nmyu * nbands * ni * type_size)
        # projectors
        box = 2 * nucleus.setup.pt_j[0].get_cutoff() / h_c
        # func + derivatives
        mem_nuclei[nucleus.rank] += (4 * ni * box[0] * box[1] * box[2]
                                     * float_size)
        # work
        mem_nuclei[nucleus.rank] += (box[0] * box[1] * box[2] * type_size)
        # vbar
        box = 4 * nucleus.setup.vbar.get_cutoff() / h_c
        mem_nuclei[nucleus.rank] += (5 * box[0] * box[1] * box[2] * float_size)
        # step
        mem_nuclei[nucleus.rank] += (2 * box[0] * box[1] * box[2] * float_size)
        # ghat and vhat
        box = 4 * nucleus.setup.ghat_l[0].get_cutoff() / h_c
        nl = 0
        for ghat in nucleus.setup.ghat_l:
            l = ghat.get_angular_momentum_number()
            nl += 2 * l + 1
        mem_nuclei[nucleus.rank] += (4 * nl * box[0] * box[1] * box[2]
                                     * float_size)
        mem_nuclei[nucleus.rank] += (box[0] * box[1] * box[2] * float_size)
        # nct
        box = 2 * nucleus.setup.rcore / h_c
        
        # XXX Why times 5?  There would be 4 from the func + derivs,
        # and conceivably another 4 if including tauct, but never 5.
        # Won't work for HGH setups which have no core density.
        mem_nuclei[nucleus.rank] += (5 * box[0] * box[1] * box[2] * float_size)
        
    mem_nuclei = max(mem_nuclei)
    mem += mem_nuclei
    """
    mem_nuclei = 0.0


    #eigensolver (estimate for RMM-DIIS, CG higher)
    #preconditioner
    mem_eigen = grid_size * 4
    # Htpsit + dR + 1 temp for subspace diagonalization !!!remember to add
    mem_eigen += (nbands + 1) * grid_size
    mem_eigen += nbands * nbands
    mem_eigen *= type_size
    mem += mem_eigen

    # Wave functions
    mem_wave_functions = nmyu * nbands * grid_size
    mem_wave_functions *= type_size
    mem += mem_wave_functions

    # temporary data in initialization
    ### Fixme! with LCAO below estimate is no longer correct!
    nao_tot = 0
    mem_temp = npy.zeros(paw.gd.comm.size, dtype=long)
    #for nucleus in nuclei:
    #    box = 2 * nucleus.setup.phit_j[0].get_cutoff() / h_c
    #    box_size = box[0] * box[1] * box[2]
    #    box_size = min(grid_size, box_size)
    #    nao = nucleus.setup.niAO
    #    nao_tot += nao
    #    mem_temp[nucleus.rank] += (2 * nao * box_size)
    mem_temp = max(mem_temp)
    # print >> out, "Atomic orbitals:                  %.3f" % (mem_temp * type_size)

    mem_temp += nao_tot * grid_size
    mem_temp += (nao_tot - nbands) * grid_size
    mem_temp *= type_size
    # print >> out, "Temp. for initial wave functions: %.3f" % (mem_temp)
    # mem += mem_temp

    if mem > 1024.0**3:
        scalename = 'GB'
    else:
        scalename = 'MB'

    scale = scales[scalename]

    # Output
    print >> out
    #if paw.world.size > 1:
    #    print >> out, "Estimated maximum memory consumption per processor (%s):" % scalename
    #else:
    if 1:
        print >> out, "Estimated memory consumption (%s):" % scalename
    print >> out, "Initial overhead:       %.2f" % (mem_init / scale)
    print >> out, "Density object:         %.2f" % (mem_density / scale)
    print >> out, "Hamiltonian object:     %.2f" % (mem_hamilt / scale)
    print >> out, "Localized functions:    %.2f" % (mem_nuclei / scale)
    print >> out, "Eigensolver:            %.2f" % (mem_eigen / scale)
    print >> out, "Wave functions:         %.2f" % (mem_wave_functions / scale)
    print >> out, "---------------------------------"
    print >> out, "Total:                  %.2f %s" % (mem / scale, scalename)

class MemNode:
    """Represents the estimated memory use of an object and its components."""
    floatsize = npy.array(1, float).itemsize
    complexsize = npy.array(1, complex).itemsize
    itemsize = {float : floatsize, complex : complexsize}
    
    def __init__(self, name, basesize):
        """Create node with specified name and intrinsic size without
        subcomponents."""
        self.name = name
        self.basesize = float(basesize)
        self.totalsize = npy.nan # Size including sub-objects
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
