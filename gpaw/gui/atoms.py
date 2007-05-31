from math import sqrt

import numpy as npy
from ASE.ChemicalElements.covalent_radius import covalent_radii

from gpaw.gui.read import read_from_files
from gpaw.gui.write import write_to_file


covalent_radii = npy.array([r or 2.0 for r in covalent_radii])


class Atoms:
    def reset(self):
        self.set_atoms(npy.identity(3),
                       (True, True, True),
                       npy.empty(0, int),
                       npy.empty((1, 0, 3)),
                       npy.array([npy.nan]),
                       npy.empty((1, 0, 3)))
        
    def read(self, filenames, slice):
        cell, periodic, Z, dft, RR, EE, FF = read_from_files(filenames, slice)
        self.set_atoms(cell, periodic, Z, RR, EE, FF)
        self.dft = dft
        
    def pckl(self, data):
        self.set_atoms(data['cell'], data['periodic'], data['numbers'],
                       data['positions'],
                       data['energies'],
                       data['forces'])

    def set_atoms(self, cell, periodic, Z, RR, EE, FF):
        self.RR = RR
        self.EE = EE
        self.FF = FF
        self.nframes = len(RR)
        self.cell = npy.asarray(cell)
        self.Z = npy.asarray(Z)
        n = self.natoms = len(self.Z)
        self.selected = npy.zeros(n, bool)
        self.set_dynamic()
        self.nselected = 0
        self.repeat = npy.ones(3, int)
        self.r = covalent_radii[self.Z] * 0.8

    def repeat_atoms(self, repeat):
        n = self.repeat.prod()
        repeat = npy.array(repeat)
        self.repeat = repeat
        N = repeat.prod()
        natoms = self.natoms // n
        RR = npy.empty((self.nframes, natoms * N, 3))
        FF = npy.empty((self.nframes, natoms * N, 3))
        Z = npy.empty(natoms * N, int)
        r = npy.empty(natoms * N)
        dynamic = npy.empty(natoms * N, bool)
        a0 = 0
        for i0 in range(repeat[0]):
            for i1 in range(repeat[1]):
                for i2 in range(repeat[2]):
                    a1 = a0 + natoms
                    RR[:, a0:a1] = self.RR[:, :natoms] + npy.dot((i0, i1, i2),
                                                                 self.cell)
                    FF[:, a0:a1] = self.FF[:, :natoms]
                    Z[a0:a1] = self.Z[:natoms]
                    r[a0:a1] = self.r[:natoms]
                    dynamic[a0:a1] = self.dynamic[:natoms]
                    a0 = a1
        self.RR = RR
        self.FF = FF
        self.Z = Z
        self.r = r
        self.dynamic = dynamic
        self.natoms = natoms * N
        self.selected = npy.zeros(natoms * N, bool)
        self.nselected = 0
        
    def graph(self, expr):
        code = compile(expr + ',', 'atoms.py', 'eval')

        n = self.nframes
        def d(n1, n2):
            return sqrt(((R[n1] - R[n2])**2).sum())
        A = self.cell
        S = self.selected
        D = self.dynamic[:, npy.newaxis]
        E = self.EE
        s = 0.0
        data = []
        for i in range(n):
            R = self.RR[i]
            F = self.FF[i]
            fmax = sqrt(max(((F * D)**2).sum(1)))
            e = E[i]
            data = eval(code)
            if i == 0:
                m = len(data)
                xy = npy.empty((m, n))
            xy[:, i] = data
            if i + 1 < n:
                s += sqrt(((self.RR[i + 1] - R)**2).sum())
        return xy

    def set_dynamic(self):
        if self.nframes == 1:
            self.dynamic = npy.ones(self.natoms, bool)
        else:
            self.dynamic = npy.zeros(self.natoms, bool)
            R0 = self.RR[0]
            for R in self.RR[1:]:
                self.dynamic |= (R0 != R).any(1)

    def write(self, filename, rotations, show_unit_cell):
        indices = range(self.nframes)
        p = filename.rfind('@')
        if p != -1:
            slice = filename[p + 1:]
            filename = filename[:p]
            indices = eval('indices[%s]' % slice)
            if isinstance(indices, int):
                indices = [indices]
        suffix = filename.split('.')[-1]
        if suffix not in ['traj', 'xyz', 'py', 'eps', 'png']:
            suffix = 'traj'
        write_to_file(filename, self, suffix, indices,
                      rotations=rotations, show_unit_cell=show_unit_cell)

    def delete(self, i):
        self.nframes -= 1
        RR = npy.empty((self.nframes, self.natoms, 3))
        FF = npy.empty((self.nframes, self.natoms, 3))
        EE = npy.empty(self.nframes)
        RR[:i] = self.RR[:i]
        RR[i:] = self.RR[i + 1:]
        self.RR = RR
        FF[:i] = self.FF[:i]
        FF[i:] = self.FF[i + 1:]
        self.FF = FF
        EE[:i] = self.EE[:i]
        EE[i:] = self.EE[i + 1:]
        self.EE = EE

    def aneb(self):
        n = self.nframes
        assert n % 5 == 0
        levels = n // 5
        n = self.nframes = 2 * levels + 3
        RR = npy.empty((self.nframes, self.natoms, 3))
        FF = npy.empty((self.nframes, self.natoms, 3))
        EE = npy.empty(self.nframes)
        for L in range(levels):
            RR[L] = self.RR[L * 5]
            RR[n - L - 1] = self.RR[L * 5 + 4]
            FF[L] = self.FF[L * 5]
            FF[n - L - 1] = self.FF[L * 5 + 4]
            EE[L] = self.EE[L * 5]
            EE[n - L - 1] = self.EE[L * 5 + 4]
        for i in range(3):
            RR[levels + i] = self.RR[levels * 5 - 4 + i]
            FF[levels + i] = self.FF[levels * 5 - 4 + i]
            EE[levels + i] = self.EE[levels * 5 - 4 + i]
        self.RR = RR
        self.FF = FF
        self.EE = EE

    def interpolate(self, m):
        assert self.nframes == 2
        self.nframes = 2 + m
        RR = npy.empty((self.nframes, self.natoms, 3))
        FF = npy.empty((self.nframes, self.natoms, 3))
        EE = npy.empty(self.nframes)
        RR[0] = self.RR[0]
        FF[0] = self.FF[0]
        EE[0] = self.EE[0]
        for i in range(1, m + 1):
            x = i / (m + 1.0)
            y = 1 - x
            RR[i] = y * self.RR[0] + x * self.RR[1]
            FF[i] = y * self.FF[0] + x * self.FF[1]
            EE[i] = y * self.EE[0] + x * self.EE[1]
        RR[-1] = self.RR[1]
        FF[-1] = self.FF[1]
        EE[-1] = self.EE[1]
        self.RR = RR
        self.FF = FF
        self.EE = EE
        
