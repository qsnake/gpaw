# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import Numeric as num
import gridpaw.utilities.mpi as mpi


class Dummy:
    def __init__(self, ne, nspins):
        self.ne = ne
        self.nspins = nspins
        self.set_fermi_level(None)
        
    def set_communicator(self, comm):
        self.comm = comm
        
    def calculate(self, nspins, kpts):
        return 0, 0.0, 0.0

    def set_fermi_level(self, epsF):
        self.epsF = epsF

    def get_fermi_level(self):
        return self.epsF


class FixMom(Dummy):
    def __init__(self, ne, nspins, M):
        Dummy.__init__(self, ne, nspins)
        self.M = M
        
    def calculate(self, kpts):
        if self.nspins == 1:
            b = self.ne // 2
            f_n = kpts[0].f_n
            f_n[:b] = 2.0
            f_n[b:] = 0.0
            if 2 * b < self.ne:  # XXX warning here?
                f_n[b] = 1.0
            return -1, 0.0, 0.0

        ma = (self.ne + self.M) / 2
        mb = (self.ne - self.M) / 2
        fa_n, fb_n = [kpt.f_n for kpt in kpts]
        fa_n[:ma] = 1.0
        fb_n[:mb] = 1.0
        fa_n[ma:] = 0.0
        fb_n[mb:] = 0.0
        return -1, ma - mb, 0.0


class ZeroKelvin(Dummy):
    def __init__(self, ne, nspins):
        Dummy.__init__(self, ne, nspins)

    def calculate(self, kpts):
        if self.nspins == 1:
            b = self.ne // 2
            f_n = kpts[0].f_n
            f_n[:b] = 2.0
            f_n[b:] = 0.0
            if 2 * b < self.ne:  # XXX warning here?
                f_n[b] = 1.0
            return -1, 0.0, 0.0

        nb = len(kpts[0].eps_n)
        
        if self.comm.size>1: 
            nbands = len(kpts[0].eps_n)
            all_eps_n = num.zeros((self.comm.size, nb), num.Float)
            self.comm.all_gather(kpts[0].eps_n, all_eps_n)
            eps_n = all_eps_n
        else:
            eps_n = [kpt.eps_n for kpt in kpts]

        ea_n, eb_n = eps_n
        ma = 0
        mb = 0
        while ma + mb < self.ne:
            if mb == nb or (ma < nb and ea_n[ma] < eb_n[mb]):
                ma += 1
            else:
                mb += 1

        if self.comm.size>1: 
            f_n = num.zeros((self.comm.size, nb), num.Float)
        else:
            f_n = [kpt.f_n for kpt in kpts]
 
        fa_n, fb_n = f_n
        fa_n[:ma] = 1.0
        fb_n[:mb] = 1.0
        fa_n[ma:] = 0.0
        fb_n[mb:] = 0.0


        # copy back information
        if self.comm.size>1: 
            kpts[0].f_n = f_n[self.comm.rank]
            
        return -1, ma - mb, 0.0


class FermiDirac(Dummy):
    def __init__(self, ne, nspins, kT):
        Dummy.__init__(self, ne, nspins)
        self.kT = kT
        
    def calculate(self, kpts):
        if 0:
            print kpts[0].eps_n
            kpts[0].f_n[:] = (1.2,1.2,1.2,1.2,1.2, 2, 0)
            return -11, 0, 0 # XXXXXXX
    
        if self.epsF is None:
            # Fermi level not set! Make a good guess:
            self.guess_fermi_level(kpts)
            
        # Now find the correct Fermi level:
        niter = 0
        while True:
            n = 0.0
            dnde = 0.0
            magmom = 0.0
            for kpt in kpts:
                sign = 1.0 - 2 * kpt.s
                x = num.clip((kpt.eps_n - self.epsF) / self.kT, -100.0, 100.0)
                x = num.exp(x)
                kpt.f_n[:] = kpt.weight / (x + 1.0)
                dn = num.sum(kpt.f_n)
                n += dn
                magmom += sign * dn
                dnde += (dn - num.sum(kpt.f_n**2) / kpt.weight) / self.kT


            n = self.comm.sum(n)
            dnde = self.comm.sum(dnde)
            magmom = self.comm.sum(magmom)
        
            dn = self.ne - n
            if abs(dn) < 1.0e-9:
                break
            if abs(dnde) <  1.0e-9:
                self.guess_fermi_level(kpts)
                continue
            if niter > 1000:
                raise RuntimeError, 'Could not locate the Fermi level!'
            de = dn / dnde
            if abs(de) > self.kT:
                de *= self.kT / abs(de)
            self.epsF += de
            niter += 1

        S = 0.0
        for kpt in kpts:
            x = num.clip((kpt.eps_n - self.epsF) / self.kT, -100.0, 100.0)
            y = num.exp(x)
            z = y + 1.0
            y *= x
            y /= z
            y -= num.log(z)
            S -= kpt.weight * num.sum(y)

        S = self.comm.sum(S)

        if self.nspins == 1:
            magmom = 0.0

        return niter, magmom, S * self.kT

    def guess_fermi_level(self, kpts):
        nu = len(kpts) * self.comm.size
        nb = len(kpts[0].eps_n)

        # Make a long array for all the eigenvalues:
        list_eps_n =  num.array([kpt.eps_n for kpt in kpts])

        if self.comm.size > 1:
            eps_n = mpi.all_gather_array(self.comm, list_eps_n)
        else:
            eps_n = list_eps_n.flat
 
        # Sort them:
        eps_n = num.sort(eps_n)
        n = self.ne * nu
        self.epsF = 0.5 * (eps_n[n // 2] + eps_n[(n - 1) // 2])






