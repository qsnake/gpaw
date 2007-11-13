# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import Numeric as num
import gpaw.mpi as mpi


class Dummy:
    def __init__(self, ne, nspins):
        self.ne = ne
        self.nspins = nspins
        self.set_fermi_level(None)
        self.kT = 0
        self.fixmom = False
        self.magmom = 0.0
        self.niter = -1
        self.S = 0.0
        self.kpt_comm = mpi.serial_comm
        
    def set_communicator(self, kpt_comm):
        self.kpt_comm = kpt_comm
        
    def calculate(self, kpts):
        self.calculate_band_energy(kpts)

    def set_fermi_level(self, epsF):
        self.epsF = epsF

    def get_fermi_level(self):
        return self.epsF

    def fix_moment(self, M):
        self.fixmom = True
        self.M = M

    def calculate_band_energy(self, kpt_u):
        # Sum up all eigenvalues weighted with occupation numbers:
        Eband = 0.0
        for kpt in kpt_u:
            Eband += num.dot(kpt.f_n, kpt.eps_n)    
        self.Eband = self.kpt_comm.sum(Eband)


class ZeroKelvin(Dummy):
    """Occupations for Gamma-point calculations without Fermi-smearing"""

    def calculate(self, kpts):
        if self.nspins == 1:
            assert len(kpts) == 1
            b = int(self.ne // 2)
            f_n = kpts[0].f_n
            f_n[:b] = 2.0
            f_n[b:] = 0.0
            if 2 * b < self.ne:
                f_n[b] = self.ne - 2*b
            self.magmom = 0.0
        elif self.fixmom:
            M = int(round(self.M))
            ne_s = [(self.ne + M) / 2, (self.ne - M) / 2]
            for kpt in kpts:
                b = ne_s[kpt.s]
                kpt.f_n[:b] = 1.0
                kpt.f_n[b:] = 0.0
            self.magmom = M
        else:
            nb = len(kpts[0].eps_n)
            if self.kpt_comm.size>1: 
                all_eps_n = num.zeros((self.kpt_comm.size, nb), num.Float)
                self.kpt_comm.all_gather(kpts[0].eps_n, all_eps_n)
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

            if self.kpt_comm.size>1: 
                f_n = num.zeros((self.kpt_comm.size, nb), num.Float)
            else:
                f_n = [kpt.f_n for kpt in kpts]
 
            fa_n, fb_n = f_n
            fa_n[:ma] = 1.0
            fb_n[:mb] = 1.0
            fa_n[ma:] = 0.0
            fb_n[mb:] = 0.0
            self.magmom = ma - mb
            # copy back information
            if self.kpt_comm.size>1: 
                kpts[0].f_n = f_n[self.kpt_comm.rank]

        self.calculate_band_energy(kpts)


class FermiDirac(Dummy):
    """Occupations with Fermi smearing"""

    def __init__(self, ne, nspins, kT):
        Dummy.__init__(self, ne, nspins)
        self.kT = kT
        
    def calculate(self, kpts):
        
        if self.epsF is None:
            # Fermi level not set.  Make a good guess:
            self.guess_fermi_level(kpts)
            
        # Now find the correct Fermi level:
        self.find_fermi_level(kpts)

        S = 0.0
        for kpt in kpts:
            if self.fixmom:
                x = num.clip((kpt.eps_n - self.epsF[kpt.s]) / self.kT, -100.0, 100.0)
            else:
                x = num.clip((kpt.eps_n - self.epsF) / self.kT, -100.0, 100.0)
            y = num.exp(x)
            z = y + 1.0
            y *= x
            y /= z
            y -= num.log(z)
            S -= kpt.weight * num.sum(y)

        self.S = self.kpt_comm.sum(S) * self.kT
        self.calculate_band_energy(kpts)

    def guess_fermi_level(self, kpts):
        nu = len(kpts) * self.kpt_comm.size
        nb = len(kpts[0].eps_n)

        # Make a long array for all the eigenvalues:
        list_eps_n =  num.array([kpt.eps_n for kpt in kpts])

        if self.kpt_comm.size > 1:
            eps_n = mpi.all_gather_array(self.kpt_comm, list_eps_n)
        else:
            eps_n = list_eps_n.flat
 
        # Sort them:
        eps_n = num.sort(eps_n)
        n = int(self.ne * nu)
        if n // 2 == len(eps_n):
            self.epsF = 1000.0
        else:
            self.epsF = 0.5 * (eps_n[n // 2] + eps_n[(n - 1) // 2])
        if self.fixmom:
            self.epsF = num.array([self.epsF, self.epsF])

    def find_fermi_level(self, kpts):
        """Find the Fermi level by integrating in energy until
        the number of electrons is correct. For fixed spin moment calculations
        a separate Fermi level for spin up and down electrons is set
        in order to fix also the magnetic moment"""

        niter = 0
        while True:
            if self.fixmom:
                n = num.zeros(2, num.Float)
                dnde = num.zeros(2, num.Float)
            else:
                n = 0.0
                dnde = 0.0
            magmom = 0.0
            for kpt in kpts:
                sign = 1.0 - 2 * kpt.s
                if self.fixmom:
                    x = num.clip((kpt.eps_n - self.epsF[kpt.s]) / self.kT, -100.0, 100.0)
                    x = num.exp(x)
                    kpt.f_n[:] = kpt.weight / (x + 1.0)
                    dn = num.sum(kpt.f_n)
                    n[kpt.s] += dn
                    dnde[kpt.s] += (dn - num.sum(kpt.f_n**2) / kpt.weight) / self.kT
                else:
                    x = num.clip((kpt.eps_n - self.epsF) / self.kT, -100.0, 100.0)
                    x = num.exp(x)
                    kpt.f_n[:] = kpt.weight / (x + 1.0)
                    dn = num.sum(kpt.f_n)
                    n += dn
                    dnde += (dn - num.sum(kpt.f_n**2) / kpt.weight) / self.kT

                magmom += sign * dn

            # comm.sum has to be called differently when summing scalars
            # than when summing arrays
            if self.fixmom:
                self.kpt_comm.sum(n)
                self.kpt_comm.sum(dnde)
            else:
                n = self.kpt_comm.sum(n)
                dnde = self.kpt_comm.sum(dnde)
            magmom = self.kpt_comm.sum(magmom)

            if self.fixmom:
                ne = num.array([(self.ne + self.M) / 2, (self.ne - self.M) / 2])
                dn = ne - n
                if num.alltrue(abs(dn) < 1.0e-9):
                    if abs(magmom - self.M) > 1.0e-8:
                        raise RuntimeError, 'Magnetic moment not fixed'
                    break
                if num.sometrue(abs(dnde) <  1.0e-9):
                    self.guess_fermi_level(kpts)
                    continue
            else:
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

        if self.nspins == 1:
            magmom = 0.0

        self.niter = niter
        self.magmom = magmom



