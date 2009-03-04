# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import numpy as np
import gpaw.mpi as mpi


class OccupationNumbers:
    def __init__(self, ne, nspins):
        self.ne = ne
        self.nspins = nspins
        self.set_fermi_level(None)
        self.kT = 0
        self.fixmom = False
        self.magmom = 0.0
        self.niter = -1
        self.S = 0.0
        self.kpt_comm = None#mpi.serial_comm
        self.band_comm = None#mpi.serial_comm
        
    def set_communicator(self, kpt_comm, band_comm=None):
        self.kpt_comm = kpt_comm
        if band_comm is None:
            band_comm = mpi.serial_comm
        self.band_comm = band_comm

    def calculate(self, wfs):
        for kpt in wfs.kpt_u:
            if kpt.f_n is None:
                kpt.f_n = np.empty_like(kpt.eps_n)

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
            Eband += np.dot(kpt.f_n, kpt.eps_n)    
        self.Eband = self.band_comm.sum(self.kpt_comm.sum(Eband))

    def get_homo_lumo(self, wfs):
        raise NotImplementedError('get_homo_lumo() only implemented for zero '
                                  'Kelvin calculations!')

    def get_zero_kelvin_homo_eigenvalue(self, kpt_u):
        homo = (self.ne // 2)-1
        return mpi.world.max(max([ kpt.eps_n[homo] for kpt in kpt_u]))

    def get_zero_kelvin_lumo_eigenvalue(self, kpt_u):
        lumo = (self.ne // 2)
        return -mpi.world.max(-min([ kpt.eps_n[lumo] for kpt in kpt_u]))

class ZeroKelvin(OccupationNumbers):
    """Occupations for Gamma-point calculations without Fermi-smearing"""

    def calculate(self, wfs):
        OccupationNumbers.calculate(self, wfs)

        kpts = wfs.kpt_u

        if ((self.kpt_comm.size == 1 and self.nspins != len(kpts)) or
            (self.kpt_comm.size == 2 and len(kpts) != 1) or
            self.kpt_comm.size > 2):
            raise RuntimeError('width=0 only works for gamma-point ' +
                               'calculations!  Use width > 0.')
        
        if self.nspins == 1:
            lumo = int(self.ne // 2)
            f_n = kpts[0].f_n
            f_n[:lumo] = 2.0
            f_n[lumo:] = 0.0
            if 2 * lumo < self.ne:
                f_n[lumo] = self.ne - 2 * lumo # == 1.0
                lumo += 1
            self.magmom = 0.0
        elif self.fixmom:
            M = int(round(self.M))
            lumo = (self.ne + M) / 2, (self.ne - M) / 2
            for kpt in kpts:
                b = lumo[kpt.s]
                kpt.f_n[:b] = 1.0
                kpt.f_n[b:] = 0.0
            self.magmom = M
        else:
            nb = len(kpts[0].eps_n)
            if self.kpt_comm.size > 1: 
                all_eps_n = np.zeros((self.kpt_comm.size, nb))
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
            lumo = ma, mb
            
            if self.kpt_comm.size > 1: 
                f_n = np.zeros((self.kpt_comm.size, nb)) # (2, nb)
            else:
                f_n = [kpt.f_n for kpt in kpts]
 
            fa_n, fb_n = f_n
            fa_n[:ma] = 1.0
            fb_n[:mb] = 1.0
            fa_n[ma:] = 0.0
            fb_n[mb:] = 0.0
            self.magmom = ma - mb
            # copy back information
            if self.kpt_comm.size > 1: 
                kpts[0].f_n[:] = f_n[self.kpt_comm.rank]

        self.lumo = lumo
        self.calculate_band_energy(kpts)

    def get_fermi_level(self):
        raise NotImplementedError('Fermi level only defined for width > 0. '
                                  'Use get_homo_lumo() instead.')

    def get_homo_lumo(self, wfs):
        if not hasattr(self, 'lumo'):
            self.calculate(wfs)

        kpts = wfs.kpt_u

        def get(a, i):
            if i < 0:
                return np.nan
            try:
                return a[i]
            except IndexError:
                return np.nan

        if self.nspins == 1:
            e_homo = kpts[0].eps_n[self.lumo - 1]
            e_lumo = get(kpts[0].eps_n, self.lumo)
        elif self.kpt_comm.size == 1:
            e_homo = max(get(kpts[0].eps_n, self.lumo[0] - 1),
                         get(kpts[1].eps_n, self.lumo[1] - 1))
            e_lumo = min(get(kpts[0].eps_n, self.lumo[0]),
                         get(kpts[1].eps_n, self.lumo[1]))
        else:
            eps = np.zeros((2, 2)) # proc, homo/lumo
            hl = np.array([get(kpts[0].eps_n, self.lumo[0] - 1),
                           get(kpts[0].eps_n, self.lumo[0])])
            self.kpt_comm.all_gather(hl, eps)
            e_homo = eps[:, 0].max()
            e_lumo = eps[:, 1].min()

        return np.array([e_homo, e_lumo])


class FermiDirac(OccupationNumbers):
    """Occupations with Fermi smearing"""

    def __init__(self, ne, nspins, kT):
        OccupationNumbers.__init__(self, ne, nspins)
        self.kT = kT
        
    def calculate(self, wfs):
        OccupationNumbers.calculate(self, wfs)

        kpts = wfs.kpt_u

        if self.epsF is None:
            # Fermi level not set.  Make a good guess:
            self.guess_fermi_level(kpts)

        # Now find the correct Fermi level:
        self.find_fermi_level(kpts)

        S = 0.0
        for kpt in kpts:
            if self.fixmom:
                x = np.clip((kpt.eps_n - self.epsF[kpt.s]) / self.kT, -100.0, 100.0)
            else:
                x = np.clip((kpt.eps_n - self.epsF) / self.kT, -100.0, 100.0)
            y = np.exp(x)
            z = y + 1.0
            y *= x
            y /= z
            y -= np.log(z)
            S -= kpt.weight * np.sum(y)

        self.S = self.band_comm.sum(self.kpt_comm.sum(S)) * self.kT
        self.calculate_band_energy(kpts)

    def guess_fermi_level(self, kpts):

        # XXX Only domain_comm.rank == 0 should do this stuff:

        kpt_comm = self.kpt_comm
        band_comm = self.band_comm

        # Make a long array for all the eigenvalues:
        eps_n =  np.array([kpt.eps_n for kpt in kpts]).ravel()

        if kpt_comm.size > 1:
            if kpt_comm.rank == 0:
                eps_qn = np.empty((kpt_comm.size, len(eps_n)))
                kpt_comm.gather(eps_n, 0, eps_qn)
                eps_n = eps_qn.ravel()
            else:
                kpt_comm.gather(eps_n, 0)

        epsF = np.array([42.0])

        if kpt_comm.rank == 0:
            if band_comm.size > 1:
                if band_comm.rank == 0:
                    eps_qn = np.empty((band_comm.size, len(eps_n)))
                    band_comm.gather(eps_n, 0, eps_qn)
                    eps_n = eps_qn.ravel()
                else:
                    band_comm.gather(eps_n, 0)

            if band_comm.rank == 0:
                eps_n = np.sort(eps_n)
                n = int(self.ne * len(kpts) * kpt_comm.size)
                if n // 2 == len(eps_n):
                    epsF = 1000.0
                else:
                    epsF = 0.5 * (eps_n[n // 2] + eps_n[(n - 1) // 2])
                epsF = np.array([epsF])

            band_comm.broadcast(epsF, 0)

        kpt_comm.broadcast(epsF, 0)
        
        self.epsF = epsF[0]
        if self.fixmom:
            self.epsF = np.array([self.epsF, self.epsF])
        
        
    def find_fermi_level(self, kpts):
        """Find the Fermi level by integrating in energy until
        the number of electrons is correct. For fixed spin moment calculations
        a separate Fermi level for spin up and down electrons is set
        in order to fix also the magnetic moment"""

        niter = 0
        while True:
            if self.fixmom:
                n = np.zeros(2)
                dnde = np.zeros(2)
            else:
                n = 0.0
                dnde = 0.0
            magmom = 0.0
            for kpt in kpts:
                sign = 1.0 - 2 * kpt.s
                if self.fixmom:
                    x = np.clip((kpt.eps_n - self.epsF[kpt.s]) / self.kT, -100.0, 100.0)
                    x = np.exp(x)
                    kpt.f_n[:] = kpt.weight / (x + 1.0)
                    dn = np.sum(kpt.f_n)
                    n[kpt.s] += dn
                    dnde[kpt.s] += (dn - np.sum(kpt.f_n**2) / kpt.weight) / self.kT
                else:
                    x = np.clip((kpt.eps_n - self.epsF) / self.kT, -100.0, 100.0)
                    x = np.exp(x)
                    kpt.f_n[:] = kpt.weight / (x + 1.0)
                    dn = np.sum(kpt.f_n)
                    n += dn
                    dnde += (dn - np.sum(kpt.f_n**2) / kpt.weight) / self.kT

                magmom += sign * dn

            # comm.sum has to be called differently when summing scalars
            # than when summing arrays
            if self.fixmom:
                self.kpt_comm.sum(n)
                self.kpt_comm.sum(dnde)
                self.band_comm.sum(n)
                self.band_comm.sum(dnde)
            else:
                n = self.band_comm.sum(self.kpt_comm.sum(n))
                dnde = self.band_comm.sum(self.kpt_comm.sum(dnde))
            magmom = self.band_comm.sum(self.kpt_comm.sum(magmom))

            if self.fixmom:
                ne = np.array([(self.ne + self.M) / 2, (self.ne - self.M) / 2])
                # we might be dividing by dnde, so it should not be too small
                dnde = np.where(abs(dnde) < 1.5e-10, 1.0e-10, dnde)
                dn = ne - n
                if np.alltrue(abs(dn) < 1.0e-9):
                    if abs(magmom - self.M) > 1.0e-8:
                        raise RuntimeError('Magnetic moment not fixed')
                    break
                if np.alltrue(abs(dnde) <  1.0e-9):
                    # make guess only if dnde is small for both spin channels
                    self.guess_fermi_level(kpts)

                    niter += 1
                    if niter > 1000:
                        raise RuntimeError('Could not locate the Fermi level!')
                    continue
            else:
                dn = self.ne - n
                if abs(dn) < 1.0e-9:
                    break
                if abs(dnde) <  1.0e-9:
                    self.guess_fermi_level(kpts)
                    niter += 1
                    if niter > 1000:
                        raise RuntimeError('Could not locate the Fermi level!'
                                           + '  See ticket #27.')
                    continue
            if niter > 1000:
                raise RuntimeError('Could not locate the Fermi level!')
            de = dn / dnde
            if self.fixmom:
                de.clip(-self.kT, self.kT, de)
            elif abs(de) > self.kT:
                de *= self.kT / abs(de)
            self.epsF += de
            niter += 1

        if self.nspins == 1:
            magmom = 0.0

        self.niter = niter
        self.magmom = magmom


class FermiDiracFixed(FermiDirac):
    """Occupations with Fermi smearing and fixed Fermi level"""
    def __init__(self, ne, nspins, kT, epsF):
        FermiDirac.__init__(self, ne, nspins, kT)
        self.set_fermi_level(epsF)
        self.niter = 0
    
    def guess_fermi_level(self, kpts):
        pass

    def find_fermi_level(self, kpts):
        magmom = 0.0
        for kpt in kpts:
            sign = 1.0 - 2 * kpt.s
            magmom += sign * np.sum(kpt.f_n)
        magmom = self.band_comm.sum(self.kpt_comm.sum(magmom))
        self.magmom = magmom
