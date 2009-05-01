
import numpy as np

from gpaw import debug
from gpaw.utilities import is_contiguous
from gpaw.analyse.observers import Observer
from gpaw.transformers import Transformer
from gpaw.tddft import attosec_to_autime, eV_to_aufrequency

# -------------------------------------------------------------------

class DensityFourierTransform(Observer):
    def __init__(self, timestep, frequencies, width=None, interval=1):
        """
        Parameters
        ----------
        timestep: float
            Time step in attoseconds (10^-18 s), e.g., 4.0 or 8.0
        frequencies: NumPy array or list of floats
            Frequencies in eV for Fourier transforms
        width: float or None
            Width of Gaussian envelope in eV, otherwise no envelope
        interval: int
            Number of timesteps between calls (used when attaching)
        """

        Observer.__init__(self, interval)
        self.timestep = interval * timestep * attosec_to_autime # autime
        self.omega_w = np.asarray(frequencies) * eV_to_aufrequency # autime^(-1)

        if width is None:
            self.sigma = None
        else:
            self.sigma = width * eV_to_aufrequency # autime^(-1)

        self.nw = len(self.omega_w)
        self.dtype = complex # np.complex128 really, but hey...
        self.Fnt_wsG = None
        self.Fnt_wsg = None
        self.Ant_sG = None
        self.Ant_sg = None

    def initialize(self, paw, allocate=True):
        self.allocated = False

        assert hasattr(paw, 'time') and hasattr(paw, 'niter'), 'Use TDDFT!'
        self.time = paw.time
        self.niter = paw.niter

        self.world = paw.wfs.world
        self.gd = paw.density.gd
        self.finegd = paw.density.finegd
        self.nspins = paw.density.nspins
        self.stencil = paw.input_parameters.stencils[1] # i.e. r['InterpolationStencil']
        self.interpolator = paw.density.interpolator #TODO is this leak-safe?
        self.cinterpolator = Transformer(self.gd, self.finegd, self.stencil, \
                                        dtype=self.dtype, allocate=False)
        self.phase_cd = np.ones_like(paw.wfs.kpt_u[0].phase_cd) # not cool...

        self.Ant_sG = paw.density.nt_sG.copy() # TODO in allocate instead?

        # Attach to PAW-type object
        paw.attach(self, self.interval, density=paw.density)

        if allocate:
            self.allocate()

    def allocate(self):
        if not self.allocated:
            self.Fnt_wsG = self.gd.zeros((self.nw, self.nspins), \
                                        dtype=self.dtype)
            self.Fnt_wsg = None
            #self.Ant_sG = ...
            self.Ant_sg = None
            self.gamma_w = np.ones(self.nw, dtype=complex) * self.timestep
            self.cinterpolator.allocate()
            self.allocated = True

        if debug:
            assert is_contiguous(self.Fnt_wsG, self.dtype)

    def interpolate_fourier_transform(self):
        if self.Fnt_wsg is None:
            self.Fnt_wsg = self.finegd.empty((self.nw, self.nspins), \
                                            dtype=self.dtype)

        for w in range(self.nw):
            for s in range(self.nspins):
                self.cinterpolator.apply(self.Fnt_wsG[w,s], self.Fnt_wsg[w,s], \
                                        self.phase_cd)

    def interpolate_average(self):
        if self.Ant_sg is None:
            self.Ant_sg = self.finegd.empty(self.nspins, dtype=float)

        for s in range(self.nspins):
            self.interpolator.apply(self.Ant_sG[s], self.Ant_sg[s])
            
    def update(self, density):

        self.time += self.timestep

        f_w = np.exp(1.0j*self.omega_w*self.time)

        if self.sigma is not None:
            f_w *= np.exp(-self.time**2*self.sigma**2/2.0)

        for w, omega in enumerate(self.omega_w):

            # Fnt_wG[N+1] = Fnt_wG[N] + 1/sqrt(pi) * (nt_G[N+1]-avg_nt_G[N]) \
            #     * (f[N+1]*t[n+1] - gamma[N]) * dt[N+1]/(t[N+1]+dt[N+1])
            self.Fnt_wsG[w] += 1/np.pi**0.5 * (density.nt_sG - self.Ant_sG) * \
                (f_w[w]*self.time - self.gamma_w[w]) * self.timestep/(self.time + self.timestep)

        # gamma[N+1] = gamma[N] + f[N+1]*dt[N+1]
        self.gamma_w += f_w * self.timestep

        # Ant_G[N+1] = (t[N+1]*Ant_G[N] + nt_G[N+1]*dt[N+1])/(t[N+1]+dt[N+1])
        self.Ant_sG = (self.time*self.Ant_sG + density.nt_sG*self.timestep) \
            /(self.time + self.timestep)

    def get_fourier_transform(self, frequency=0, spin=0, gridrefinement=1):
        if gridrefinement == 1:
            return self.Fnt_wsG[frequency, spin]
        elif gridrefinement == 2:
            if self.Fnt_wsg is None:
                self.interpolate_fourier_transform()
            return self.Fnt_wsg[frequency, spin]
        else:
            raise NotImplementedError('Arbitrary refinement not implemented')

    def get_average(self, spin=0, gridrefinement=1):
        if gridrefinement == 1:
            return self.Ant_sG[spin]
        elif gridrefinement == 2:
            if self.Ant_sg is None:
                self.interpolate_average()
            return self.Ant_sg[spin]
        else:
            raise NotImplementedError('Arbitrary refinement not implemented')

    def dump(self, filename):
        if debug:
            assert is_contiguous(self.Fnt_wsG, self.dtype)
            assert is_contiguous(self.Ant_sG, float)

        all_Fnt_wsG = self.gd.collect(self.Fnt_wsG)
        all_Ant_sG = self.gd.collect(self.Ant_sG)

        if self.world.rank == 0:
            all_Fnt_wsG.dump(filename)
            all_Ant_sG.dump(filename+'_avg') #TODO XXX a crude hack!!!

    def load(self, filename):
        if self.world.rank == 0:
            all_Fnt_wsG = np.load(filename)
            all_Ant_sG = np.load(filename+'_avg') #TODO XXX a crude hack!!!
        else:
            all_Fnt_wsG = None
            all_Ant_sG = None

        if debug:
            assert all_Fnt_wsG is None or is_contiguous(all_Fnt_wsG, self.dtype)
            assert all_Ant_sG is None or is_contiguous(all_Ant_sG, float)

        if not self.allocated:
            self.allocate()

        self.gd.distribute(all_Fnt_wsG, self.Fnt_wsG)
        self.gd.distribute(all_Ant_sG, self.Ant_sG)


