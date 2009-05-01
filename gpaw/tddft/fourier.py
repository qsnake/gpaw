
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
        self.interpolator = Transformer(self.gd, self.finegd, self.stencil, \
                                        dtype=self.dtype, allocate=False)
        self.phase_cd = np.ones_like(paw.wfs.kpt_u[0].phase_cd) # not cool...

        # Attach to PAW-type object
        paw.attach(self, self.interval, density=paw.density)

        if allocate:
            self.allocate()

    def allocate(self):
        if not self.allocated:
            self.Fnt_wsG = self.gd.zeros((self.nw, self.nspins), \
                                        dtype=self.dtype)
            self.Fnt_wsg = None
            self.interpolator.allocate()
            self.allocated = True

        if debug:
            assert is_contiguous(self.Fnt_wsG, self.dtype)

    def interpolate(self):
        if self.Fnt_wsg is None:
            self.Fnt_wsg = self.finegd.empty((self.nw, self.nspins), \
                                            dtype=self.dtype)

        for w in range(self.nw):
            for s in range(self.nspins):
                self.interpolator.apply(self.Fnt_wsG[w,s], self.Fnt_wsg[w,s], \
                                        self.phase_cd)

    def update(self, density):

        self.time += self.timestep

        for w, omega in enumerate(self.omega_w):
            f = np.exp(1.0j*omega*self.time)

            if self.sigma is not None:
                f *= np.exp(-self.time**2*self.sigma**2/2.0)

            self.Fnt_wsG[w] += 1/np.pi**0.5 * density.nt_sG * f * self.timestep

    def get_fourier_transform(self, frequency=0, spin=0, gridrefinement=1):
        if gridrefinement == 1:
            return self.Fnt_wsG[frequency, spin]
        elif gridrefinement == 2:
            if self.Fnt_wsg is None:
                self.interpolate()
            return self.Fnt_wsg[frequency, spin]
        else:
            raise NotImplementedError('Arbitrary refinement not implemented')

    def dump(self, filename):
        if debug:
            assert is_contiguous(self.Fnt_wsG, self.dtype)

        all_nt_wsG = self.gd.collect(self.Fnt_wsG)

        if self.world.rank == 0:
            all_nt_wsG.dump(filename)

    def load(self, filename):
        if self.world.rank == 0:
            all_nt_wsG = np.load(filename)
        else:
            all_nt_wsG = None

        if debug:
            assert all_nt_wsG is None or is_contiguous(all_nt_wsG, self.dtype)

        if not self.allocated:
            self.allocate()

        self.gd.distribute(all_nt_wsG, self.Fnt_wsG)


