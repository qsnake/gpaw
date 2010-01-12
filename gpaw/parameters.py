import numpy as np
from ase.units import Hartree

from gpaw.poisson import PoissonSolver, FFTPoissonSolver
from gpaw.occupations import FermiDirac

class InputParameters(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, [
            ('h',               None),  # Angstrom
            ('xc',              'LDA'),
            ('gpts',            None),
            ('kpts',            [(0, 0, 0)]),
            ('lmax',            2),
            ('charge',          0),
            ('fixmom',          False),      # don't use this
            ('nbands',          None),
            ('setups',          'paw'),
            ('basis',           {}),
            ('width',           None),  # eV, don't use this
            ('occupations',     None),
            ('spinpol',         None),
            ('usesymm',         True),
            ('stencils',        (3, 3)),
            ('fixdensity',      False),
            ('mixer',           None),
            ('txt',             '-'),
            ('hund',            False),
            ('random',          False),
            ('maxiter',         120),
            ('parallel',        dict(domain=None,
                                     band=None,
                                     stridebands=False,
                                     scalapack=None)),
            ('parsize',         None),
            ('parsize_bands',   None),
            ('parstride_bands', False),
            ('external',        None),  # eV
            ('verbose',         0),
            ('eigensolver',     None),
            ('poissonsolver',   None),
            ('communicator' ,   None),
            ('idiotproof'   ,   True),
            ('mode',            'fd'),
            ('convergence',     {'energy':      0.001,  # eV / atom
                                 'density':     1.0e-4,
                                 'eigenstates': 1.0e-9,
                                 'bands':       'occupied'}),
            ])
        dict.update(self, kwargs)

    def __repr__(self):
        dictrepr = dict.__repr__(self)
        repr = 'InputParameters(**%s)' % dictrepr
        return repr
    
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        assert key in self
        self[key] = value

    def update(self, parameters):
        assert isinstance(parameters, dict)

        for key, value in parameters.items():
            assert key in self

            haschanged = (self[key] != value)

            if isinstance(haschanged, np.ndarray):
                haschanged = haschanged.any()

            #if haschanged:
            #    self.notify(key)

        dict.update(self, parameters)

    def read(self, reader):
        """Read state from file."""

        if isinstance(reader, str):
            reader = gpaw.io.open(reader, 'r')

        r = reader

        version = r['version']
        
        assert version >= 0.3
    
        self.xc = r['XCFunctional']
        self.nbands = r.dimension('nbands')
        self.spinpol = (r.dimension('nspins') == 2)
        self.kpts = r.get('BZKPoints')
        self.usesymm = r['UseSymmetry']
        try:
            self.basis = r['BasisSet']
        except KeyError:
            pass
        self.gpts = ((r.dimension('ngptsx') + 1) // 2 * 2,
                     (r.dimension('ngptsy') + 1) // 2 * 2,
                     (r.dimension('ngptsz') + 1) // 2 * 2)
        self.lmax = r['MaximumAngularMomentum']
        self.setups = r['SetupTypes']
        self.fixdensity = r['FixDensity']
        if version <= 0.4:
            # Old version: XXX
            print('# Warning: Reading old version 0.3/0.4 restart files ' +
                  'will be disabled some day in the future!')
            self.convergence['eigenstates'] = r['Tolerance']
        else:
            self.convergence = {'density': r['DensityConvergenceCriterion'],
                                'energy':
                                r['EnergyConvergenceCriterion'] * Hartree,
                                'eigenstates':
                                r['EigenstatesConvergenceCriterion'],
                                'bands': r['NumberOfBandsToConverge']}
            if version <= 0.6:
                mixer = 'Mixer'
                weight = r['MixMetric']
            elif version <= 0.7:
                mixer = r['MixClass']
                weight = r['MixWeight']
                metric = r['MixMetric']
                if metric is None:
                    weight = 1.0
            else:
                mixer = r['MixClass']
                weight = r['MixWeight']

            if mixer == 'Mixer':
                from gpaw.mixer import Mixer
            elif mixer == 'MixerSum':
                from gpaw.mixer import MixerSum as Mixer
            elif mixer == 'DummyMixer':
                from gpaw.mixer import DummyMixer as Mixer
            else:
                Mixer = None

            if Mixer is None:
                self.mixer = None
            else:
                self.mixer = Mixer(r['MixBeta'], r['MixOld'], weight)
            
        if version == 0.3:
            # Old version: XXX
            print('# Warning: Reading old version 0.3 restart files is ' +
                  'dangerous and will be disabled some day in the future!')
            self.stencils = (2, 3)
            self.charge = 0.0
            fixmom = False
        else:
            self.stencils = (r['KohnShamStencil'],
                             r['InterpolationStencil'])
            if r['PoissonStencil'] == 999:
                self.poissonsolver = FFTPoissonSolver()
            else:
                self.poissonsolver = PoissonSolver(nn=r['PoissonStencil'])
            self.charge = r['Charge']
            fixmom = r['FixMagneticMoment']

        self.occupations = FermiDirac(r['FermiWidth'] * Hartree,
                                      fixmagmom=fixmom)

        try:
            self.mode = r['Mode']
        except KeyError:
            self.mode = 'fd'
