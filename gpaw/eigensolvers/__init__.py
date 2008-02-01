"""Module defining  ``Eigensolver`` classes."""

from gpaw.eigensolvers.rmm_diis import RMM_DIIS
from gpaw.eigensolvers.rmm_diis2 import RMM_DIIS2
from gpaw.eigensolvers.cg import CG
from gpaw.eigensolvers.davidson import Davidson
from gpaw.lcao.eigensolver import LCAO


def get_eigensolver(name, **kwargs):
    """Create eigensolver object."""
    return {'rmm-diis': RMM_DIIS,
            'rmm-diis2': RMM_DIIS2,
            'cg': CG,
            'dav': Davidson,
            'lcao': LCAO}[name](**kwargs)
