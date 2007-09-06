"""Module defining  ``Eigensolver`` classes."""

from gpaw.eigensolvers.rmm_diis import RMM_DIIS
from gpaw.eigensolvers.cg import CG
from gpaw.eigensolvers.davidson import Davidson
from gpaw.lcao.eigensolver import LCAO


def eigensolver(name, paw):
    """Create eigensolver object."""
    return {'rmm-diis': RMM_DIIS,
            'cg': CG,
            'dav': Davidson,
            'lcao': LCAO}[name](paw)
