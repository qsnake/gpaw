"""Module defining  ``Eigensolver`` classes."""

from gpaw.eigensolvers.rmm_diis import RMM_DIIS
from gpaw.eigensolvers.rmm_diis2 import RMM_DIIS2
from gpaw.eigensolvers.cg import CG
from gpaw.eigensolvers.davidson import Davidson
from gpaw.lcao.eigensolver import LCAO


def get_eigensolver(name, mode, convergence=None):
    """Create eigensolver object."""
    if name is None:
        name = {'fd': 'rmm-diis', 'lcao': 'lcao'}[mode]
    if isinstance(name, str):
        eigensolver = {'rmm-diis':  RMM_DIIS,
                       'rmm-diis2': RMM_DIIS2,
                       'cg':        CG,
                       'dav':       Davidson,
                       'lcao':      LCAO
                       }[name]()
    else:
        eigensolver = name
    
    if isinstance(eigensolver, CG):
        eigensolver.tolerance = convergence['eigenstates']

    assert isinstance(eigensolver, LCAO) == (mode == 'lcao')

    return eigensolver
