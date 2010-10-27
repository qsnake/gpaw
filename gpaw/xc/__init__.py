from gpaw.xc.libxc import LibXC
from gpaw.xc.lda import LDA
from gpaw.xc.gga import GGA
from gpaw.xc.mgga import MGGA


def XC(kernel, parameters=None):
    if isinstance(kernel, str):
        name = kernel
        if name in ['vdW-DF', 'vdW-DF2']:
            from gpaw.xc.vdw import FFTVDWFunctional
            return FFTVDWFunctional(name)
        elif name in ['EXX', 'PBE0', 'B3LYP']:
            from gpaw.xc.hybrid import HybridXC
            return HybridXC(name)
        elif name == 'BEE1':
            from gpaw.xc.bee import BEE1
            kernel = BEE1(parameters)
        elif name.startswith('GLLB'):
            from gpaw.xc.gllb.nonlocalfunctionalfactory import \
                 NonLocalFunctionalFactory
            xc = NonLocalFunctionalFactory().get_functional_by_name(name)
            xc.print_functional()
            return xc
        else:
            kernel = LibXC(kernel)
    if kernel.type == 'LDA':
        return LDA(kernel)
    elif kernel.type == 'GGA':
        return GGA(kernel)
    else:
        return MGGA(kernel)

