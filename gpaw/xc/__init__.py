from gpaw.xc.libxc import LibXC
from gpaw.xc.lda import LDA
from gpaw.xc.gga import GGA
from gpaw.xc.mgga import MGGA


def XC(kernel, parameters=None):
    """Create XCFunctional object.

    kernel: XCKernel object or str
        Kernel object or name of functional.
    parameters: ndarray
        Parameters for BEE functional.

    Recognized names are: LDA, PW91, PBE, revPBE, RPBE, BLYP, HCTH407,
    TPSS, M06L, revTPSS, vdW-DF, vdW-DF2, EXX, PBE0, B3LYP, BEE,
    GLLBSC.  One can also use libxc names like this:
    GGA_X_PBE+GGA_C_PBE would be equivalent to PBE, and LDA_X would
    give LDA exchange.  See gpaw.xc.libxc.py for a list of libxc
    names.  """
    
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
        elif name == 'LB94':
            from gpaw.xc.lb94 import LB94
            kernel = LB94()
        else:
            kernel = LibXC(kernel)
    if kernel.type == 'LDA':
        return LDA(kernel)
    elif kernel.type == 'GGA':
        return GGA(kernel)
    else:
        return MGGA(kernel)

