from os.path import isfile

from gpaw.atom.generator import Generator, parameters
from gpaw.xc_functional import XCFunctional
from gpaw.setup import Setup
from gpaw import setup_paths

class Lxc_testsetups:
    """Generation and cleaning of setups used in tests.
    """
    def __init__(self):
        setup_paths = ['.'] # read setups from current directory only

    def create(self):
        """First call to self.create generates all the setups,
        subsequent calls generate missing setups
        (those removed by test or manually!)."""
        self.files = generate()

    def clean(self):
        """self.clean removes only setups generated
        by this instance of self!."""
        from os import remove
        for file in self.files:
            if ((file is not None) and isfile(file)):
                remove(file)

def generate():
    files = []
    for symbol in ['Be']: # needed by lxc_exx.py
        for functional in [
            'X-C_PW'
            ]:
            files.append(gen(symbol, functional))
    for symbol in ['Li']: # needed by lxc_spinpol_Li.py
        for functional in [
            'X-C_PW', 'X_PBE-C_PBE', 'X_PBE_R-C_PBE',
            'RPBE', 'PW91', 'oldLDA'
            ]:
            files.append(gen(symbol, functional))
    for symbol in ['N']: # needed by lxc_xcatom.py
        for functional in [
            'X-None', 'X-C_PW', 'X-C_VWN', 'X-C_PZ',
            'X_PBE-C_PBE', 'X_PBE_R-C_PBE',
            'X_B88-C_P86', 'X_B88-C_LYP',
            'X_FT97_A-C_LYP'
            ]:
            files.append(gen(symbol, functional))
    return files

def gen(symbol, xcname):
    value = None
    try:
        xcfunc = XCFunctional(xcname, 1)
        s = Setup(symbol, xcfunc)
    except (IOError, RuntimeError):
        g = Generator(symbol, xcname, scalarrel=True, nofiles=True)
        g.run(exx=True, **parameters[symbol])
        # list generated setups only - tests can handle setups on their own:
        # gen returns None if setup is read - see self.clean
        value = '%s.%s' % (symbol, XCFunctional(xcname).get_name())
    return value
