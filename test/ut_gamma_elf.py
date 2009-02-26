
import unittest
import os.path
import numpy as npy

from ase import write
from gpaw import GPAW
from gpaw.elf import ELF

from ut_gamma_dft import UTGammaPointSetup

# -------------------------------------------------------------------

class UTGammaPointSetup_ELF(UTGammaPointSetup):
    """
    Calculate ELF for a simple gamma point calculation with DFT."""

    name = 'ut_gamma_elf'

    def setUp(self):
        name_old,self.name = self.name,UTGammaPointSetup.name
        UTGammaPointSetup.setUp(self)
        self.name = name_old

        self.assertTrue(self.restarted)

        # Construct ELF object and update it
        self.elf = ELF(self.calc)

    def tearDown(self):
        del self.elf
        UTGammaPointSetup.tearDown(self)

    # =================================

    def test_auto_update_safety(self):

        gd = self.calc.gd
        ne_pre = gd.integrate(self.calc.density.nt_sG).sum()

        self.elf.update(self.calc.wfs)

        ne_post = gd.integrate(self.calc.density.nt_sG).sum()

        self.assertEqual(ne_pre,ne_post)

    def test_consistency(self):

        self.assertTrue(self.calc.initialized)
        self.assertTrue(self.calc.scf.converged)

        self.elf.update(self.calc.wfs)
        elf_g = self.elf.get_electronic_localization_function(spin=0,
                                                         gridrefinement=2)
        iso_elf = 0.5**0.5
        nt_g = self.calc.density.nt_sg[0]
        iso_ne = npy.sum(nt_g[elf_g>iso_elf])*self.calc.finegd.dv

        self.assertAlmostEqual(iso_ne,1.5,places=3)

        write(self.name+'.cube', self.atoms, data=elf_g)


# -------------------------------------------------------------------

if __name__ == '__main__':
    testrunner = unittest.TextTestRunner(verbosity=2)

    testcases = [UTGammaPointSetup_ELF]

    for test in testcases:
        info = '\n' + test.__name__ + '\n' + test.__doc__.strip('\n') + '\n'
        testsuite = unittest.defaultTestLoader.loadTestsFromTestCase(test)
        testrunner.stream.writeln(info)
        testrunner.run(testsuite)

