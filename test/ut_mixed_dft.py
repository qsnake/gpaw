
import unittest
import os.path

from ase import Atoms,Atom
from gpaw import GPAW
from gpaw.mixer import Mixer,MixerSum

#from numpy import ndarray,zeros,any,all,abs

# -------------------------------------------------------------------

class UTMixedBCSetup(unittest.TestCase):
    """
    Setup a mixed periodic, multiple k-point calculation with DFT."""

    name = 'ut_mixed_dft'
    usesymmetry = True

    # Number of additional bands e.g. for DSCF linear expansion
    nextra = 0

    def setUp(self):
        self.restartfile = self.name+'.gpw'
        self.restarted = os.path.isfile(self.restartfile)

        if self.restarted:
            self.calc = GPAW(self.restartfile,txt=None)
            self.atoms = self.calc.get_atoms()
        else:
            self.initialize()

    def tearDown(self):
        if not self.restarted:
            self.calc.write(self.restartfile, mode='all')

        del self.atoms
        del self.calc

    # =================================

    def initialize(self):
        # Bond lengths between H-C and C-C for ethyne (acetylene) cf.
        # CRC Handbook of Chemistry and Physics, 87th ed., p. 9-28
        dhc = 1.060
        dcc = 1.203

        self.atoms = Atoms([Atom('H', (0, 0, 0)),
                    Atom('C', (dhc, 0, 0)),
                    Atom('C', (dhc+dcc, 0, 0)),
                    Atom('H', (2*dhc+dcc, 0, 0))], pbc=(0,1,1))

        self.atoms.center(vacuum=4.0)

        # Number of occupied and unoccupied bands to converge
        nbands = int(10/2.0)+3

        self.calc = GPAW(h=0.2,
                    nbands=nbands+self.nextra,
                    kpts=(1,4,4),
                    xc='RPBE',
                    spinpol=True,
                    eigensolver='cg',
                    mixer=MixerSum(nmaxold=5, beta=0.1, weight=100),
                    convergence={'eigenstates': 1e-9, 'bands':nbands},
                    #width=0.1, #TODO might help convergence?
                    usesymm=self.usesymmetry,
                    txt=self.name+'.txt')

        self.atoms.set_calculator(self.calc)

    # =================================

    def test_consistency(self):

        self.assertEqual(self.calc.initialized,self.restarted)
        #self.assertEqual(self.calc.scf.converged,self.restarted)

        Epot = self.atoms.get_potential_energy()

        self.assertAlmostEqual(Epot,-22.8109,places=4)

        self.assertTrue(self.calc.initialized)
        self.assertTrue(self.calc.scf.converged)


# -------------------------------------------------------------------

if __name__ == '__main__':
    testrunner = unittest.TextTestRunner(verbosity=2)

    testcases = [UTMixedBCSetup]

    for test in testcases:
        info = '\n' + test.__name__ + '\n' + test.__doc__.strip('\n') + '\n'
        testsuite = unittest.defaultTestLoader.loadTestsFromTestCase(test)
        testrunner.stream.writeln(info)
        testrunner.run(testsuite)

