
import unittest
import os.path

from ase import Atoms,Atom
from gpaw import GPAW
from gpaw.mixer import Mixer,MixerSum
from gpaw.utilities.dscftools import dscf_find_atoms,dscf_linear_combination

from ut_gamma_dft import UTGammaPointSetup

# -------------------------------------------------------------------

class UTGammaPointSetup_DSCFGroundState(UTGammaPointSetup):
    """
    Setup a DSCF-compatible ground state gamma point calculation with DFT."""

    name = 'ut_gamma_dscf_ground'

    # Number of additional bands e.g. for DSCF linear expansion
    nextra = 10

    # =================================

    def test_degeneracy(self):

        degeneracies = [(3,4),(5,6)]

        for kpt in self.calc.wfs.kpt_u:
            for (a,b) in degeneracies:
                self.assertAlmostEqual(kpt.eps_n[a],kpt.eps_n[b],places=4)

    def test_occupancy(self):

        ne_u = [5., 5.]

        for kpt,ne in zip(self.calc.wfs.kpt_u,ne_u):
            self.assertAlmostEqual(sum(kpt.f_n),ne,places=4)

# -------------------------------------------------------------------

class UTGammaPointSetup_DSCFExcitedState(UTGammaPointSetup):
    """
    Setup an excited state gamma point calculation with DSCF."""

    name = 'ut_gamma_dscf_excited'

    # =================================

    def initialize(self):
        # Construct would-be wavefunction from ground state gas phase calculation
        self.calc = GPAW(UTGammaPointSetup_DSCFGroundState.name+'.gpw',
                    #eigensolver='cg',
                    #mixer=MixerSum(nmaxold=5, beta=0.1, weight=100),
                    #convergence={'eigenstates': 1e-7, 'bands':nbands},
                    txt=self.name+'.txt')

        self.atoms = self.calc.get_atoms()

        mol = dscf_find_atoms(self.atoms,'C')
        sel_n = [5,6]
        coeff_n = [1.0,0.0]

        (P_aui,wf_u,) = dscf_linear_combination(self.calc,mol,sel_n,coeff_n)

        from gpaw.dscf import AEOrbital,dscf_calculation

        # Setup dSCF calculation to occupy would-be wavefunction
        sigma_star = AEOrbital(self.calc,wf_u,P_aui,molecule=mol)

        # Force one electron (spin down) into the sigma star orbital
        dscf_calculation(self.calc, [[1.0,sigma_star,1]], self.atoms)

    # =================================

    def test_consistency(self):

        self.assertTrue(self.calc.initialized)
        #self.assertEqual(self.calc.scf.converged,self.restarted)

        Epot = self.atoms.get_potential_energy()

        self.assertAlmostEqual(Epot,-17.0975,places=4)

        self.assertTrue(self.calc.initialized)
        self.assertTrue(self.calc.scf.converged)

    def test_degeneracy(self):

        # The px/py-degeneracy is lifted for both spins
        degeneracies = [(3,4),(5,6)]

        for kpt in self.calc.wfs.kpt_u:
            for (a,b) in degeneracies:
                self.failIfAlmostEqual(kpt.eps_n[a],kpt.eps_n[b],places=4)

    def test_occupancy(self):

        ne_u = [4., 5.]

        for kpt,ne in zip(self.calc.wfs.kpt_u,ne_u):
            self.assertAlmostEqual(sum(kpt.f_n),ne,places=4)

        ne_ou = [[0., 1.]]

        for o,ne_u in enumerate(ne_ou):
            for kpt,ne in zip(self.calc.wfs.kpt_u,ne_u):
                self.assertAlmostEqual(kpt.ne_o,ne,places=9)
                self.assertAlmostEqual(sum(abs(kpt.c_on[o])**2),1,places=9)

# -------------------------------------------------------------------

if __name__ == '__main__':
    testrunner = unittest.TextTestRunner(verbosity=2)

    testcases = [UTGammaPointSetup_DSCFGroundState,UTGammaPointSetup_DSCFExcitedState]

    for test in testcases:
        info = '\n' + test.__name__ + '\n' + test.__doc__.strip('\n') + '\n'
        testsuite = unittest.defaultTestLoader.loadTestsFromTestCase(test)
        testrunner.stream.writeln(info)
        testrunner.run(testsuite)

