
import unittest
import os.path
import numpy as npy

from ase import Atoms,Atom
from gpaw import GPAW
from gpaw.mixer import Mixer,MixerSum
from gpaw.utilities.dscftools import dscf_find_atoms,dscf_linear_combination

from ut_mixed_dft import UTMixedBCSetup

# -------------------------------------------------------------------

class UTMixedBCSetup_DSCFGroundState(UTMixedBCSetup):
    """
    Setup a DSCF-compatible ground state mixed periodic, multiple k-point calculation with DFT."""

    name = 'ut_mixed_dscf_ground'
    usesymmetry = False

    # Number of additional bands e.g. for DSCF linear expansion
    nextra = 10

    # =================================

    def test_degeneracy(self):

        degeneracies = [(3,4)]

        for kpt in self.calc.wfs.kpt_u:
            for (a,b) in degeneracies:
                self.assertAlmostEqual(kpt.eps_n[a],kpt.eps_n[b],places=3)

    def test_occupancy(self):

        weight_k = self.calc.get_k_point_weights()
        ne_u = npy.array([5*weight_k]).repeat(2,axis=0).flatten()

        for kpt,ne in zip(self.calc.wfs.kpt_u,ne_u):
            self.assertAlmostEqual(sum(kpt.f_n),ne,places=4)

# -------------------------------------------------------------------

class UTMixedBCSetup_DSCFExcitedState(UTMixedBCSetup):
    """
    Setup an excited state gamma point calculation with DSCF."""

    name = 'ut_mixed_dscf_excited'

    # =================================

    def initialize(self):
        # Construct would-be wavefunction from ground state gas phase calculation
        self.calc = GPAW(UTMixedBCSetup_DSCFGroundState.name+'.gpw',
                    #eigensolver='cg',
                    #mixer=MixerSum(nmaxold=5, beta=0.1, weight=100),
                    #mixer=MixerSum(beta=0.1, nmaxold=5, metric='new', weight=100),
                    #mixer=Mixer(beta=0.1, nmaxold=5, metric='new', weight=100),
                    #convergence={'eigenstates': 1e-7, 'bands':nbands},
                    txt=self.name+'.txt')
                    #txt=None)


        self.atoms = self.calc.get_atoms()

        mol = dscf_find_atoms(self.atoms,'C')
        sel_n = [5,6,7] #TODO!!!
        coeff_n = [1,0,0] #[1/2**0.5,1j/2**0.5,0]
        #coeff_n = [1/3**0.5, 1/3**0.5, 1/3**0.5]
        #sel_n = [5]
        #coeff_n = [1.] #TODO!!!

        (P_aui,wf_u,) = dscf_linear_combination(self.calc,mol,sel_n,coeff_n)

        """
        del self.calc, self.atoms

        self.calc = GPAW(UTMixedBCSetup_DSCFGroundState.name+'.gpw',
                    usesymm=False,
                    txt=self.name+'.txt')

        self.atoms = self.calc.get_atoms()
        """

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

        self.assertAlmostEqual(Epot,-17.1042,places=4)

        self.assertTrue(self.calc.initialized)
        self.assertTrue(self.calc.scf.converged)

    def test_degeneracy(self):

        #TODO apparently the px/py-degeneracy is lifted for both spins?
        degeneracies = [(3,4)]

        for kpt in self.calc.wfs.kpt_u:
            for (a,b) in degeneracies:
                self.failIfAlmostEqual(kpt.eps_n[a],kpt.eps_n[b],places=3)

    def test_occupancy(self):

        #TODO
        ne_u = [4., 5.]

        for kpt,ne in zip(self.calc.wfs.kpt_u,ne_u):
            self.assertAlmostEqual(sum(kpt.f_n),ne,places=4)

        ne_ou = [[0., 1.]]

        for o,ne_u in enumerate(ne_ou):
            for kpt,ne in zip(self.calc.wfs.kpt_u,ne_u):
                self.assertAlmostEqual(kpt.ne_o,ne,places=9)
                self.assertAlmostEqual(sum(kpt.c_on[o].real**2),1,places=9)


# -------------------------------------------------------------------

if __name__ == '__main__':
    testrunner = unittest.TextTestRunner(verbosity=2)

    testcases = [UTMixedBCSetup_DSCFGroundState,UTMixedBCSetup_DSCFExcitedState]

    for test in testcases:
        info = '\n' + test.__name__ + '\n' + test.__doc__.strip('\n') + '\n'
        testsuite = unittest.defaultTestLoader.loadTestsFromTestCase(test)
        testrunner.stream.writeln(info)
        testrunner.run(testsuite)

