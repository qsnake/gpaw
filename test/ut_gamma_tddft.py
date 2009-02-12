
import unittest

#from gpaw.output import Output
from gpaw.tddft import TDDFT

from ut_gamma_dft import UTGammaPointSetup

# -------------------------------------------------------------------

#debug = Output()
#debug.set_text('ut_gamma_tddft.log')

class UTGammaPointTDDFT(UTGammaPointSetup):
    """
    Propagate a gamma point calculation with TDDFT."""

    name = 'ut_gamma_tddft'
    tolerance = 1e-8

    def setUp(self):
        name_old,self.name = self.name,UTGammaPointSetup.name
        UTGammaPointSetup.setUp(self)
        self.name = name_old

        self.assertTrue(self.restarted)

        self.tdcalc = TDDFT(self.restartfile, txt=self.name+'.txt',
                    propagator=self.propagator, solver=self.solver,
                    tolerance=self.tolerance)#, debug=debug) TODO!

        self.time_step = 5.0     # 1 attoseconds = 0.041341 autime
        self.iterations = 10     # 10 x 5 as => 2.067050 autime

    def tearDown(self):
        del self.tdcalc
        UTGammaPointSetup.tearDown(self)

    # =================================

    def test_propagation(self):
        # Propagate without saving the time-dependent dipole moment
        # to a .dat-file, nor periodically dumping a restart file
        self.tdcalc.propagate(self.time_step, self.iterations)

# -------------------------------------------------------------------

class UTGammaPointTDDFT_ECN_CSCG(UTGammaPointTDDFT):
    __doc__ = UTGammaPointTDDFT.__doc__ + """
    Propagator is ECN and solver CSCG."""

    name = UTGammaPointTDDFT.name + '_ecn_cscg'
    propagator = 'ECN'
    solver = 'CSCG'

class UTGammaPointTDDFT_SICN_CSCG(UTGammaPointTDDFT):
    __doc__ = UTGammaPointTDDFT.__doc__ + """
    Propagator is SICN and solver CSCG."""

    name = UTGammaPointTDDFT.name + '_sicn_cscg'
    propagator = 'SICN'
    solver = 'CSCG'

class UTGammaPointTDDFT_SITE_CSCG(UTGammaPointTDDFT):
    __doc__ = UTGammaPointTDDFT.__doc__ + """
    Propagator is SITE and solver CSCG."""

    name = UTGammaPointTDDFT.name + '_site_cscg'
    propagator = 'SITE'
    solver = 'CSCG'

class UTGammaPointTDDFT_SIKE_CSCG(UTGammaPointTDDFT):
    __doc__ = UTGammaPointTDDFT.__doc__ + """
    Propagator is SIKE6 and solver CSCG."""

    name = UTGammaPointTDDFT.name + '_sike_cscg'
    propagator = 'SIKE6'
    solver = 'CSCG'

# -------------------------------------------------------------------

if __name__ == '__main__':

    testrunner = unittest.TextTestRunner(verbosity=2)

    testcases = [UTGammaPointTDDFT_ECN_CSCG, UTGammaPointTDDFT_SICN_CSCG,
                UTGammaPointTDDFT_SITE_CSCG, UTGammaPointTDDFT_SIKE_CSCG]

    for test in testcases:
        info = '\n' + test.__name__ + '\n' + test.__doc__.strip('\n') + '\n'
        testsuite = unittest.defaultTestLoader.loadTestsFromTestCase(test)
        testrunner.stream.writeln(info)
        testrunner.run(testsuite)


