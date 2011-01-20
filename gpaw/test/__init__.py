import os
import gc
import sys
import time
import signal
import traceback

import numpy as np

from gpaw.atom.generator import Generator
from gpaw.atom.configurations import parameters
from gpaw.utilities import devnull
from gpaw import setup_paths
from gpaw import mpi
import gpaw


def equal(x, y, tolerance=0, fail=True, msg=''):
    """Compare x and y."""

    if not np.isfinite(x - y) or abs(x - y) > tolerance:
        msg = (msg + '%.9g != %.9g (error: |%.9g| > %.9g)' %
               (x, y, x - y, tolerance))
        if fail:
            raise AssertionError(msg)
        else:
            sys.stderr.write('WARNING: %s\n' % msg)

def gen(symbol, exx=False, name=None, **kwargs):
    if mpi.rank == 0:
        if 'scalarrel' not in kwargs:
            kwargs['scalarrel'] = True
        g = Generator(symbol, **kwargs)
        g.run(exx=exx, name=name, use_restart_file=False, **parameters[symbol])
    mpi.world.barrier()
    if setup_paths[0] != '.':
        setup_paths.insert(0, '.')

def wrap_pylab(names=[]):
    """Use Agg backend and prevent windows from popping up."""
    import matplotlib
    matplotlib.use('Agg')
    import pylab
    def show(names=names):
        if names:
            name = names.pop(0)
        else:
            name = 'fig.png'
        pylab.savefig(name)
    pylab.show = show


tests = [
    'ase3k_version.py',
    'numpy_core_multiarray_dot.py',
    'numpy_zdotc_graphite.py',
    'gemm_complex.py',
    'lapack.py',
    'mpicomm.py',
    'eigh.py',
    'xc.py',
    'gradient.py',
    'pbe_pw91.py',
    'cg2.py',
    'd2Excdn2.py',
    'dot.py',
    'blas.py',
    'gp2.py',
    'non_periodic.py',
    'erf.py',
    'lf.py',
    'lxc_fxc.py',
    'Gauss.py',
    'cluster.py',
    'derivatives.py',
    'second_derivative.py',
    'integral4.py',
    'transformations.py',
    'nabla.py',
    'aeatom.py',
    'pbc.py',
    'atoms_too_close.py',
    'poisson.py',
    'XC2.py',
    'multipoletest.py',
    'proton.py',
    'vdwradii.py',
    'parallel/ut_parallel.py',
    'parallel/compare.py',
    'ase3k.py',
    'laplace.py',
    'ds_beta.py',
    'gauss_wave.py',
    'planewavebasis.py',
    'coulomb.py',
    'timing.py',
    'lcao_density.py',
    'gauss_func.py',
    'ah.py',
    'ylexpand.py',
    'wfs_io.py',
    'wfs_auto.py',
    'xcatom.py',
    'parallel/overlap.py',
    'symmetry.py',
    'pes.py',
    'elf.py',
    'lebedev.py',
    'usesymm.py',
#    'usesymm2.py',
    'eed.py',
    'partitioning.py',
    'mixer.py',
    'broydenmixer.py',
    'restart.py',
    'mgga_restart.py',
    'gga_atom.py',
    'bee1.py',
    'refine.py',
    'revPBE.py',
    'lcao_largecellforce.py',
    'lcao_h2o.py',
    'spin_contamination.py',
    'lrtddft2.py',
    'stdout.py',
    'nonselfconsistentLDA.py',
    'nonselfconsistent.py',
    'ewald.py',
    'harmonic.py',
    'ut_csh.py',
    'ut_rsh.py',
    'spinpol.py',
    'kptpar.py',
    'plt.py',
    'parallel/hamiltonian.py',
    'restart2.py',
    'hydrogen.py',
    'H_force.py',
    'Cl_minus.py',
    'blocked_rmm_diis.py',
    'degeneracy.py',
    'h2o_xas.py',
    'fermilevel.py',
    'al_chain.py',
    'bulk.py',
    'si.py',
    'gemm.py',
    'gemv.py',
    'asewannier.py',
    'davidson.py',
    'cg.py',
    'h2o_xas_recursion.py',
    'lrtddft.py',
    'spectrum.py',
    'lcao_bsse.py',
    'lcao_force.py',
    'parallel/lcao_hamiltonian.py',
    'parallel/lcao_parallel.py',
    'parallel/lcao_parallel_kpt.py',
    'parallel/fd_parallel.py',
    'parallel/fd_parallel_kpt.py',
    'gllbatomic.py', 
    'ne_gllb.py',
    'ne_disc.py', 
    'wannier_ethylene.py',
    'CH4.py',
    'neb.py',
    'diamond_absorption.py',
    'aluminum_EELS.py',
    'dump_chi0.py',
    'au02_absorption.py',
    'bse_aluminum.py',
    'bse_diamond.py',
    'bse_vs_lrtddft.py',
    'hgh_h2o.py',
    'apmb.py',
    'relax.py',
    'muffintinpot.py',
    'fixmom.py',
    'fermisplit.py',
    'be_nltd_ip.py',
    'lcao_bulk.py',
    'jstm.py',
    'simple_stm.py',
    'guc_force.py',
    'td_na2.py',
    'ldos.py',
    'exx_coarse.py',
    '2Al.py',
    'lxc_xcatom.py',
    'aedensity.py',
    'si_primitive.py',
    'restart_band_structure.py',
    'IP_oxygen.py',
    'atomize.py',
    'dipole.py',
    'Hubbard_U.py',
    'Hubbard_U_Zn.py',
    'revPBE_Li.py',
    'si_xas.py',
    'tpss.py',
    'nsc_MGGA.py',
    '8Si.py',
    'dscf_lcao.py',
    'coreeig.py',
    'Cu.py',
    'diamond_gllb.py',
    'exx.py',
    'h2o_dks.py',
    'nscfsic.py',
    'scfsic_h2.py',
    'scfsic_n2.py',
    'lb94.py',
    'aluminum_EELS_lcao.py',
    'vdw/quick.py',
    'vdw/potential.py',
    'vdw/quick_spin.py',
    'vdw/ar2.py',
    'fd2lcao_restart.py',
#    'eigh_perf.py', # Requires LAPACK 3.2.1 or later
    'parallel/parallel_eigh.py',
    'parallel/ut_hsops.py',
    'parallel/ut_hsblacs.py',
    'parallel/ut_invops.py',
    'parallel/ut_kptops.py',
    'parallel/ut_redist.py',
    'parallel/pblas.py',
    'parallel/blacsdist.py',
    'parallel/scalapack.py',
    'parallel/realspace_blacs.py',
    'parallel/lcao_projections.py',
    #'dscf_forces.py',
    'lrtddft3.py',
    'AA_exx_enthalpy.py',
    'transport.py',
    'constant_electric_field.py',
    'stark_shift.py',
    'aluminum_testcell.py',
    ]

exclude = []
if mpi.size > 1:
    exclude += ['pes.py',
                'nscfsic.py',
                'coreeig.py',
                'asewannier.py',
                'wannier_ethylene.py',
                'muffintinpot.py',
                'stark_shift.py']

if mpi.size > 2:
    exclude += ['neb.py']

if mpi.size < 4:
    exclude += ['parallel/pblas.py',
                'parallel/scalapack.py',
                'parallel/realspace_blacs.py',
                'AA_exx_enthalpy.py',
                'bse_aluminum.py',
                'bse_diamond.py',
                'bse_vs_lrtddft.py']

if mpi.size != 4:
    exclude += ['parallel/lcao_parallel.py']
    exclude += ['parallel/fd_parallel.py']

if mpi.size == 8:
    exclude += ['transport.py']

if mpi.size != 8:
    exclude += ['parallel/lcao_parallel_kpt.py']
    exclude += ['parallel/fd_parallel_kpt.py']

try:
    import scipy
except ImportError:
    exclude += ['diamond_absorption.py',
                'aluminum_EELS.py',
                'aluminum_EELS_lcao.py',
                'aluminum_testcell.py',
                'au02_absorption.py',
                'bse_aluminum.py',
                'bse_diamond.py',
                'bse_vs_lrtddft.py',
                'aeatom.py']

for test in exclude:
    if test in tests:
        tests.remove(test)


class TestRunner:
    def __init__(self, tests, stream=sys.__stdout__, jobs=1):
        if mpi.size > 1:
            assert jobs == 1
        self.jobs = jobs
        self.tests = tests
        self.failed = []
        self.garbage = []
        if mpi.rank == 0:
            self.log = stream
        else:
            self.log = devnull
        self.n = max([len(test) for test in tests])

    def run(self):
        self.log.write('=' * 77 + '\n')
        sys.stdout = devnull
        ntests = len(self.tests)
        t0 = time.time()
        if self.jobs == 1:
            self.run_single()
        else:
            # Run several processes using fork:
            self.run_forked()

        sys.stdout = sys.__stdout__
        self.log.write('=' * 77 + '\n')
        self.log.write('Ran %d tests out of %d in %.1f seconds\n' %
                       (ntests - len(self.tests), ntests, time.time() - t0))
        if self.failed:
            self.log.write('Tests failed: %d\n' % len(self.failed))
        else:
            self.log.write('All tests passed!\n')
        self.log.write('=' * 77 + '\n')
        return self.failed

    def run_single(self):
        while self.tests:
            test = self.tests.pop(0)
            try:
                self.run_one(test)
            except KeyboardInterrupt:
                self.tests.append(test)
                break

    def run_forked(self):
        j = 0
        pids = {}
        while self.tests or j > 0:
            if self.tests and j < self.jobs:
                test = self.tests.pop(0)
                pid = os.fork()
                if pid == 0:
                    exitcode = self.run_one(test)
                    os._exit(exitcode)
                else:
                    j += 1
                    pids[pid] = test
            else:
                try:
                    while True:
                        pid, exitcode = os.wait()
                        if pid in pids:
                            break
                except KeyboardInterrupt:
                    for pid, test in pids.items():
                        os.kill(pid, signal.SIGHUP)
                        self.write_result(test, 'STOPPED', time.time())
                        self.tests.append(test)
                    break
                if exitcode:
                    self.failed.append(pids[pid])
                del pids[pid]
                j -= 1

    def run_one(self, test):
        if self.jobs == 1:
            self.log.write('%*s' % (-self.n, test))
            self.log.flush()

        t0 = time.time()
        filename = gpaw.__path__[0] + '/test/' + test

        try:
            loc = {}
            execfile(filename, loc)
            loc.clear()
            del loc
            self.check_garbage()
        except KeyboardInterrupt:
            self.write_result(test, 'STOPPED', t0)
            raise
        except:
            failed = True
        else:
            failed = False

        mpi.ibarrier(timeout=60.0) # guard against parallel hangs

        me = np.array(failed)
        everybody = np.empty(mpi.size, bool)
        mpi.world.all_gather(me, everybody)
        failed = everybody.any()

        if failed:
            self.fail(test, np.argwhere(everybody).ravel(), t0)
        else:
            self.write_result(test, 'OK', t0)

        return failed

    def check_garbage(self):
        gc.collect()
        n = len(gc.garbage)
        self.garbage += gc.garbage
        del gc.garbage[:]
        assert n == 0, ('Leak: Uncollectable garbage (%d object%s) %s' %
                        (n, 's'[:n > 1], self.garbage))

    def fail(self, test, ranks, t0):
        if mpi.rank in ranks:
            if sys.version_info >= (2, 4, 0, 'final', 0):
                tb = traceback.format_exc()
            else:  # Python 2.3! XXX
                tb = ''
                traceback.print_exc()
        else:
            tb = ''
        if mpi.size == 1:
            text = 'FAILED!\n%s\n%s%s' % ('#' * 77, tb, '#' * 77)
            self.write_result(test, text, t0)
        else:
            tbs = {tb: [0]}
            for r in range(1, mpi.size):
                if mpi.rank == r:
                    mpi.send_string(tb, 0)
                elif mpi.rank == 0:
                    tb = mpi.receive_string(r)
                    if tb in tbs:
                        tbs[tb].append(r)
                    else:
                        tbs[tb] = [r]
            if mpi.rank == 0:
                text = ('FAILED! (rank %s)\n%s' %
                        (','.join([str(r) for r in ranks]), '#' * 77))
                for tb, ranks in tbs.items():
                    if tb:
                        text += ('\nRANK %s:\n' %
                                 ','.join([str(r) for r in ranks]))
                        text += '%s%s' % (tb, '#' * 77)
                self.write_result(test, text, t0)

        self.failed.append(test)

    def write_result(self, test, text, t0):
        t = time.time() - t0
        if self.jobs > 1:
            self.log.write('%*s' % (-self.n, test))
        self.log.write('%10.3f  %s\n' % (t, text))


if __name__ == '__main__':
    TestRunner(tests).run()
