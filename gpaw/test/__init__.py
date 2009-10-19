import os
import gc
import sys
import time
import signal
import traceback

import numpy as np

from gpaw.atom.generator import Generator, parameters
from gpaw.utilities import devnull
from gpaw import setup_paths
import gpaw.mpi as mpi
import gpaw


def equal(x, y, tolerance=0, fail=True):
    if abs(x - y) > tolerance:
        msg = '%g != %g (error: %g > %g)' % (x, y, abs(x - y), tolerance)
        if fail:
            raise AssertionError(msg)
        else:
            sys.stderr.write('WARNING: %s\n' % msg)

def gen(symbol, name=None, **kwargs):
    if mpi.rank == 0:
        if 'scalarrel' not in kwargs:
            kwargs['scalarrel'] = True
        g = Generator(symbol, **kwargs)
        g.run(name=name, **parameters[symbol])
    mpi.world.barrier()
    if '.' not in setup_paths:
        setup_paths.append('.')

tests = [
    'ase3k_version.py',
    'lapack.py',
    'eigh.py',
    'setups.py',
    'xc.py',
    'xcfunc.py',
    'gradient.py',
    'pbe_pw91.py',
    'cg2.py',
    'd2Excdn2.py',
    'dot.py',
    'blas.py',
    'gp2.py',
    'non_periodic.py',
    'lf.py',
    'lxc_xc.py',
    'lxc_fxc.py',
    'Gauss.py',
    'cluster.py',
    'derivatives.py',
    'integral4.py',
    'transformations.py',
    'pbc.py',
    'poisson.py',
    'XC2.py',
    'XC2Spin.py',
    'multipoletest.py',
    'proton.py',
    'parallel/ut_parallel.py',
    'parallel/compare.py',
    'coulomb.py',
    'ase3k.py',
    'eed.py',
    'timing.py',
    'gauss_wave.py',
    'gauss_func.py',
    'xcatom.py',
    'kptpar.py',
    'parallel/overlap.py',
    'symmetry.py',
    'pes.py',
    'usesymm.py',
    'mixer.py',
    'mixer_broydn.py',
    'ylexpand.py',
    'wfs_io.py',
    'restart.py',
    'gga_atom.py',
    'nonselfconsistentLDA.py',
    'bee1.py',
    'refine.py',
    'revPBE.py',
    'jstm.py',
    'lcao_largecellforce.py',
    'lcao_h2o.py',
    'lrtddft2.py',
    'nonselfconsistent.py',
    'stdout.py',
    'ewald.py',
    'spinpol.py',
    'plt.py',
    'parallel/hamiltonian.py',
    'bulk.py',
    'restart2.py',
    'hydrogen.py',
    'aedensity.py',
    'H_force.py',
    'CL_minus.py',
    'gemm.py',
    'gemv.py',
    'fermilevel.py',
    'degeneracy.py',
    'h2o_xas.py',
    'si.py',
    'simple_stm.py',
    'asewannier.py',
    'vdw/quick.py',
    'vdw/potential.py',
    'vdw/quick_spin.py',
    'lxc_xcatom.py',
    'davidson.py',
    'cg.py',
    'h2o_xas_recursion.py',
    'atomize.py',
    'Hubbard_U.py',    
    'lrtddft.py',
    'lcao_force.py',
    'parallel/lcao_hamiltonian.py',
    'wannier_ethylene.py',
    'CH4.py',
    'neb.py',
    'hgh_h2o.py',
    'apmb.py',
    'relax.py',
    'muffintinpot.py',
    'restart_band_structure.py',
    'ldos.py',
    'lcao_bulk.py',
    'revPBE_Li.py',
    'fixmom.py',
    'xctest.py',
    'td_na2.py',
    'exx_coarse.py',
    'lcao_bsse.py',
    '2Al.py',
    'si_primitive.py',
    'si_xas.py',
    'tpss.py',
    'atomize.py',
    'nsc_MGGA.py',
    '8Si.py',
    'coreeig.py',
    'transport.py',
    'Cu.py',
    'IP_oxygen.py',
    'exx.py',
    'dscf_CO.py',
    'h2o_dks.py',
    'H2Al110.py',
    'nscfsic.py',
    'ltt.py',
    'vdw/ar2.py',
    'mgga_restart.py',
    'fd2lcao_restart.py',
    'parallel/ut_hsops.py',
    'parallel/ut_invops.py',
    'parallel/ut_kptops.py',
    'parallel/scalapack.py',
    'parallel/scalapack2.py',
    'parallel/lcao_projections.py',
    'parallel/n2.py',
    ]

exclude = []
if mpi.size > 1:
    exclude += ['pes.py',
                'nscfsic.py',
                'coreeig.py',
                'asewannier.py',
                'wannier_ethylene.py',
                'muffintinpot.py']

if mpi.size > 2:
    exclude += ['neb.py']

if mpi.size != 4:
    exclude += ['parallel/scalapack.py', 'parallel/scalapack2.py']

if mpi.size != 4 or not gpaw.debug:
    exclude += ['parallel/n2.py']

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
            self.log.write('%-30s' % test)
            self.log.flush()
            
        t0 = time.time()
        filename = gpaw.__path__[0] + '/test/' + test
        
        try:
            execfile(filename, {})
            self.check_garbage()
        except KeyboardInterrupt:
            self.write_result(test, 'STOPPED', t0)
            raise
        except:
            failed = True
        else:
            failed = False

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
            tb = traceback.format_exc()
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
            self.log.write('%-30s' % test)
        self.log.write('%9.3f  %s\n' % (t, text))
