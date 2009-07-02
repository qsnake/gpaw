"""Run longer test jobs in parallel on Niflheim."""

import os
import sys
import time
import glob

from gpaw.atom.generator import parameters as setup_parameters

class Job:
    def __init__(self, path, tmax=20, ncpu=4, deps=None, arg=''):
        self.dir = os.path.dirname(path)
        self.name = os.path.basename(path)
        self.id = self.name + arg
        self.prefix = path + arg
        self.tmax = tmax
        self.ncpu = ncpu
        if deps is None:
            deps = []
        self.deps = deps
        self.arg = arg
        self.status = 'waiting'

# Exercises:
path = '../../doc/exercises/'
jobs = [
    Job(path + 'neb/neb1'),
    Job(path + 'aluminium/Al_fcc'),
    Job(path + 'aluminium/Al_fcc_convergence'),
    Job(path + 'surface/work_function', 20, deps=['testAl100']),
    Job(path + 'surface/testAl100'),
    Job(path + 'diffusion/initial'),
    Job(path + 'diffusion/densitydiff', 20, deps=['solution']),
    Job(path + 'diffusion/solution'),
    Job(path + 'vibrations/H2O_vib', 20, deps=['h2o']),
    Job(path + 'vibrations/h2o'),
    Job(path + 'band_structure/Na_band'),
    Job(path + 'band_structure/plot_band', 20, deps=['Na_band']),
    Job(path + 'wannier/wannier-si'),
    Job(path + 'wannier/wannier-benzene', 20, deps=['benzene']),
    Job(path + 'wannier/benzene'),
    Job(path + 'lrtddft/ground_state'),
    Job(path + 'transport/pt_h2_tb_transport'),
    Job(path + 'transport/pt_h2_lcao', 20, deps=['makebasis']),
    Job(path + 'transport/pt_h2_lcao_transport', 20, deps=['makebasis']),
    Job(path + 'transport/makebasis', 5, 1),
    Job(path + 'dos/testdos', 20, 1,
        deps=['ferro', 'anti', 'non', 'CO', 'si', 'Al_fcc']),
    Job(path + 'stm/HAl100'),
    Job(path + 'wannier/si'),
    Job(path + 'wavefunctions/CO'),
    Job(path + 'iron/PBE', 20, deps=['ferro', 'anti', 'non']),
    Job(path + 'iron/ferro'),
    Job(path + 'iron/anti'),
    Job(path + 'iron/non'),
    Job(path + 'stm/teststm', 20, 1, deps=['HAl100']),
    ]

jobs = []
#setup_parameters = ['H', 'Li']
for symbol in setup_parameters:
    jobs.append(Job('../../doc/setups/make_setup_pages_data',
                    tmax=60, ncpu=1, arg=symbol))
jobs.append(Job('../../doc/setups/make_setup_pages_data', tmax=60, ncpu=1, 
                deps=['make_setup_pages_data' + symbol
                      for symbol in setup_parameters]))

jobs = [
    Job('Ru001/ruslab', tmax=5*60, ncpu=4),
    Job('Ru001/ruslab', tmax=5*60, ncpu=8, arg='H'),
    Job('Ru001/ruslab', tmax=5*60, ncpu=8, arg='N'),
    Job('Ru001/ruslab', tmax=5*60, ncpu=12, arg='O'),
    Job('Ru001/molecules', tmax=20, ncpu=4),
#   Job('Ru001/result', ncpu=1, deps=['ruslab', 'ruslabN', 'ruslabO', 'NO']),
#    Job('COAu38/Au038to', 10),
#    Job('O2Pt/o2pt', 40),
#    Job('../vdw/interaction', 60, deps=['dimers']),
#    Job('../vdw/dimers', 60),
    ]

class Jobs:
    def __init__(self, log=sys.stdout):
        """Test jobs."""
        self.jobs = {}
        self.ids = []
        if isinstance(log, str):
            self.fd = open(log, 'w')
        else:
            self.fd = log
        
    def log(self, *args):
        self.fd.write(' '.join(args) + '\n')
        self.fd.flush()
        
    def add(self, jobs):
        for job in jobs:
            assert job.id not in self.jobs
            self.jobs[job.id] = job
            self.ids.append(job.id)
                              
    def run(self):
        jobs = self.jobs
        while True:
            done = True
            for id, job in jobs.items():
                if job.status == 'waiting':
                    done = False
                    ready = True
                    for dep in job.deps:
                        if jobs[dep].status != 'done':
                            ready = False
                            break
                    if ready:
                        self.start(job)
                elif job.status == 'running':
                    done = False

            if done:
                return

            time.sleep(60.0)

            for id, job in jobs.items():
                filename = job.prefix + '.done'
                if job.status == 'running' and os.path.isfile(filename):
                    code = int(open(filename).readlines()[-1])
                    if code == 0:
                        job.status = 'done'
                        self.log(id, 'done.')
                    else:
                        job.status = 'failed'
                        self.log('%s exited with errorcode: %d' % (id, code))
                        self.fail(id)
                filename = job.prefix + '.start'
                if job.status == 'running' and os.path.isfile(filename):
                    t0 = float(open(filename).readline())
                    if time.time() - t0 > job.tmax * 60:
                        job.status = 'failed'
                        self.log('%s timed out!' % id)
                        self.fail(id)

    def fail(self, failed_id):
        """Recursively disable jobs depending on failed job."""
        for id, job in self.jobs.items():
            if failed_id in job.deps:
                job.status = 'disabled'
                self.log('Disabling %s' % id)
                self.fail(id)

    def print_results(self):
        for id in self.ids:
            job = self.jobs[id]
            status = job.status
            filename = job.prefix + '.done'
            if status != 'disabled' and os.path.isfile(filename):
                t = (float(open(filename).readline()) -
                     float(open(filename[:-4] + 'start').readline()))
                t = '%8.1f' % t
            else:
                t = '        '
            self.log('%20s %s %s' % (id, t, status))

    def start(self, job):
        try:
            os.remove(job.prefix + '.done')
        except OSError:
            pass

        gpaw_python = (self.gpawdir +
                       '/gpaw/build/bin.linux-x86_64-2.3/gpaw-python')
        cmd = (
            'cd %s/gpaw/test/long/%s; ' % (self.gpawdir, job.dir) +
            'export LD_LIBRARY_PATH=/opt/acml-4.0.1/gfortran64/lib:' +
            '/opt/acml-4.0.1/gfortran64/lib:' +
            '/usr/local/openmpi-1.2.5-gfortran/lib64 && ' +
            'export PATH=/usr/local/openmpi-1.2.5-gfortran/bin:${PATH} && '+
            'mpirun ' +
            '-x PYTHONPATH=%s/gpaw ' % self.gpawdir +
            '-x GPAW_SETUP_PATH=%s ' % self.setupsdir +
            '-x GPAW_VDW=/home/camp/jensj/VDW ' +
            '%s _%s.py %s > %s.output' %
            (gpaw_python, job.id, job.arg, job.id))
        header = '\n'.join(
            ['import matplotlib',
             "matplotlib.use('Agg')",
             'import pylab',
             '_n = 1',
             'def show():',
             '    global _n',
             "    pylab.savefig('x%d.png' % _n)",
             '    _n += 1',
             'pylab.show = show',
             ''])
        i = open('%s-job.py' % job.id, 'w')
        i.write('\n'.join(
            ['#!/usr/bin/env python',
             'import os',
             'import time',
             'f = open("%s/_%s.py", "w")' % (job.dir, job.id),
             'f.write("""%s""")' % header,
             'f.write(open("%s/%s.py", "r").read())' % (job.dir, job.name),
             'f.close()',
             'f = open("%s/%s.start", "w")' % (job.dir, job.id),
             'f.write("%f\\n" % time.time())',
             'f.close()',
             'x = os.system("%s")' % cmd,
             'f = open("%s/%s.done", "w")' % (job.dir, job.id),
             'f.write("%f\\n%d\\n" % (time.time(), x))',
             '\n']))
        i.close()
        if job.ncpu == 1:
            ppn = 1
            nodes = 1
        else:
            assert job.ncpu % 4 == 0
            ppn = 4
            nodes = job.ncpu // 4
        options = ('-l nodes=%d:ppn=%d:ethernet -l walltime=%d:%02d:00' %
                   (nodes, ppn, job.tmax // 60, job.tmax % 60))
        
        x = os.popen('qsub %s %s-job.py' %
                     (options, job.id), 'r').readline().split('.')[0]

        self.log('Started: %s, %s' % (job.id, x))
        job.status = 'running'

    def install(self):
        """Install ASE and GPAW."""
        dir = '/home/camp/jensj/test-gpaw-%s' % time.asctime()
        dir = dir.replace(' ', '_').replace(':', '.')
        os.mkdir(dir)
        os.chdir(dir)

        # Export a fresh version and install:
        if os.system('svn export ' +
                     'https://svn.fysik.dtu.dk/projects/gpaw/trunk gpaw') != 0:
            raise RuntimeError('Export of GPAW failed!')
        if os.system('svn export ' +
                     'https://svn.fysik.dtu.dk/projects/ase/trunk ase') != 0:
            raise RuntimeError('Export of ASE failed!')

        os.chdir('gpaw')
        
        if os.system(
            'source /usr/local/openmpi-1.2.5-gfortran/bin/mpivars-1.2.5.sh; ' +
            'cp doc/install/Linux/Niflheim/customize_ethernet.py customize.py;'
            +
            'python setup.py build_ext ' +
            '2>&1 | grep -v "c/libxc/src"') != 0:
            raise RuntimeError('Installation failed!')

        os.system('mv ../ase/ase .')

        os.system('wget --no-check-certificate --quiet ' +
                  'http://wiki.fysik.dtu.dk/stuff/gpaw-setups-latest.tar.gz')
        os.system('tar xzf gpaw-setups-latest.tar.gz')
        self.setupsdir = dir + '/gpaw/' + glob.glob('gpaw-setups-[0-9]*')[0]
        self.gpawdir = dir
        os.chdir(self.gpawdir + '/gpaw/test/long')

    def cleanup(self):
        for id in self.ids:
            j = self.jobs[id]
            print (j.dir, j.id, j.name, j.prefix,
                   j.tmax, j.ncpu, j.deps, j.arg, j.status)

        
j = Jobs('long.log')
j.add(jobs)
j.install()
try:
    j.run()
except:
    j.cleanup()
    raise
else:
    j.print_results()
