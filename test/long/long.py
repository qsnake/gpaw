"""Run longer test jobs in parallel on Niflheim."""

import os
import sys
import time
import glob

# Test jobs:
jobs = [
#   (name, #cpus, minutes, dependencies),
    ('COAu38/Au038to', 4, 10, []),
    ('O2Pt/o2pt', 4, 40, []),
    ('../vdw/interaction', 4, 60, ['dimers']),
    ('../vdw/dimers', 4, 30, []),
    ]

# Test all exercises:
exercises = [
    ('../../doc/exercises/neb/neb1', 4, 20, []),
    ('../../doc/exercises/aluminium/Al_fcc', 4, 20, []),
    ('../../doc/exercises/aluminium/Al_fcc_convergence', 4, 20, []),
    ('../../doc/exercises/surface/work_function', 4, 20, ['testAl100']),
    ('../../doc/exercises/surface/testAl100', 4, 20, []),
    ('../../doc/exercises/diffusion/initial', 4, 20, []),
    ('../../doc/exercises/diffusion/densitydiff', 4, 20, ['solution']),
    ('../../doc/exercises/diffusion/solution', 4, 20, []),
    ('../../doc/exercises/vibrations/H2O_vib', 4, 20, ['h2o']),
    ('../../doc/exercises/vibrations/h2o', 4, 20, []),
    ('../../doc/exercises/band_structure/Na_band', 4, 20, []),
    ('../../doc/exercises/band_structure/plot_band', 4, 20, ['Na_band']),
    ('../../doc/exercises/wannier/wannier-si', 4, 20, []),
    ('../../doc/exercises/wannier/wannier-benzene', 1, 20, ['benzene']),
    ('../../doc/exercises/wannier/benzene', 4, 20, []),
    ('../../doc/exercises/lrtddft/ground_state', 4, 20, []),
    ('../../doc/exercises/transport/pt_h2_tb_transport', 4, 20, []),
    ('../../doc/exercises/transport/pt_h2_lcao', 4, 20, ['makebasis']),
    ('../../doc/exercises/transport/pt_h2_lcao_transport', 4, 20,
     ['makebasis']),
    ('../../doc/exercises/transport/makebasis', 1, 5, []),
    ('../../doc/exercises/dos/testdos', 4, 20,
     ['ferro', 'anti', 'non', 'CO', 'si', 'Al_fcc']),
    ('../../doc/exercises/stm/HAl100', 4, 20, []),
    ('../../doc/exercises/wannier/si', 4, 20, []),
    ('../../doc/exercises/wavefunctions/CO', 4, 20, []),
    ('../../doc/exercises/iron/PBE', 4, 20, ['ferro', 'anti', 'non']),
    ('../../doc/exercises/iron/ferro', 4, 20, []),
    ('../../doc/exercises/iron/anti', 4, 20, []),
    ('../../doc/exercises/iron/non', 4, 20, []),
    ('../../doc/exercises/stm/teststm', 1, 20, ['HAl100']),
    ]

jobs += exercises
#jobs = [('COAu38/Au038to', 4, 10, [])]


class Jobs:
    def __init__(self, log=sys.stdout):
        """Run jobs.
        
        jobs is a list of tuples containing:

        * Name of the python script without the '.py' part.
        * Number of processors to run the job on.
        * Approximate walltime for job.
        * List of dependencies.
        """
        self.jobs = {}
        self.names = []
        self.status = {}
        self.ids = {}
        if isinstance(log, str):
            self.fd = open(log, 'w')
        else:
            self.fd = log
        
    def log(self, *args):
        self.fd.write(' '.join(args) + '\n')
        self.fd.flush()
        
    def add(self, jobs):
        for name, p, t, dependencies in jobs:
            dir = os.path.dirname(name)
            name = os.path.basename(name)
            print name
            assert name not in self.jobs
            self.jobs[name] = (dir, p, t, dependencies)
            self.names.append(name)
                              
    def run(self):
        status = self.status

        for name in self.jobs:
            status[name] = 'waiting'

        os.chdir(self.gpawdir + '/gpaw/test/long')

        while True:
            done = True
            for name in self.jobs:
                if status[name] == 'waiting':
                    done = False
                    ready = True
                    dir, p, t, deps = self.jobs[name]
                    for dep in deps:
                        if status[dep] != 'done':
                            ready = False
                            break
                    if ready:
                        self.start(name, dir, p, t)
                elif status[name] == 'running':
                    done = False

            if done:
                return

            time.sleep(20.0)

            for name in self.jobs:
                dir, p, t, deps = self.jobs[name]
                filename = '%s/%s.done' % (dir, name)
                if status[name] == 'running' and os.path.isfile(filename):
                    code = int(open(filename).readlines()[-1])
                    if code == 0:
                        status[name] = 'done'
                        self.log(name, 'done.')
                    else:
                        status[name] = 'failed'
                        self.log('%s exited with errorcode: %d' % (name, code))
                        self.fail(name)

    def fail(self, failed_name):
        """Recursively disable jobs depending on failed job."""
        for name in self.jobs:
            dir, p, t, deps = self.jobs[name]
            if failed_name in deps:
                self.status[name] = 'disabled'
                self.log('Disabling %s' % name)
                self.fail(name)

    def print_results(self):
        for name in self.names:
            status = self.status[name]
            dir, p, t, deps = self.jobs[name]
            filename = '%s/%s.done' % (dir, name)
            if status != 'disabled' and os.path.isfile(filename):
                t = (float(open(filename).readline()) -
                     float(open(filename[:-4] + 'start').readline()))
                t = '%8.1f' % t
            else:
                t = '        '
            self.log('%20s %s %s' % (name, t, status))

    def start(self, name, dir, p, t):
        self.log('Starting: %s' % name)
        self.status[name] = 'running'

        try:
            os.remove(dir + '/' + name + '.done')
        except OSError:
            pass

        gpaw_python = (self.gpawdir +
                       '/gpaw/build/bin.linux-x86_64-2.3/gpaw-python')
        cmd = (
            'cd %s/gpaw/test/long/%s; ' % (self.gpawdir, dir) +
            'export LD_LIBRARY_PATH=/opt/acml-4.0.1/gfortran64/lib:' +
            '/opt/acml-4.0.1/gfortran64/lib:' +
            '/usr/local/openmpi-1.2.5-gfortran/lib64 && ' +
            'export PATH=/usr/local/openmpi-1.2.5-gfortran/bin:${PATH} && '+
            'mpirun ' +
            '-x PYTHONPATH=%s/gpaw ' % self.gpawdir +
            '-x GPAW_SETUP_PATH=%s ' % self.setupsdir +
            '-x GPAW_VDW=/home/camp/jensj/VDW ' +
            '%s _%s.py > %s.output' % (gpaw_python, name, name))
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
        i = open('%s-job.py' % name, 'w')
        i.write('\n'.join(
            ['#!/usr/bin/env python',
             'import os',
             'import time',
             'f = open("%s/_%s.py", "w")' % (dir, name),
             'f.write("""%s""")' % header,
             'f.write(open("%s/%s.py", "r").read())' % (dir, name),
             'f.close()',
             'f = open("%s/%s.start", "w")' % (dir, name),
             'f.write("%f\\n" % time.time())',
             'x = os.system("%s")' % cmd,
             'f = open("%s/%s.done", "w")' % (dir, name),
             'f.write("%f\\n%d\\n" % (time.time(), x))',
             '\n']))
        i.close()
        if p == 1:
            ppn = 1
            nodes = 1
        else:
            assert p % 4 == 0
            ppn = 4
            nodes = p // 4
        options = ('-l nodes=%d:ppn=%d:ethernet -l walltime=%d:%02d:00' %
                   (nodes, ppn, t // 60, t % 60))
        
        id = os.popen('qsub %s %s-job.py' % (options, name), 'r').readline()
        self.ids[name] = id.split('.')[0]

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

    def cleanup(self):
        print self.jobs
        print self.status

        
j = Jobs('long.log')
j.add(jobs)
j.install()
try:
    j.run()
except KeyboardInterrupt:
    j.cleanup()
else:
    j.print_results()
