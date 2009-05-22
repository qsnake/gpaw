"""Run longer test jobs in parallel on Niflheim."""

import os
import time
import glob

# Test jobs:
jobs = [
#   (name, #cpus, minutes, dependencies),
    ('COAu38/Au038to', 4, 10, []),
    ('O2Pt/o2pt', 4, 40, []),
    ('../vdw/interaction', 4, 60,
     [('../vdw/dimers', 4, 30, [])]),
    ]

# Test all exercises:
exercises = [
    ('../../doc/exercises/neb/neb1', 4, 20, []),
    ('../../doc/exercises/aluminium/Al_fcc_convergence', 4, 20, []),
    ('../../doc/exercises/surface/Al100', 4, 20, []),
    ('../../doc/exercises/surface/work_function', 4, 20, []),
    ('../../doc/exercises/diffusion/initial', 4, 20, []),
    ('../../doc/exercises/diffusion/solution', 4, 20, []),
    ('../../doc/exercises/diffusion/densitydiff', 4, 20, []),
    ('../../doc/exercises/vibrations/h2o', 4, 20, []),
    ('../../doc/exercises/vibrations/H2O_vib', 4, 20, []),
    ('../../doc/exercises/band_structure/Na_band', 4, 20, []),
    ('../../doc/exercises/band_structure/plot_band', 4, 20, []),
    ('../../doc/exercises/wannier/wannier-si', 4, 20, []),
    ('../../doc/exercises/wannier/benzene', 4, 20, []),
    ('../../doc/exercises/wannier/wannier-benzene', 4, 20, []),
    ('../../doc/exercises/dos/pdos', 4, 20, []),
    ('../../doc/exercises/lrtddft/ground_state', 4, 20, []),
    ('../../doc/exercises/transport/pt_h2_tb_transport', 4, 20, []),
    ('../../doc/exercises/transport/pt_h2_lcao', 4, 20, []),
    ('../../doc/exercises/transport/pt_h2_lcao_transport', 4, 20, []),
    ('../../doc/exercises/test', 4, 20,
     [('../../doc/exercises/stm/HAl100', 4, 20, []),
      ('../../doc/exercises/aluminium/Al_fcc', 4, 20, []),
      ('../../doc/exercises/wannier/si', 4, 20, []),
      ('../../doc/exercises/wavefunctions/CO', 4, 20, []),
      ('../../doc/exercises/iron/PBE', 4, 20,
       [('../../doc/exercises/iron/ferro', 4, 20, []),
        ('../../doc/exercises/iron/anti', 4, 20, []),
        ('../../doc/exercises/iron/non', 4, 20, [])])])
    ]

#jobs += exercises
jobs = [('COAu38/Au038to', 4, 10, [])]


class Jobs:
    def __init__(self, jobs):
        """Run jobs.
        
        jobs is a list of tuples containing:

        * Name of the python script without the '.py' part.
        * Number of processors to run the job on.
        * Approximate walltime for job.
        * List of dependencies.
        """
        self.jobs = {}
        self.names = []
        self.add(jobs)
        self.status = dict([(name, 'queued') for name in self.jobs])
        self.install()

    def add(self, jobs):
        names = []
        for name, p, t, dependencies in jobs:
            self.jobs[name] = (p, t, self.add(dependencies))
            self.names.append(name)
            names.append(name)
        return names
                              
    def run(self):
        status = self.status
        while True:
            done = True
            for name in self.jobs:
                if status[name] == 'queued':
                    done = False
                    ready = True
                    p, t, deps = self.jobs[name]
                    for dep in deps:
                        if status[dep] != 'done':
                            ready = False
                            break
                    if ready:
                        self.start(name, p, t)
                elif status[name] == 'running':
                    done = False

            if done:
                return

            time.sleep(20.0)

            for name in self.jobs:
                if (status[name] == 'running' and
                    os.path.isfile(name + '.done')):
                    code = int(open(name + '.done').readlines()[-1])
                    if code == 0:
                        status[name] = 'done'
                        print name + '.py done.'
                    else:
                        status[name] = 'failed'
                        print '%s.py exited with errorcode: %d' % (name, code)
                        self.fail(name)

    def fail(self, name):
        p, t, deps = self.jobs[name]
        for dep in deps:
            self.status[dep] = 'disabled'
            print 'Disabling %s.py' % dep
            self.fail(dep)

    def print_results(self):
        for name in self.names:
            status = self.status[name]
            if status != 'disabled' and os.path.isfile(name + '.done'):
                t = (float(open(name + '.done').readline()) -
                     float(open(name + '.start').readline()))
                t = '%8.1f' % t
            else:
                t = '--------'
            print '%40s %s %s' % (name, t, status)

    def start(self, name, p, t):
        print 'Starting: %s.py' % name
        self.status[name] = 'running'
        try:
            os.remove(name + '.done')
        except OSError:
            pass

        dir = os.path.dirname(name)
        name = os.path.basename(name)
        gpaw_python = self.gpawdir + '/bin/gpaw-python'
        cmd = (
            'cd %s/gpaw/test/long/%s; ' % (self.gpawdir, dir) +
            'export LD_LIBRARY_PATH=/opt/acml-4.0.1/gfortran64/lib:' +
            '/opt/acml-4.0.1/gfortran64/lib:' +
            '/usr/local/openmpi-1.2.5-gfortran/lib64 && ' +
            'export PYTHONPATH=%s/lib64/python && ' % self.gpawdir +
            'export GPAW_SETUP_PATH=%s && ' % self.setupsdir +
            'export GPAW_VDW=/home/camp/jensj/VDW && ' +
            'export PATH=/usr/local/openmpi-1.2.5-gfortran/bin:${PATH} && '+
            'mpirun %s %s.py > %s.output' % (gpaw_python, name, name))
        i = open('%s-job.py' % name, 'w')
        i.write('\n'.join(
            ['#!/usr/bin/env python',
             'import os',
             'import time',
             'f = open("%s/%s.start", "w")' % (dir, name),
             'f.write("%f\\n" % time.time())',
             'x = os.system("%s")' % cmd,
             'f = open("%s/%s.done", "w")' % (dir, name),
             'f.write("%f\\n%d\\n" % (time.time(), x))',
             '\n']))
        i.close()
        assert p % 4 == 0
        options = ('-l nodes=%d:ppn=4:ethernet -l walltime=%d:%02d:00' %
                   (p // 4, t // 60, t % 60))
        
        os.system('qsub %s %s-job.py' % (options, name))


    def install(self):
        dir = '/home/camp/jensj/test-%s' % time.asctime().replace(' ', '_')
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
            'python setup.py install --home=%s ' % dir +
            '2>&1 | grep -v "c/libxc/src"') != 0:
            raise RuntimeError('Installation failed!')

        os.system('mv ../ase/ase ../lib64/python')

        os.system('wget --no-check-certificate --quiet ' +
                  'http://wiki.fysik.dtu.dk/stuff/gpaw-setups-latest.tar.gz')
        os.system('tar xzf gpaw-setups-latest.tar.gz')
        self.setupsdir = dir + '/gpaw/' + glob.glob('gpaw-setups-*')[0]
        self.gpawdir = dir
        
j = Jobs(jobs)
j.run()
j.print_results()
