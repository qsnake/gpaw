import os
import sys
import glob

class QueueingSystem:
    pass

class Niflheim(QueueingSystem):
    compile_cmd = ('source /home/camp/modulefiles.sh&& ' +
                   'module load NUMPY&& '+
                   'python setup.py --remove-default-flags ' +
                   '--customize=doc/install/Linux/Niflheim/' +
                   'customize-thul-acml.py ' +
                   'install --home=.')
    gpawrepo = 'https://svn.fysik.dtu.dk/projects/gpaw/trunk'
    aserepo = 'https://svn.fysik.dtu.dk/projects/ase/trunk'
    lib = 'lib64'

    def install(self):
        """Install ASE and GPAW."""


        # Export a fresh version and install:
        if os.system('svn export %s gpaw' % self.gpawrepo) != 0:
            raise RuntimeError('Export of GPAW failed!')
        if os.system('svn export %s ase' % self.aserepo) != 0:
            raise RuntimeError('Export of ASE failed!')

        os.chdir('gpaw')

        if os.system(self.compile_cmd) != 0:
            raise RuntimeError('Installation failed!')

        os.system('mv ../ase/ase ../%s/python' % self.lib)

        os.chdir('..')

        os.system('wget --no-check-certificate --quiet ' +
                  'http://wiki.fysik.dtu.dk/stuff/gpaw-setups-latest.tar.gz')
        os.system('tar xzf gpaw-setups-latest.tar.gz')
        self.setupsdir = os.path.join(glob.glob('gpaw/gpaw-setups-[0-9]*')[0])

    def submit(self, job):
        gpaw_python = 'gpaw/bin/gpaw-python'
        cmd = (
            'cd gpaw/gpaw/test/big/%s; ' % job.directory +
            'mpiexec --mca mpi_paffinity_alone 1 ' +
            '-x PYTHONPATH=gpaw/%s/python:$PYTHONPATH ' % self.lib +
            '-x GPAW_SETUP_PATH=%s ' % self.setupsdir +
            '%s %s %s > %s.output' %
            (gpaw_python, job.filename, job.argstring, job.identifier))


        if job.ncpu == 1:
            ppn = 1
            nodes = 1
        else:
            assert job.ncpu % 8 == 0
            ppn = 8
            nodes = job.ncpu // 8

        options = ('-l nodes=%d:ppn=%d:xeon5570 -l walltime=%d:%02d:00' %
                   (nodes, ppn, job.tmax // 60, job.tmax % 60))
        
        print 'qsub %s %s-job.py' % (options, job.id)
        x = os.popen('/usr/local/bin/qsub %s %s-job.py' %
                     (options, job.id), 'r').readline().split('.')[0]

        self.log('# Started: %s, %s' % (job.id, x))
        job.status = 'running'
