import os
import glob
import subprocess

from gpaw.test.big.agts import Cluster


class Niflheim(Cluster):

    gpawrepo = 'https://svn.fysik.dtu.dk/projects/gpaw/trunk'
    aserepo = 'https://svn.fysik.dtu.dk/projects/ase/trunk'

    def __init__(self):
        self.dir = os.getcwd()

    def install_gpaw(self):
        if os.system('svn export %s gpaw' % self.gpawrepo) != 0:
            raise RuntimeError('Export of GPAW failed!')

        if os.system('cd gpaw&& ' +
                     'source /home/camp/modulefiles.sh&& ' +
                     'module load NUMPY&& '+
                     'python setup.py --remove-default-flags ' +
                     '--customize=doc/install/Linux/Niflheim/' +
                     'customize-thul-acml.py ' +
                     'install --home=..') != 0:
            raise RuntimeError('Installation of GPAW failed!')

        os.system('wget --no-check-certificate --quiet ' +
                  'http://wiki.fysik.dtu.dk/stuff/gpaw-setups-latest.tar.gz')
        os.system('tar xzf gpaw-setups-latest.tar.gz')
        os.system('mv gpaw-setups-[0-9]* gpaw-setups')

    def install_ase(self):
        if os.system('svn checkout %s ase' % self.aserepo) != 0:
            raise RuntimeError('Checkout of ASE failed!')
        if os.system('cd ase; python setup.py install --home=..'):
            raise RuntimeError('Installation of ASE failed!')

    def install(self):
        self.install_gpaw()
        self.install_ase()

    def submit(self, job):
        dir = os.getcwd()
        os.chdir(job.dir)
        gpaw_python = os.path.join(self.dir, 'bin/gpaw-python')

        if job.ncpus == 1:
            ppn = 1
            nodes = 1
        else:
            assert job.ncpus % 8 == 0
            ppn = 8
            nodes = job.ncpus // 8

        p = subprocess.Popen(
            ['/usr/local/bin/qsub',
             '-l',
             'nodes=%d:ppn=%d:xeon5570' % (nodes, ppn),
             '-l',
             'walltime=%d:%02d:00' %
             (job.walltime // 3600, job.walltime % 3600 // 60),
             '-N',
             job.name],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        p.stdin.write(
            'touch %s.start\n' % job.name +
            'mpiexec --mca mpi_paffinity_alone 1 ' +
            '-x PYTHONPATH=%s/lib/python:%s/lib64/python:$PYTHONPATH ' %
            (self.dir, self.dir) +
            '-x GPAW_SETUP_PATH=%s/gpaw-setups ' % self.dir +
            '%s %s %s > %s.output\n' %
            (gpaw_python, job.name, job.args, job.name) +
            'echo $? > %s.done\n' % job.name)
        p.stdin.close()
        id = p.stdout.readline().split('.')[0]
        job.pbsid = id
        os.chdir(dir)


if __name__ == '__main__':
    from gpaw.test.big.agts import AGTSQueue
    
    os.chdir(os.path.join(os.environ['HOME'], 'weekend-tests'))

    niflheim = Niflheim()
    if not os.path.isfile('bin/gpaw-python'):
        niflheim.install()

    os.chdir('gpaw')
    queue = AGTSQueue()
    queue.collect()

    if 0:
        queue.jobs = [j for j in queue.jobs if j.walltime < 3*60]

    queue.run(niflheim)

    queue.copy_created_files('../files')
