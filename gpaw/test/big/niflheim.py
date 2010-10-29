import os
import glob
import subprocess

from gpaw.test.big.agts import Cluster


class Niflheim(Cluster):

    gpawrepo = 'https://svn.fysik.dtu.dk/projects/gpaw/trunk'
    aserepo = 'https://svn.fysik.dtu.dk/projects/ase/trunk'

    def __init__(self):
        self.dir = os.getcwd()
        self.revision = None

    def install_gpaw(self):
        if os.system('svn checkout %s gpaw' % self.gpawrepo) != 0:
            raise RuntimeError('Checkout of GPAW failed!')

        p = subprocess.Popen(['svnversion', 'gpaw'], stdout=subprocess.PIPE)
        self.revision = int(p.stdout.read())

        if os.system('cd gpaw&& ' +
                     'source /home/camp/modulefiles.sh&& ' +
                     'module load NUMPY&& '+
                     'module load open64/4.2.3-0 && ' +
                     'python setup.py --remove-default-flags ' +
                     '--customize=doc/install/Linux/Niflheim/' +
                     'el5-xeon-open64-goto2-1.13-acml-4.4.0.py ' +
                     'build_ext') != 0:
            raise RuntimeError('Installation of GPAW (Xeon) failed!')
        if os.system('ssh fjorm "cd weekend-tests/gpaw&& ' +
                     'source /home/camp/modulefiles.sh&& ' +
                     'module load NUMPY&& '+
                     'module load open64/4.2.3-0 && ' +
                     'python setup.py --remove-default-flags ' +
                     '--customize=doc/install/Linux/Niflheim/' +
                     'el5-opteron-open64-goto2-1.13-acml-4.4.0.py ' +
                     'build_ext"') != 0:
            raise RuntimeError('Installation of GPAW (Opteron) failed!')
        
        os.system('wget --no-check-certificate --quiet ' +
                  'http://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-latest.tar.gz')
        os.system('tar xzf gpaw-setups-latest.tar.gz')
        os.system('rm gpaw-setups-latest.tar.gz')
        os.system('mv gpaw-setups-[0-9]* gpaw/gpaw-setups')

    def install_ase(self):
        if os.system('svn checkout %s ase' % self.aserepo) != 0:
            raise RuntimeError('Checkout of ASE failed!')

    def install(self):
        self.install_gpaw()
        self.install_ase()

    def submit(self, job):
        dir = os.getcwd()
        os.chdir(job.dir)

        self.write_pylab_wrapper(job)

        if job.queueopts is None:
            if job.ncpus == 1:
                ppn = '1:opteron2218:ethernet'
                nodes = 1
                arch = 'linux-x86_64-opteron-2.4'
            elif job.ncpus % 8 == 0:
                ppn = '8:xeon5570'
                nodes = job.ncpus // 8
                arch = 'linux-x86_64-xeon-2.4'
            else:
                assert job.ncpus % 4 == 0
                ppn = '4:opteron2218:ethernet'
                nodes = job.ncpus // 4
                arch = 'linux-x86_64-opteron-2.4'
            queueopts = '-l nodes=%d:ppn=%s' % (nodes, ppn)
        else:
            queueopts = job.queueopts
            arch = 'linux-x86_64-xeon-2.4'
            
        gpaw_python = os.path.join(self.dir, 'gpaw', 'build',
                                   'bin.' + arch, 'gpaw-python')

        submit_pythonpath = ':'.join([
            '%s/ase' % self.dir,
            '%s/gpaw' % self.dir,
            '%s/gpaw/build/lib.%s' % (self.dir, arch),
            '$PYTHONPATH'])
        submit_gpaw_setup_path = '%s/gpaw/gpaw-setups' % self.dir

        run_command = '. /home/camp/modulefiles.sh&& '
        run_command += 'module load MATPLOTLIB&& ' # loads numpy, mpl, ...

        if job.ncpus == 1:
            # don't use mpi here,
            # this allows one to start mpi inside the *.agts.py script
            run_command += ' PYTHONPATH=' + submit_pythonpath
            run_command += ' GPAW_SETUP_PATH=' + submit_gpaw_setup_path
        else:
            run_command += 'module load '
            run_command += 'openmpi/1.3.3-1.el5.fys.gfortran43.4.3.2&& '
            run_command += 'mpiexec --mca mpi_paffinity_alone 1 '
            run_command += '-x PYTHONPATH=' + submit_pythonpath
            run_command += ' -x GPAW_SETUP_PATH=' + submit_gpaw_setup_path
            run_command += ' -x OMP_NUM_THREADS=1'

        p = subprocess.Popen(
            ['/usr/local/bin/qsub',
             '-V',
             queueopts,
             '-l',
             'walltime=%d:%02d:00' %
             (job.walltime // 3600, job.walltime % 3600 // 60),
             '-N',
             job.name],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        p.stdin.write(
            'touch %s.start\n' % job.name +
            run_command +
            ' %s %s.py %s > %s.output\n' %
            (gpaw_python, job.script, job.args, job.name) +
            'echo $? > %s.done\n' % job.name)
        p.stdin.close()
        id = p.stdout.readline().split('.')[0]
        job.pbsid = id
        os.chdir(dir)


if __name__ == '__main__':
    from gpaw.test.big.agts import AGTSQueue

    os.chdir(os.path.join(os.environ['HOME'], 'weekend-tests'))

    niflheim = Niflheim()
    if 1:
        niflheim.install()

    os.chdir('gpaw')
    queue = AGTSQueue()
    queue.collect()

    # examples of selecting jobs
    #queue.jobs = [j for j in queue.jobs if j.script == 'testsuite.agts.py']
    #queue.jobs = [j for j in queue.jobs if j.script == 'neb.agts.py']
    #queue.jobs = [j for j in queue.jobs if j.dir.startswith('doc')]
    #queue.jobs = [j for j in queue.jobs
    #              if j.dir.startswith('gpaw/test/big/bader_water')]
    #queue.jobs = [j for j in queue.jobs
    #              if j.dir.startswith('doc/devel/memory_bandwidth')]

    nfailed = queue.run(niflheim)

    queue.copy_created_files('/home/camp2/jensj/WWW/gpaw-files')

    # Analysis:
    import matplotlib
    matplotlib.use('Agg')
    from gpaw.test.big.analysis import analyse
    user = os.environ['USER']
    analyse(queue,
            '../analysis/analyse.pickle',  # file keeping history
            '../analysis',                 # Where to dump figures
            rev=niflheim.revision,
            mailto='gpaw-developers@listserv.fysik.dtu.dk',
            #mailto='jensj@fysik.dtu.dk',
            mailserver='servfys.fysik.dtu.dk',
            attachment='status.log')

    if nfailed == 0:
        tag = 'success'
    else:
        tag = 'failed'

    os.chdir('..')
    dir = os.path.join('/scratch', user, 'gpaw-' + tag)
    os.system('rm -rf %s-old' % dir)
    os.system('mv %s %s-old' % (dir, dir))
    os.system('mv gpaw %s' % dir)
