#! /usr/bin/python

class RunProfile:

    def __init__(self):
        pass

    def __compile_gpaw(self, compile_cmd):
        """called by derived classes to actually compile gpaw."""
        print "Compililing gpaw with", compile_cmd
        if os.system(compile_cmd) != 0:
            raise RuntimeError('Installation of gpaw failed!')
        return True

    def compile_gpaw(self, install_directory):
        return "Please overwrite"

    def submit(self, script, env_vars):
        return "Please overwrite"

    def set_paths(self, paths):
        self.paths = paths.copy()

    def get_path(self, name):
        """possible names are ase, gpaw, setups, root, scripts"""
        return self.paths[name]


class Niflheim(RunProfile):
    """This is the profile to run the tests on Niflheim"""
    def compile_gpaw(self, install_directory):
        """Compiles gpaw"""
        compile_cmd = 'source /home/camp/modulefiles.sh&& ' +\
                      'module load NUMPY&& '+\
                      'python setup.py --remove-default-flags ' +\
                      '--customize=doc/install/Linux/Niflheim/' +\
                      'customize-thul-acml.py ' +\
                      'install --home='+install_directory+' 2>&1 | ' +\
                      'grep -v "c/libxc/src"'

        return RunProfile.__compile_gpaw(self, compile_cmd)

    def submit(self, script, env_vars, args):
        """todo: unclear yet how environment variables should be set for jobs
        when submitted"""
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


class Local(RunProfile):
    """This is the profile to run jobs locally"""
    def compile_gpaw(self, install_directory):
        """Compiles gpaw"""
        compile_cmd = 'python setup.py  ' +\
                      'install --home='+install_directory+' 2>&1 | ' +\
                      'grep -v "c/libxc/src"'
        return RunProfile.__compile_gpaw(self, compile_cmd)

    def submit(self, script, env_vars, args):
        """simply run the script env_vars is a dictionary with environment variables"""
        #todo:
        # 1. set the environment variables
        # 2. run/submit the script)
        os.system("python "+script)

