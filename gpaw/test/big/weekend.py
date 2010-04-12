#!/usr/bin/env python
"""Run longer test jobs in parallel on Niflheim."""

# crontab -e
# MAILTO=name
#dom=day of month
#dow=day of week (0 - 6) (Sunday=0)
#min hr dom month dow cmd
#*  *  *    *    * python ..../weekend.py

import os
import sys
import glob
import datetime
import atexit
import tempfile

tmpfilename = "/tmp/long_tests"

#from optparse import OptionParser

class RunProfile:

    def __init__(self):
        pass

    def gpaw_compile_cmd(self):
        return "Please overwrite"

    def submit(self, script, env_vars):
        return "Please overwrite"

class Niflheim(RunProfile):
    def gpaw_compile_cmd(self):
        compile_cmd = 'source /home/camp/modulefiles.sh&& ' +\
                      'module load NUMPY&& '+\
                      'python setup.py --remove-default-flags ' +\
                      '--customize=doc/install/Linux/Niflheim/' +\
                      'customize-thul-acml.py ' +\
                      'install --home='+install_directory+' 2>&1 | ' +\
                      'grep -v "c/libxc/src"'

        return compile_cmd

    def submit(self, job, env_vars):
        #todo:
        # 1. set the environment variables correctly when running job
        # 2. run/submit the script)

        try:
            os.remove(job.prefix + '.done')
        except OSError:
            pass

        gpaw_python = self.gpawdir + '/bin/gpaw-python'
        cmd = (
            'cd %s/gpaw/gpaw/sunday/%s; ' % (self.gpawdir, job.dir) +
            'mpiexec --mca mpi_paffinity_alone 1 ' +
            '-x PYTHONPATH=%s/lib64/python:$PYTHONPATH ' % self.gpaw_dir +
            '-x GPAW_SETUP_PATH=%s ' % self.setupsdir +
            '%s _%s.py %s > %s.output' %
            (gpaw_python, job.id, job.arg, job.id))

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
        #x = os.system('/usr/local/bin/qsub %s %s-job.py' % (options, job.id))

        self.log('# Started: %s, %s' % (job.id, x))
        job.status = 'running'


class Local(RunProfile):
    def gpaw_compile_cmd(self, install_directory):
        return        'python setup.py  ' +\
                      'install --home='+install_directory+' 2>&1 | ' +\
                      'grep -v "c/libxc/src"'

    def submit(self, script, env_vars, args):
        """simply run the script env_vars is a dictionary with environment variables"""
        #todo:
        # 1. set the environment variables
        # 2. run/submit the script)
        os.system("python "+script)



class Test:
    
    def __init__(self, profile):
        self.profile = profile
        self.run_day = 1 #0=Monday, 1=Tuesday, ...
        self.test_dir = os.path.join(
                            os.environ["HOME"],
                            "agts",
                            self.get_current_test_directory_name())
        self.ase_dir = os.path.join(self.test_dir, "ase")
        self.gpaw_dir = os.path.join(self.test_dir, "gpaw")
        self.script_dir = os.path.join(self.test_dir, "scripts")
        self.determine_state()

    def get_current_test_directory_name(self):
        #determine the last day it should have run
        today = datetime.date.today()
        if self.run_day == today.weekday():
            last_run = today
        else:
            week_offset = datetime.timedelta(days=(today.weekday()-self.run_day))
            last_run = today + week_offset
        return last_run.strftime("%y.%m.%d")

    def setup(self):
        """creates the setup and copies script instances to execute_path"""
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        if not os.path.exists(self.gpaw_dir):
            #install gpaw, if not already
            self.install_gpaw()

        #Copy the script files to the /output
        


    def install_gpaw(self):
        """Install ASE and GPAW."""
        # Export a fresh version and install:
        if os.system('svn export ' +
                     'https://svn.fysik.dtu.dk/projects/gpaw/trunk '+self.gpaw_dir) != 0:
            raise RuntimeError('Export of GPAW failed!')
        if os.system('svn export ' +
                     'https://svn.fysik.dtu.dk/projects/ase/trunk '+self.ase_dir) != 0:
            raise RuntimeError('Export of ASE failed!')

        os.chdir(self.gpaw_dir)

        compile_cmd = self.profile.gpaw_compile_cmd(self.gpaw_dir)

        print "Compile command", compile_cmd
        if os.system(compile_cmd) != 0:
            raise RuntimeError('Installation failed!')

        os.system('mv ../ase/ase ../lib64/python')

        os.system('wget --no-check-certificate --quiet ' +
                  'http://wiki.fysik.dtu.dk/stuff/gpaw-setups-latest.tar.gz')
        os.system('tar xzf gpaw-setups-latest.tar.gz')
        self.setups_dir = os.path.join(self.gpaw_dir + glob.glob('gpaw-setups-[0-9]*')[0])

    def determine_state(self):
        print "Load all scripts and determine dependencies."

    def run_jobs(self):
        print "Run the pending jobs"

    def run(self):
        """runs or continues the long tests"""
        self.setup()
        self.determine_state()
        self.run_jobs()
        self.print_state()

    def print_state(self):
        state_text = "State for the weekly tests"
        print state_text
        print len(state_text)*"="
        print "Test directory:  ", self.test_dir
        print "Ase directory:   ", self.ase_dir
        print "GPAW directory:  ", self.gpaw_dir
        print "Script directory:", self.script_dir
        print ""
        print "Not yet implemented"


def lock():
    if os.path.exists(tmpfilename):
        #print "Other instance is already running."
        return False

    fout = open(tmpfilename, "w+")
    fout.close()
    return os.path.exists(tmpfilename)

def unlock():
    os.remove(tmpfilename)

def print_usage():
    print "Please use"
    print "    python", sys.argv[0] + " --info"
    print "for status and informations or"
    print "    python", sys.argv[0] + " --local"
    print "to run it locally (default is Niflheim"

def main():
    #make sure only one instance is running at any time:
    if not lock():
        return

    atexit.register(unlock)

    if len(sys.argv)>=2:
        if sys.argv[1]=="--info":
            t.print_state()
            return
        elif sys.argv[1]=="--local":
            t = Test(Local())
        else:
            print_usage()
            return
    else:
            t = Test(Niflheim())
    t.print_state()
    t.run()

if __name__ == "__main__":
    main()