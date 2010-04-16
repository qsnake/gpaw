#!/usr/bin/env python
"""Run longer test jobs in parallel on Niflheim."""

################################################################################
## Register cronjob:                                                          ##
################################################################################

# 1. edit the crontab-file
#    $ crontab -e
# 2. adapt the following text
#    MAILTO=name
#    # dom=day of month
#    # dow=day of week (0 - 6) (Sunday=0)
#    # min hr dom month dow cmd
#    *  *  *    *    * python weekend.py -q -niflheim

# Do not import anything else than standard python modules here:
import os
import sys
import glob
import datetime
import atexit
import tempfile
import platform

################################################################################
##                                                                            ##
## Global settings that define behaviour of the tests                         ##
##                                                                            ##
## 1. define here the weekday the tests should run:                           ##
run_day = 1 #0=Monday, 1=Tuesday, ...                                         ##
##                                                                            ##
##                                                                            ##
## 2. define the base directory for the tests:                                ##
test_base_path = os.path.join(os.environ["HOME"],                             ##
                              "agts")                                         ##
##                                                                            ##
##                                                                            ##
## define the name of the lock file                                           ##
tmpfilename = os.path.join(tempfile.gettempdir(), "big_tests.lock")           ##
##                                                                            ##
##                                                                            ##
################################################################################


def get_current_test_directory_name():
    #determine the last day it should have run
    today = datetime.date.today()
    if run_day == today.weekday():
        last_run = today
    else:
        week_offset = datetime.timedelta(days=(today.weekday()-run_day))
        last_run = today - week_offset
    return os.path.join(test_base_path,
                        last_run.strftime("%y.%m.%d"))

def get_paths():
    return {
        "ase":os.path.join(get_current_test_directory_name(), "ase"),
        "gpaw":os.path.join(get_current_test_directory_name(), "gpaw"),
        "root":get_current_test_directory_name(),
        "setups":os.path.join(get_current_test_directory_name(),"gpaw_setups"),
        "test_scripts":os.path.join(get_current_test_directory_name(), "test_scripts")
    }

def prepend_env(env_var_name, list_of_paths):
    """prepends paths to the environment variables"""
    if os.environ.has_key(env_var_name):
        paths = os.path.pathsep+os.environ[env_var_name]
    else:
        paths = ""
    
    for path in list_of_paths:
        paths =  path + os.path.pathsep + paths

    if paths.endswith(os.path.pathsep):
        paths = paths[:-1]

    os.environ[env_var_name] = paths

def is_installed():
    """true, if the test directory exist"""
    return os.path.exists(get_current_test_directory_name())

def install_all():
    """calls functions to install ase and gpaw and sets up the directories"""
    if not os.path.exists(test_base_path):
        os.makedirs(test_base_path)
    if not os.path.exists(get_paths()["gpaw"]):
        install_ase_gpaw()
    #create the other directories
    for path in get_paths().values():
        if not os.path.exists(path):
            os.makedirs(path)

def install_ase_gpaw():
    """Install ASE and GPAW."""
    # Export a fresh version and install:
    if os.system('svn export ' +
                 'https://svn.fysik.dtu.dk/projects/gpaw/trunk '+get_paths()["gpaw"]) != 0:
        raise RuntimeError('Export of GPAW failed!')
    if os.system('svn export ' +
                 'https://svn.fysik.dtu.dk/projects/ase/trunk '+get_paths()["ase"]) != 0:
        raise RuntimeError('Export of ASE failed!')

    os.chdir(get_paths()["gpaw"])

    if platform.architecture()[0].find("32")!=-1:
        lib = "lib"
    elif platform.architecture()[0].find("64")!=-1:
        lib = "lib64"
    else:
        raise Exception("Unknown architecture: ", platform.architecture()[0])

    compile_cmd = 'source /home/camp/modulefiles.sh&& ' +\
                  'module load NUMPY&& '+\
                  'python setup.py --remove-default-flags ' +\
                  '--customize=doc/install/Linux/Niflheim/' +\
                  'customize-thul-acml.py ' +\
                  'install --home='+get_paths()["gpaw"]+' 2>&1 | ' +\
                  'grep -v "c/libxc/src"'

    os.system(compile_cmd)

    os.system('mv ../ase/ase ../%s/python' % lib)

    os.system('wget --no-check-certificate --quiet ' +
              'http://wiki.fysik.dtu.dk/stuff/gpaw-setups-latest.tar.gz')
    os.system('tar xzf gpaw-setups-latest.tar.gz')
    setups_dir = os.path.join(get_paths()["gpaw"], glob.glob('gpaw-setups-[0-9]*')[0])
    os.system("mv "+setups_dir+" "+get_paths()["setups"])

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


from optparse import OptionParser

def main():
    parser = OptionParser()
    parser.add_option("-p", "--profile", dest="profile_name",
                      help="select a profile - currently local "+\
                           "and niflheim are supported", action="store", \
                        default="local")
    parser.add_option("-l", "--local",
                      help="select the local profile (same as \"-p local\")",\
                      action="store_const", dest="profile_name",const="local")
    parser.add_option("-n", "--niflheim",
                      help="select the niflheim profile (same as \"-p niflheim\")",\
                      action="store_const", dest="profile_name",const="niflheim")
    parser.add_option("-x", "--pure_local",
                      help="runs in the local environment without installing ase/gpaw from svn",\
                      action="store_true", dest="pure", default=False)
    parser.add_option("-q", "--qiet",
                      help="verbose mode",\
                      action="store_false", dest="verbose", default=True)

    (options, args) = parser.parse_args()
    verbose = options.verbose

    welcome = "Welcome to AGTS"

    if verbose:
        print len(welcome)*"*"
        print welcome
        print len(welcome)*"*"
        print
        print "Command line options: ", options

    #make sure only one instance is running at any time:
    if not lock():
        #print "An other agts process is already running - aborting"
        return
    atexit.register(unlock)

    if not options.pure and not is_installed():
        installed = False
        install_all()
    else:
        installed = True

    if not options.pure:
        print "Setting PYTHONPATH to", [get_paths()["ase"],get_paths()["gpaw"]]
        prepend_env("PYTHONPATH", [get_paths()["ase"],get_paths()["gpaw"]])
        print "Setting GPAW_SETUP_PATH to", get_paths()["setups"]
        prepend_env("GPAW_SETUP_PATH", [get_paths()["setups"]])
        print "Paths:"
        for name in get_paths().keys():
            print name+":",get_paths()[name]

    # The environment is set up, now we are allowed to to compile
    # and load gpaw etc.

    if verbose:
        print "Loading profile \""+options.profile_name+"\" ..."

    if options.profile_name=="local":
        from profile import Local
        profile = Local()
    elif options.profile_name=="niflheim":
        from profile import Niflheim
        profile = Niflheim()
    else:
        raise Exception("unknown profile \""+options.profile_name+"\"")
    
    profile.set_paths(get_paths())

    if not installed:
        if verbose:
            print "compiling gpaw ..."
        profile.compile_gpaw(get_paths()["gpaw"])

    print
    print
    print

    if options.pure:
        print 20*"*"
        print "Using the current environment with the installed ase and gpaw"
        print "The profile for compiling and submitting is", profile
        print "Todo: create AGTSQueue object and run"
    else:
        # 
        #
        print 20*"*"
        print "Using the modified environment with ase and gpaw from svn"
        print "The profile for compiling and submitting is", profile
        print "Paths:"
        for name in get_paths().keys():
            print name+":",get_paths()[name]
        print
        print "Todo: create AGTSQueue object and run"
    print "Remember to use the correct environment (variables) when running jobs!"


if __name__ == "__main__":
    main()