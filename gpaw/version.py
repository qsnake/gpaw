# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

version = '0.6'

ase_required_version = '3.1.0'
ase_required_svnrevision = '1062'

from os import popen3, path

def write_svnrevision(output):
    f = open(path.join('gpaw', 'svnrevision.py'),'w')
    f.write('svnrevision = "%s"\n' % output)
    f.close()
    print 'svnrevision = ' +output+' written to gpaw/svnrevision.py'
    # assert svn:ignore property if the installation is under svn control
    # because svnrevision.py has to be ignored by svn!
    cmd = popen3('svn propset svn:ignore svnrevision.py gpaw')[1]
    output = cmd.read()
    cmd.close()

def read_svnrevision(filename):
    f = open(filename,'r')
    s = f.read()
    f.close()
    exec(s)
##    print 'svnrevision = ' +svnrevision+' read from '+filename # MDTMP
    return svnrevision

def get_svnversion(dir='gpaw'):
    # try to get the last svn revision number from svnversion
    try: 
        # subprocess was introduced with python 2.4
        from subprocess import Popen, PIPE
        cmd = Popen('svnversion -n '+dir, 
                    shell=True, stdout=PIPE, stderr=PIPE, close_fds=True).stdout
    except:
        cmd = popen3('svnversion -n '+dir)[1] # assert that we are in gpaw project
    output = cmd.read()
    cmd.close()
    svnrevisionfile = path.join(dir, 'svnrevision.py')
    # we build from exported source (e.g. rpmbuild)
    if output.startswith('exported') and path.isfile(svnrevisionfile):
        # read the last svn revision number
        output = read_svnrevision(svnrevisionfile)
    return output

def svnversion(version):
    revision = version # the default output of this function
    try:
        # try to get the last svn revision number from gpaw/svnrevision.py
        from gpaw.svnrevision import svnrevision
        # gpaw is installed, here:
        from gpaw import __file__ as f
        # get the last svn revision number from svnversion gpaw/gpaw dir
        gpawdir = path.abspath(path.dirname(f))
        # assert we have gpaw/svnrevision.py file
        svnrevisionfile = path.join(gpawdir, 'svnrevision.py')
        # we build from exported source (e.g. rpmbuild)
        assert path.isfile(svnrevisionfile)
        if path.split(
            path.abspath(path.join(gpawdir, path.pardir)))[1] == 'gpaw':
            # or from svnversion gpaw dir
            gpawdir = path.join(gpawdir, path.pardir)
        # version.py can be called from any place so we need to specify gpawdir
        output = get_svnversion(gpawdir)
        if (output != '') and (output != svnrevision) and (not output.startswith('exported')):
            # output the current svn revision number into gpaw/svnrevision.py
            svnrevision = output
        version = version+'.'+svnrevision
    except:
        # gpaw is not installed:
        # try to get the last svn revision number from svnversion
        output = get_svnversion()
        if (output != '') and (not output.startswith('exported')):
            # svnversion exists:
            # we are sure to have the write access as what we are doing
            # is running setup.py now (even during rpmbuild)!
            # save the current svn revision number into gpaw/svnrevision.py
            write_svnrevision(output)
            svnrevision = output
            version = version+'.'+svnrevision
    ##
    return version

version = svnversion(version)
