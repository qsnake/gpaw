# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os
import sys
from os.path import dirname, isfile, join
from distutils.util import get_platform
import cPickle as pickle
import socket

from gpaw import debug, trace
from gpaw.utilities.socket import send, recv
from gpaw.paw import Paw
from gpaw.mpi.config import get_mpi_command
import gpaw.utilities.timing as timing


class MPIPaw:
    # List of methods for Paw object:
    paw_methods = dir(Paw)
    
    def __init__(self, hostfile, out, *args):
        self.out = out
        # Make connection:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        port = 17000
        host = socket.gethostname()
        while True:
            try:
                s.bind((host, port))
            except socket.error:
                port += 1
            else:
                break

        s.listen(1)

        # Check if we are running in source directory and
        # have custom interpreter in the build directory:
        dir = dirname(__file__)
        gpaw_python = join(dir, '../..', 'build',
                           'bin.%s-%s/gpaw-python' % (get_platform(), sys.version[0:3]))
        if not isfile(gpaw_python):
            gpaw_python = None
            # Look in the PATH
            paths = os.environ.get('PATH')
            paths = paths.split(os.pathsep)
            for path in paths:
                if isfile(join(path, 'gpaw-python')):
                    gpaw_python = join(path, 'gpaw-python')
                    break

        # If the environment variable GPAW_PYTHON is set, use that:
        gpaw_python = os.environ.get('GPAW_PYTHON', gpaw_python)
        if gpaw_python is None:
            raise RuntimeError('Custom interpreter is not found')
        
        # This is the Python command that all processors wil run:
        # line = 'from gpaw.mpi.run import run; run("%s", %d)' % (host, port)
        f=open('par_run.py','w')
        print >> f, 'from gpaw.mpi.run import run'
        print >> f, 'run("%s",%d)' % (host,port)
        f.close()
        
        #job = python + " -c '" + line + "' --gpaw-parallel"
        job = gpaw_python + ' par_run.py'

        if debug:
            job += ' --gpaw-debug'
        if trace:
            job += ' --gpaw-trace'

        # Get the command to start mpi.  Typically this will be
        # something like:
        #
        #   cmd = 'mpirun -np %(np)d --hostfile %(hostfile)s %(job)s &'
        #
        cmd = get_mpi_command(debug)
        try:
            np = len(open(hostfile).readlines())
        except:
            np = None

        # Insert np, hostfile and job:
        cmd = cmd % {'job': job,
                     'hostfile': hostfile,
                     'np': np}

        error = os.system(cmd)
        if error != 0:
            raise RuntimeError

        self.sckt, addr = s.accept()
        
        string = pickle.dumps(args, -1)

        send(self.sckt, string)
        ack = recv(self.sckt)
        if ack != 'Got your arguments - now give me some commands':
            raise RuntimeError
        s.close()

    def stop(self):
        send(self.sckt, pickle.dumps(("Stop", (), {}), -1))
        string = recv(self.sckt)
        self.out.write(string)
        send(self.sckt,
             'Got your output - now send me your CPU time')
        cputime = pickle.loads(recv(self.sckt))
        return cputime

    def __del__(self):
        self.sckt.close()
        try:
            os.remove('par_run.py')
        except OSError:
            pass

    def __getattr__(self, attr):
        """Catch calls to methods and attributes."""
        if attr in self.paw_methods:
            self.methodname = attr
            return self.method
        
        # OK, attr was not a method - it was an attribute.
        # Send attribue name:
        string = pickle.dumps((attr, None, None), -1)
        send(self.sckt, string)
        return pickle.loads(recv(self.sckt))

    def method(self, *args, **kwargs):
        """Communicate with remote calculation.

        Method name and arguments are send to the parallel calculation
        and results are received.  Output flushed from the calculation
        is also picked up and passed on."""

        # Send method name and arguments:
        string = pickle.dumps((self.methodname, args, kwargs), -1)
        send(self.sckt, string)
        
        # Wait for result:
        while True:
            tag, stuff = pickle.loads(recv(self.sckt))
            if tag == 'output':
                # This was not the result - the output was flushed:
                self.out.write(stuff)
                self.out.flush()
                timing.update()
            elif tag == 'result':
                return stuff
            else:
                raise RuntimeError('Unknown tag: ' + tag)
