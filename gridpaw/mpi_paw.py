# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os
import sys
import pickle
import socket

from gridpaw import debug
from gridpaw.utilities.socket import send, recv
from gridpaw.paw import Paw
import gridpaw.utilities.timing as timing


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

        if os.environ.has_key('GRIDPAW_PYTHON'):
            job = os.environ.get('GRIDPAW_PYTHON')
            job += """ -c 'from gridpaw.mpi_run import run; run("%s",%d)'""" % (host,port)
        else:
            job = """python -c 'from gridpaw.mpi_run import run; run("%s",%d)'""" % (host,port)

        job += ' --gridpaw-parallel'
        if debug:
            job += ' --gridpaw-debug'

        # Command for remote calculator:
        if os.environ.has_key('GRIDPAW_MPIRUN'):
            cmd=os.environ.get('GRIDPAW_MPIRUN')
            if cmd.find('job') == -1:
                raise RuntimeError, 'Invalid GRIDPAW_MPIRUN: ' + cmd
            cmd=cmd.replace('job',job)

        elif os.uname()[4] == 'sun4u':
            n = len(open(hostfile).readlines())
            cmd = ('GRIDPAW_PARALLEL=1; ' +
                   'export GRIDPAW_PARALLEL; ' +
                   'mprun -np %d %s &' % (n, job))

        elif sys.platform == 'aix5':
            if os.environ.has_key('LOADL_PROCESSOR_LIST'):
                cmd = ('export GRIDPAW_PARALLEL=1; ' +
                       "poe '%s' &" % job)
            else:
                n = len(open(hostfile).readlines())
                cmd = ('export GRIDPAW_PARALLEL=1; ' +
                       "poe '%s' -procs %d -hfile %s &" %
                       (job, n, hostfile))
        elif os.uname()[1]=='sepeli.csc.fi':
            n = len(open(hostfile).readlines())
            cmd = ('export GRIDPAW_PARALLEL=1; ' +
                  'mpiexec -n %d %s' % (n, job))

        else:
            n0 = len(open(hostfile).readlines())
            i, o, e = os.popen3('lamnodes C', 'r')
            n = len(o.readlines())
            cmd = ''
            if n != n0:
                if n > 0:
                    cmd = 'lamhalt; '
                if debug:
                    cmd += 'lamboot -v %s; ' % hostfile
                else:
                    cmd += 'lamboot -H %s; ' % hostfile
            if debug:
                cmd += 'mpirun -v -nw -x GRIDPAW_PARALLEL=1 C %s' % job
            else:
                cmd += 'mpirun -nw -x GRIDPAW_PARALLEL=1 C %s' % job

        # Start remote calculator:
        error = os.system(cmd)
        if error != 0:
            raise RuntimeError

        self.sckt, addr = s.accept()
        print >> out, addr
        
        string = pickle.dumps(args)

        send(self.sckt, string)
        ack = recv(self.sckt)
        if ack != 'Got your arguments - now give me some commands':
            raise RuntimeError
        s.close()

    def stop(self):
        send(self.sckt, pickle.dumps(("Stop", (), {})))
        string = recv(self.sckt)
        self.out.write(string)
        send(self.sckt,
             'Got your output - now send me your CPU time')
        cputime = pickle.loads(recv(self.sckt))
        return cputime

    def __del__(self):
        self.sckt.close()

    def __getattr__(self, attr):
        """Catch calls to methods and attributes."""
        if attr in self.paw_methods:
            self.methodname = attr
            return self.method
        
        # OK, attr was not a method - it was an attribute.
        # Send attribue name:
        string = pickle.dumps((attr, None, None))
        send(self.sckt, string)
        return pickle.loads(recv(self.sckt))

    def method(self, *args, **kwargs):
        """Communicate with remote calculation.

        Method name and arguments are send to the parallel calculation
        and results are received.  Output flushed from the calculation
        is also picked up and passed on."""

        # Send method name and arguments:
        string = pickle.dumps((self.methodname, args, kwargs))
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
