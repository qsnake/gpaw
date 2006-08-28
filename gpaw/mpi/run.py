import cPickle as pickle
import socket
import StringIO

import Numeric as num

import gpaw.mpi as mpi
from gpaw.startup import create_paw_object
from gpaw.utilities.socket import send, recv
from gpaw.utilities.timing import clock
from gpaw.utilities import DownTheDrain


MASTER = 0

"""Start a PAW calculation and listen for commands sent through a socket."""


class SocketStringIO(StringIO.StringIO):
    def __init__(self, sckt):
        StringIO.StringIO.__init__(self)
        self.sckt = sckt

    def flush(self):
        send(self.sckt, pickle.dumps(('output', self.getvalue()), -1))
        self.truncate(0)


def run(host,port):
    if mpi.rank == MASTER:
        # Establish socket connection:
        sckt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sckt.connect((host, port))
        # Get pickled arguments:
        string = recv(sckt)
        mpi.broadcast_string(string)
        output = SocketStringIO(sckt)
    else:
        output = DownTheDrain()
        string = mpi.broadcast_string()

    args = pickle.loads(string)
    # Start a PAW calculator:
    paw = create_paw_object(output, *args)

    if mpi.rank == MASTER:
        send(sckt, 'Got your arguments - now give me some commands')

    # Wait for commands and delegate them to Paw objects:
    while True:
        if mpi.rank == MASTER:
            string = recv(sckt)
            mpi.broadcast_string(string)
        else:
            string = mpi.broadcast_string()

        attr, args, kwargs = pickle.loads(string)
        if args is None:
            # We just need an attribute:
            if mpi.rank == MASTER:
                obj = getattr(paw, attr)
                assert isinstance(obj, (float, int, bool, num.ArrayType))
                send(sckt, pickle.dumps(obj, -1))
            continue

        # We need to call a method:
        if attr == 'Stop':
            break
        method = getattr(paw, attr)
        result = method(*args, **kwargs)
        if mpi.rank == MASTER:
            string = pickle.dumps(('result', result), -1)
            send(sckt, string)

    # Done!
    del paw, method  # ??????!!!!!!

    if mpi.rank == MASTER:
        # Send output:
        string = output.getvalue()
        send(sckt, string)

        if recv(sckt) != 'Got your output - now send me your CPU time':
            raise RuntimeError

    cputime = mpi.world.sum(clock(), MASTER)

    if mpi.rank == MASTER:
        send(sckt, pickle.dumps(cputime, -1))

        sckt.close()
