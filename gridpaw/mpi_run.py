import pickle
import socket
import StringIO

import gridpaw.utilities.mpi as mpi
from gridpaw.startup import create_paw_object
from gridpaw.utilities.socket import send, recv
from gridpaw.utilities.timing import clock


MASTER = 0


"""Start a PAW calculation and listen for commands send through a
socket."""


class SocketStringIO(StringIO.StringIO):
    def __init__(self, sckt):
        StringIO.StringIO.__init__(self)
        self.sckt = sckt

    def flush(self):
        send(self.sckt, pickle.dumps(('output', self.getvalue())))
        self.truncate(0)


def run(port):
    if mpi.rank == MASTER:
        # Establish socket connection:
        sckt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sckt.connect(('localhost', port))
        # Get pickled arguments:
        string = recv(sckt)
        mpi.broadcast_string(string)
        output = SocketStringIO(sckt)
    else:
        output = None
        string = mpi.broadcast_string()

    args = pickle.loads(string)
    # Start a PAW calculator:
    paw = create_paw_object(output, *args)

    if mpi.rank == MASTER:
        send(sckt, 'Got your arguments - now give me some commands')

    # Wait for commands and delegate them to calculators:
    while True:
        if mpi.rank == MASTER:
            string = recv(sckt)
            mpi.broadcast_string(string)
        else:
            string = mpi.broadcast_string()
        methodname, args, kwargs = pickle.loads(string)

        if methodname == 'Stop':
            break
        method = getattr(paw, methodname)
        result = method(*args, **kwargs)
        if mpi.rank == MASTER:
            string = pickle.dumps(('result', result))
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
        send(sckt, pickle.dumps(cputime))

        sckt.close()
