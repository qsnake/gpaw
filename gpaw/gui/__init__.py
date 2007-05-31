import os
import pickle
import tempfile

import numpy as npy

def gui(atoms):
    positions = npy.array(atoms.GetCartesianPositions())
    positions.shape = (1,) + positions.shape
    forces = npy.empty_like(positions)
    forces.fill(npy.nan)
    data = {'cell': npy.array(atoms.GetUnitCell()),
            'periodic': atoms.GetBoundaryConditions(),
            'numbers': npy.array(atoms.GetAtomicNumbers()),
            'positions': positions,
            'energies': npy.array([npy.nan]),
            'forces': forces}
    fd, filename = tempfile.mkstemp('.pckl', 'g2-')
    os.write(fd, pickle.dumps(data))
    os.close(fd)
    os.system('(g2 --read-pickled-data-from-file %s &); (sleep 5; rm %s) &' %
              (filename, filename))
