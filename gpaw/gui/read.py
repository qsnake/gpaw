import pickle
from tarfile import is_tarfile
from zipfile import is_zipfile, ZipFile

import numpy as npy
import Numeric as num
from ASE import ListOfAtoms, Atom
from ASE.ChemicalElements.symbol import symbols
from ASE.Units import units, Convert


def read_from_files(filenames, slice=':'):
    RR = []
    E = []
    F = []
    first = True
    for filename in filenames:
        images, energies, forces, dft = read_from_file(filename, slice)
        if first:
            Z = npy.array(images[0].GetAtomicNumbers())
            cell = npy.array(images[0].GetUnitCell())
            periodic = images[0].GetBoundaryConditions()
            dft0 = dft
        else:
            del dft
            if (images[0].GetAtomicNumbers() != Z).any():
                raise ValueError('sdfasdf')
        RR += [npy.array(image.GetCartesianPositions()) for image in images]
        E += energies
        F += forces

    if None in E:
        E = npy.empty(len(E))
        E[:] = npy.nan
    else:
        E = npy.array(E)
    if None in F:
        F = npy.empty((len(F), len(Z), 3))
        F[:] = npy.nan
    else:
        F = npy.array(F)
    return cell, periodic, Z, dft0, npy.array(RR), E, F

def read_from_file(filename, slice):
    p = filename.rfind('@')
    if p != -1:
        slice = filename[p + 1:]
        filename = filename[:p]

    type = 'text'
    if is_tarfile(filename):
        type = 'gpaw'
    elif open(filename).read(3) == 'CDF':
        from Scientific.IO.NetCDF import NetCDFFile
        nc = NetCDFFile(filename)
        if 'number_of_dynamic_atoms' in nc.dimensions:
            type = 'dacapo'
        else:
            history = nc.history
            if history == 'GPAW restart file':
                type = 'gpaw'
            elif history == 'ASE trajectory':
                type = 'ase'
            elif history == 'Dacapo':
                type = 'dacapo'
            else:
                raise IOError('Unknown netCDF file!')
    elif is_zipfile(filename):
        type = 'vnl'
        
    if type == 'gpaw':
        from gpaw import Calculator
        atoms = Calculator.ReadAtoms(filename, out=None)
        e = atoms.GetPotentialEnergy()
        f = atoms.GetCartesianForces()
        return [atoms], [e], [f], {'calc': atoms.GetCalculator()}

    if type == 'ase':
        from ASE.Trajectories.NetCDFTrajectory import NetCDFTrajectory
        traj = NetCDFTrajectory(filename)
        indices = range(len(traj))
        indices = eval('indices[%s]' % slice)
        if isinstance(indices, int):
            indices = [indices]
        images = []
        energies = []
        forces = []
        for i in indices:
            image = traj.GetListOfAtoms(i)
            try:
                e = image.GetPotentialEnergy()
            except KeyError:
                e = None
            try:
                f = image.GetCartesianForces()
            except KeyError:
                f = None
            image.SetCalculator(None)
            images.append(image)
            energies.append(e)
            forces.append(f)
        return images, energies, forces, {}
            
    if type == 'dacapo':
        from Dacapo import Dacapo
        atoms = Dacapo.ReadAtoms(filename)
        e = atoms.GetPotentialEnergy()
        f = atoms.GetCartesianForces()
        calc = atoms.GetCalculator()
        dft = {'type': 'Dacapo',
               'xc': calc.GetXCFunctional(),
               'nbands': calc.GetNumberOfBands()}
        atoms.SetCalculator(None)
        return [atoms], [e], [f], dft

    if type == 'vnl':
        return read_vnl(filename)
    
    file = open(filename)
    lines = file.readlines()

    if lines[1] == '  ___ ___ ___ _ _ _  \n':
        images, energies, forces, dft = read_gpaw_text(lines)

    elif (' &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n'
          in lines[:90]):
        images, energies, forces, dft = read_dacapo_text(lines)

    elif lines[1].startswith('OUTER LOOP:'):
        images, energies, forces, dft = read_cube(lines)

    else:
        L1 = lines[0].split()
        if len(L1) == 1:
            del lines[:2]
            natoms = int(L1[0])
        else:
            natoms = len(lines)
        images = []
        energies = []
        forces = []
        while len(lines) > 0:
            atoms = ListOfAtoms([])
            for line in lines[:natoms]:
                symbol, x, y, z = line.split()
                atoms.append(Atom(symbol, [float(x), float(y), float(z)]))
            del lines[:natoms + 2]
            images.append(atoms)
            energies.append(None)
            forces.append(None)
        dft = {}
        
    indices = range(len(images))
    indices = eval('indices[%s]' % slice)
    if isinstance(indices, int):
        indices = [indices]
    return ([images[i] for i in indices],
            [energies[i] for i in indices],
            [forces[i] for i in indices],
            dft)

def read_gpaw_text(lines):
    i = lines.index('unitcell:\n')
    cell = [float(line.split()[2]) for line in lines[i + 3:i + 6]]
    images = []
    energies = []
    forces = []
    while True:
        try:
            i = lines.index('Positions:\n')
        except ValueError:
            break
        atoms = ListOfAtoms([], cell=cell)
        for line in lines[i + 1:]:
            words = line.split()
            if len(words) != 5:
                break
            n, symbol, x, y, z = words
            symbol = symbol.split('.')[0]
            atoms.append(Atom(symbol, [float(x), float(y), float(z)]))
        try:
            i = lines.index('-------------------------\n')
        except ValueError:
            e = None
        else:
            line = lines[i + 9]
            assert line.startswith('zero Kelvin:')
            e = float(line.split()[-1])
        if i + 15 < len(lines) and lines[i + 15].startswith('forces'):
            f = []
            for i in range(i + 15, i + 15 + len(atoms)):
                x, y, z = lines[i].split('[')[1][:-2].split()
                f.append((float(x), float(y), float(z)))
        else:
            f = None

        if len(images) > 0 and e is None:
            break

        images.append(atoms)
        energies.append(e)
        forces.append(f)
        lines = lines[i:]
        
    return images, energies, forces, {}

def read_dacapo_text(lines):
    i = lines.index(' Structure:             A1           A2            A3\n')
    cell = npy.array([[float(w) for w in line.split()[2:5]]
                      for line in lines[i + 1:i + 4]]).transpose()
    i = lines.index(' Structure:  >>         Ionic positions/velocities ' +
                    'in cartesian coordinates       <<\n')
    atoms = ListOfAtoms([], cell=cell.tolist())
    for line in lines[i + 4:]:
        words = line.split()
        if len(words) != 9:
            break
        Z, x, y, z = words[2:6]
        atoms.append(Atom(Z=int(Z), position=(float(x), float(y), float(z))))
    return [atoms], [None], [None], {}

def read_cube(lines):
    n = int(lines[2].split()[0])
    a0 = Convert(1, 'Bohr', units.GetLengthUnit())
    cell = []
    for line in lines[3:6]:
        g, x, y, z = line.split()
        g = int(g) * a0
        cell.append((g * float(x), g * float(y), g * float(z)))
    atoms = ListOfAtoms([], cell=cell, periodic=True)
    for line in lines[6:6 + n]:
        Z, a, x, y, z = line.split()
        atoms.append(Atom(Z=Z, position=(a0 * float(x),
                                         a0 * float(y),
                                         a0 * float(z))))
    return [atoms], [None], [None], {}


class VNL:
    def __setstate__(self, data):
        self.data = data

def ac(shape, typecode, data, endian):
    x = num.fromstring(data, typecode)
    try:
        x.shape = shape
    except ValueError:
        x = x[::2].copy()
        x.shape = shape
        
    if num.LittleEndian != endian: 
        return x.byteswapped() 
    else: 
        return x 

class VNLUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'VNLATKStorage.Core.Sample':
            return VNL
        if name == 'array_constructor':
            return ac
        return pickle.Unpickler.find_class(self, module, name)
    
def read_vnl(filename):
    from cStringIO import StringIO
    vnl = VNLUnpickler(StringIO(ZipFile(filename).read('0_object'))).load()
    conf = vnl.data['__properties__']['Atomic Configuration'].data
    numbers = conf['_dataarray_']
    positions = conf['_positions_'].data['_dataarray_']
    atoms = ListOfAtoms([Atom(Z=Z) for Z in numbers])
    atoms.SetCartesianPositions(positions)
    return [atoms], [None], [None], {}
