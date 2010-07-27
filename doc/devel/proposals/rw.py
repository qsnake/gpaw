import sys
import pickle
from StringIO import StringIO

import numpy as np

import gpaw.mpi as mpi


class Writer:
    """Base clas for all GPAW writers."""
    def __init__(self, data=None, child=False, world=mpi.world):
        """Create writer object.

        Sub-writers are created with child=True.

        The data dictionary holds:

        * data for type bool, int, float, complex and str
        * shape and dtype for ndarrays
        * class names for other objects

        These other objects must have a write() method and a static
        read() method."""
        
        if data is None:
            data = {}
        self.data = data
        self.child = child
        self.world = world
        
    def write(self, **kwargs):
        """Write data.

        Use::

            writer.write(n=7, s='abc', a=np.zeros(3), density=density).
        """
        
        for name, value in kwargs.items():
            if isinstance(value, (bool, int, float, str)):
                self.data[name] = value
            elif isinstance(value, np.ndarray):
                self.add_array(name, value.shape, value.dtype)
                self.fill_array(value)
            else:
                self.data[name] = {'_type':
                                   value.__module__ + '.' +
                                   value.__class__.__name__}
                writer = self.subwriter(name)
                value.write(writer)

    def subwriter(self, name):
        """Create subwriter."""
        raise NotImplementedError
    
    def add_array(self, name, shape, dtype=float):
        # XXX Todo: translate to real dtype objects?
        if isinstance(shape, int):
            shape = (shape,)
        self.data[name] = {'_type': 'numpy.ndarray',
                           'shape': shape,
                           'dtype': dtype}

    def fill_array(self, a):
        """Fill in ndarray data."""
        raise NotImplementedError
        
    def close(self):
        if not self.child:
            self.write_data()

    def write_data(self):
        """Write data dictionary.

        Write bool, int, float, complex and str data, shapes and
        dtypes for ndarrays and class names for other objects."""
        
        raise NotImplementedError

class Reader:
    """Base clas for all GPAW writers."""
    def __init__(self, data):
        """Create hierarchy of readers.

        Store data as attributes for easy access and to allow
        tab-completion."""
        
        for name, value in data.items():
            if isinstance(value, dict):
                if value['_type'] == 'numpy.ndarray':
                    del value['_type']
                    value = self.get_ndarray(**value)
                else:
                    value = self.subreader(value)
            setattr(self, name, value)
        
    def subreader(self, value):
        """Create subreader."""
        raise NotImplementedError

    def read(self):
        """Read object."""
        mod, cls = self._type.rsplit('.', 1)
        __import__(mod)
        module = sys.modules[mod]
        return getattr(module, cls).read(self)

    def get_ndarray(self, shape, dtype, **kwargs):
        """Create ndarray wrapper."""
        raise NotImplementedError

        
class Writer1(Writer):
    """Simple prototype implementation."""
    def __init__(self, fd, data=None, child=False, world=mpi.world):
        Writer.__init__(self, data, child, world)
        if self.world.rank == 0:
            if not self.child:
                # Write file format identifier:
                fd.write('GPAW1...')
                # Write dummy data size:
                fd.write('7654321\n')
            self.fd = fd
        else:
            self.fd = None
        
    def subwriter(self, name):
        return Writer1(self.fd, self.data[name], child=True, world=self.world)

    def add_array(self, name, shape, dtype=float):
        Writer.add_array(self, name, shape, dtype)
        if self.world.rank == 0:
            self.data[name]['offset'] = self.fd.tell()
        
    def fill_array(self, a):
        if self.world.rank == 0:
            self.fd.write(a.tostring())

    def write_data(self):
        if self.world.rank == 0:
            i = self.fd.tell()
            pickle.dump(self.data, self.fd)
            self.fd.seek(8)
            # Write size of data chunk:
            self.fd.write('%7d' % i)
            if not isinstance(self.fd, StringIO):
                self.fd.close()
            

class Reader1(Reader):
    """Simple prototype implementation."""
    def __init__(self, fd, data=None):
        self.fd = fd
        if data is None:
            fd.seek(0)
            id = fd.read(8)
            assert id == 'GPAW1...'
            i = int(fd.read(8)[:-1])
            fd.seek(i)
            data = pickle.load(fd)
        Reader.__init__(self, data)
        
    def subreader(self, data):
        return Reader1(self.fd, data)
    
    def get_ndarray(self, shape, dtype, offset):
        self.fd.seek(offset)
        size = dtype.itemsize * np.prod(shape)
        a = np.fromstring(fd.read(size), dtype)
        a.shape = shape
        # XXX Todo: return wrapper instead of ndarray
        return a


class A:
    def write(self, writer):
        writer.write(x=np.ones((2, 3)))

    @staticmethod
    def read(reader):
        a = A()
        a.x = reader.x[:]
        return a

    
fd = StringIO()
w = Writer1(fd)
w.write(a=A(), y=9)
w.write(s='abc')
w.close()
print w.data


r = Reader1(fd)
print r.y, r.s
print r.a.read()
print r.a.x, r.a.read().x
