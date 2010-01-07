import os
import time
import h5py

import numpy as np

intsize = 4
floatsize = np.array([1], float).itemsize
complexsize = np.array([1], complex).itemsize
itemsizes = {'int': intsize, 'float': floatsize, 'complex': complexsize}

class Writer:
    def __init__(self, name):
        self.dims = {}        
        if os.path.isfile(name):
            os.rename(name, name[:-5] + '.old'+name[-5:])

        self.file = h5py.File(name, 'w')
        self.dims_grp = self.file.create_group("Dimensions")
        self.params_grp = self.file.create_group("Parameters")
        self.file.attrs['title'] = 'gpaw_io version="0.1"'
        
    def dimension(self, name, value):
        if name in self.dims.keys() and self.dims[name] != value:
            raise Warning('Dimension %s changed from %s to %s' % \
                          (name, self.dims[name], value))
        self.dims[name] = value
        self.dims_grp.attrs[name] = value

    def __setitem__(self, name, value):
        self.params_grp.attrs[name] = value

    def add(self, name, shape, array=None, dtype=None, units=None):
        if array is not None:
            array = np.asarray(array)

        self.dtype, type, itemsize = self.get_data_type(array, dtype)
        shape = [self.dims[dim] for dim in shape]
        if not shape:
            shape = [1,]
        self.dset = self.file.create_dataset(name, shape, type)
        if array is not None:
            self.fill(array)

    def fill(self, array, *indices):
        if indices is None:
            self.dset[:] = array
        else:
            self.dset[indices] = array

    def get_data_type(self, array=None, dtype=None):
        if dtype is None:
            dtype = array.dtype

        if dtype in [int, bool]:
            dtype = np.int32

        dtype = np.dtype(dtype)
        type = {np.int32: 'int',
                np.float64: 'float',
                np.complex128: 'complex'}[dtype.type]

        return dtype, type, dtype.itemsize

    def append(self, name):
        self.file = h5py.File(name, 'a')


    def close(self):
        mtime = int(time.time())
        self.file.attrs['mtime'] = mtime
        self.file.close()
        
class Reader:
    def __init__(self, name):
        self.file = h5py.File(name, 'r')
        self.params_grp = self.file['Parameters']

    def dimension(self, name):
        dims_grp = self.file['Dimensions']
        return dims_grp.attrs[name]
    
    def __getitem__(self, name):
        value = self.params_grp.attrs[name]
        try:
            value = eval(value, {})
        except (SyntaxError, NameError, TypeError):
            pass
        return value

    def has_array(self, name):
        return name in self.file.keys()
    
    def get(self, name, *indices):
        dset = self.file[name]
        array = dset[indices]
        if array.shape == ():
            return array.item()
        else:
            return array

    def get_reference(self, name, *indices):
        dset = self.file[name]
        array = dset[indices]
        return array

    def get_parameters(self):
        return self.params_grp.attrs
    
    def close(self):
        self.file.close()
