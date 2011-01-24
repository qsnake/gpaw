import os
import sys
import time
from hdf5_highlevel import File, selection_from_list

import numpy as np

intsize = 4
floatsize = np.array([1], float).itemsize
complexsize = np.array([1], complex).itemsize
itemsizes = {'int': intsize, 'float': floatsize, 'complex': complexsize}

class Writer:
    def __init__(self, name, comm=None):
        self.dims = {}        
        try:
           if comm.rank == 0:
               if os.path.isfile(name):
                   os.rename(name, name[:-5] + '.old'+name[-5:])
           comm.barrier()
        except AttributeError:
           if os.path.isfile(name):
               os.rename(name, name[:-5] + '.old'+name[-5:])

        self.file = File(name, 'w', comm)
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

    def add(self, name, shape, array=None, dtype=None, units=None, 
            parallel=False):
        if array is not None:
            array = np.asarray(array)

        self.dtype, type, itemsize = self.get_data_type(array, dtype)
        shape = [self.dims[dim] for dim in shape]
        if not shape:
            shape = [1,]
        self.dset = self.file.create_dataset(name, type, shape)
        if array is not None:
            self.fill(array, parallel=parallel)

    def fill(self, array, indices, **kwargs):

        try:
            parallel = kwargs['parallel']
        except KeyError:
            parallel = False

        try:
            write = kwargs['write']
        except KeyError:
            write = True

        if parallel:
            collective = True
        else:
            collective = False

        if write:
            self.dset.write(array, indices, collective)            
        else:
            self.dset.write(array, None, collective)            

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
    def __init__(self, name, comm=False):
        self.file = File(name, 'r')
        self.params_grp = self.file['Parameters']
        self.hdf5_reader = True

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
    
    def get(self, name, indices, **kwargs):

        try:
            parallel = kwargs['parallel']
        except KeyError:
            parallel = False

        if parallel:
            collective = True
        else: 
            collective = False

        dset = self.file[name]
        offset, stride, count = selection_from_list(indices)
        mshape = tuple(count)
        array = np.empty(mshape, dset.dtype)
        dset.read(array, indices, collective)
        if array.shape == ():
            return array.item()
        else:
            return array

    def get_reference(self, name, indices):
        dset = self.file[name]
        array = dset[indices]
        return array

    def get_parameters(self):
        return self.params_grp.attrs
    
    def close(self):
        self.file.close()
