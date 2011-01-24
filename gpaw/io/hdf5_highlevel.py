"""Light weight Python interface to HDF5, inspired largely by h5py"""

#  Copyright (C) 2010       CSC - IT Center for Science Ltd.
#  Please see the accompanying LICENSE file for further information.


import numpy as np
from _hdf5 import *

def numpy_type_from_h5(datatype):
    """Simple conversion from HDF5 datatype to NumPy dtype"""
    cls = h5t_get_class(datatype)
    if cls == H5T_INTEGER:
        dtype = int
    elif cls == H5T_FLOAT:
        dtype = float
    elif cls == H5T_COMPOUND:
        dtype = complex
    elif cls == H5T_STRING:
        str_size = h5t_get_size(datatype)
        dtype = np.dtype('|S' + str(str_size))
    else:
        raise RuntimeError('Unsupported HDF5 datatype')

    return dtype

def selection_from_list(indices, shape):
    """Convert Python style slice: 
         start, stop, step 
       to the HDF5 style hyperslab:
         offset, stride, count

       The block parameter of HDF5 hyperslab is not used here"""
       
    if len(indices) != len(shape):
        raise RuntimeError('Invalid selection')

    offset = []
    stride = []
    count = []
    for ind, dim in zip(indices, shape):
        if isinstance(ind, int):
            offset.append(ind)
            stride.append(1)
            count.append(1)
        elif isinstance(ind, slice):
            start, stop, step = ind.start, ind.stop, ind.step
            start = 0 if start is None else int(start)
            stop = dim if stop is None else int(stop)
            step = 1 if step is None else int(step)
            c = (stop-start)//step
            if (stop-start) % step != 0:
                c += 1
            offset.append(start)
            stride.append(step)
            count.append(c)

    return np.array(offset), np.array(stride), np.array(count)
                


class Group:
    def __init__(self, loc_id, name, create=False):
        if create:
            self.id = h5g_create(loc_id, name)
        else:
            self.id = h5g_open(loc_id, name)

        self.attrs = Attributes(self.id)

        self.opened = True

    def create_group(self, name):
        return Group(self.id, name, create=True)

    def create_dataset(self, name, dtype, shape):
        """Create a dataset with the NumPy equivalent type type
           and shape"""

        return Dataset(self.id, name, dtype, shape, create=True)

    def close(self):
        h5g_close(self.id)
        self.opened = False

    def __getitem__(self, key):
        oid = h5o_open(self.id, key)
        obj_type = h5i_get_type(oid)
        h5o_close(oid)
        if obj_type == H5I_GROUP:
            return Group(self.id, key)
        elif obj_type == H5I_DATASET:
            return Dataset(self.id, key)
        else:
           # This shoul never happen
           raise RuntimeError('Accessing unknown object type')

    def __del__(self):
        if self.opened:
            self.close()

class File(Group):
    """This class defines a HDF5 file, ..."""

    def __init__(self, name, mode='r', comm=None):

        if comm is None:
            plist = H5P_DEFAULT
        else:
            plist = h5p_create(H5P_FILE_ACCESS)
            h5p_set_fapl_mpiio(plist, comm)

        if mode == 'r':
            self.id = h5f_open(name, mode, plist)
        elif mode == 'w':
            self.id = h5f_create(name, plist)
        else:
            raise RuntimeError('Unsupported file open/create mode')

        self.attrs = Attributes(self.id)

        if comm is None:
            h5p_close(plist)

        self.opened = True

    def close(self):
        h5f_close(self.id)
        self.opened = False

class Dataset:
    """This class defines a HDF5 dataset, ..."""

    def __init__(self, loc_id, name, dtype=None, shape=None, create=False):
        if create:
            self.shape = shape
            self.dtype = dtype
            self.dataspace = h5s_create(np.asarray(shape))
            self.datatype = h5_type_from_numpy(np.ndarray((1,), dtype))
            self.id = h5d_create(loc_id, name, self.datatype, self.dataspace)
        else:
            self.id = h5d_open(loc_id, name)
            self.dataspace = h5d_get_space(self.id)
            self.datatype = h5d_get_type(self.id)
            self.shape = h5s_get_shape(self.dataspace)
            self.dtype = numpy_type_from_h5(self.datatype)

        self.attrs = Attributes(self.id)

        self.opened = True

    def write(self, data, selection='all', collective=False):

        if collective:
            plist = h5p_create(H5P_DATASET_XFER)
            h5p_set_dxpl_mpiio(plist)
        else:
            plist = H5P_DEFAULT

        filespace = self.dataspace
        memspace = h5s_create(np.asarray(data.shape))
        memtype = h5_type_from_numpy(np.ndarray((1,), self.dtype))

        if selection is None:
            h5s_select_none(memspace)
            h5s_select_none(filespace)

        if isinstance(selection, list):
            offset, stride, count = selection_from_list(selection, self.shape)
            h5s_select_hyperslab(filespace, offset, stride, count)
            
        h5d_write(self.id, memtype, memspace, filespace, data, plist)
        
        h5s_close(memspace)
        h5t_close(memtype)
        if collective:
            h5p_close(plist)

    def read(self, data, selection='all', collective=False):

        if collective:
            plist = h5p_create(H5P_DATASET_XFER)
            h5p_set_dxpl_mpiio(plist)
        else:
            plist = H5P_DEFAULT

        filespace = self.dataspace
        memspace = h5s_create(np.asarray(data.shape))
        memtype = h5_type_from_numpy(np.ndarray((1,), self.dtype))

        if selection is None:
            h5s_select_none(memspace)
            h5s_select_none(filespace)

        if isinstance(selection, list):
            offset, stride, count = selection_from_list(selection, self.shape)
            h5s_select_hyperslab(filespace, offset, stride, count)

        h5d_read(self.id, memtype, memspace, filespace, data, plist)
        
        h5s_close(memspace)
        h5t_close(memtype)
        if collective:
            h5p_close(plist)

    def close(self):
        h5t_close(self.datatype)
        h5s_close(self.dataspace)
        h5d_close(self.id)
        self.opened = False
        
    def __del__(self):
        if self.opened:
            self.close()

class Attributes:
    """A dictionary like interface to HDF5 attributes.
       Attributes can be written with the 
          attrs['name'] = value
       and read with the
          value = attrs['name'] 
       syntax. 
       Values are returned always as NumPy arrays."""

    def __init__(self, parent_id):
        self.loc_id = parent_id

    def __setitem__(self, key, data):
        # Should we delete existing attributes?
        data = np.asarray(data)
        dataspace = h5s_create(np.asarray(data.shape))
        datatype = h5_type_from_numpy(data)
        id = h5a_create(self.loc_id, key, datatype, dataspace)
        h5a_write(id, datatype, data)
        h5s_close(dataspace)
        h5t_close(datatype)
        h5a_close(id)

    def __getitem__(self, key):
        id = h5a_open(self.loc_id, key)
        dataspace = h5a_get_space(id)
        datatype = h5a_get_type(id)
        shape = h5s_get_shape(dataspace)
        dtype = numpy_type_from_h5(datatype)
        data = np.empty(shape, dtype)
        h5a_read(id, datatype, data)
        h5s_close(dataspace)
        h5t_close(datatype)
        h5a_close(id)

        return data
