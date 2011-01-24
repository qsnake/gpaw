import os
import sys
import time
import h5py
import h5py.selections as sel
import _gpaw

import numpy as np

intsize = 4
floatsize = np.array([1], float).itemsize
complexsize = np.array([1], complex).itemsize
itemsizes = {'int': intsize, 'float': floatsize, 'complex': complexsize}

class File(h5py.File):
    """Represents an parallel IO -enabled HDF5 file on disk"""

    def __init__(self, name, mode, comm=None, driver=None, **driver_kwds):
        """Stripped down copy-paste of h5py File __init__ with additional communicator argument.""" 

        plist = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
        plist.set_fclose_degree(h5py.h5f.CLOSE_STRONG)
        if comm:
            _gpaw.h5_set_fapl_mpio(plist.id, comm)

        try:
            # If the byte string doesn't match the default encoding, just
            # pass it on as-is.  Note Unicode objects can always be encoded.
            name = name.encode(sys.getfilesystemencoding())
        except (UnicodeError, LookupError):
            pass

        if mode == 'r':
            self.fid = h5py.h5f.open(name, h5py.h5f.ACC_RDONLY, fapl=plist)
        elif mode == 'w':
            self.fid = h5py.h5f.create(name, h5py.h5f.ACC_TRUNC, fapl=plist)
        else:
            raise ValueError("Invalid mode; must be one of r, w")

        self.id = self.fid  # So the Group constructor can find it.
        h5py.Group.__init__(self, self, '/')

        self._mode = mode

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
        self.dset = self.file.create_dataset(name, shape, type)
        if array is not None:
            self.fill(array, parallel=parallel)

    def fill(self, array, *indices, **kwargs):

        try:
            parallel = kwargs['parallel']
        except KeyError:
            parallel = False

        try:
            write = kwargs['write']
        except KeyError:
            write = True

        if parallel:
            # Create H5P_DATASET_XFER property list
            plist = h5py.h5p.create(h5py.h5p.DATASET_XFER)
            _gpaw.h5_set_dxpl_mpio(plist.id)
        else:
            plist = None

        mshape = array.shape
        mtype = None
        fshape = self.dset.shape        

        # Be careful to pad memory shape with ones to avoid HDF5 chunking
        # glitch, which kicks in for mismatched memory/file selections
        if(len(mshape) < len(fshape)):
            mshape_pad = (1,)*(len(fshape)-len(mshape)) + mshape
        else:
            mshape_pad = mshape
        mspace = h5py.h5s.create_simple(mshape_pad,
                                   (h5py.h5s.UNLIMITED,)*len(mshape_pad))
        if write == False:
            h5py.h5s.select_none(mspace)
        
        if indices is None:
            fspace = h5py.h5s.create_simple(fshape, (h5py.h5s.UNLIMITED,)*len(fshape))
            self.dset.id.write(mspace, fspace, array, mtype, plist)
        else:
            selection = sel.select(fshape, indices)
            for fspace in selection.broadcast(mshape):
                if write == False:
                    h5py.h5s.select_none(fspace)
                self.dset.id.write(mspace, fspace, array, mtype, plist)

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
    
    def get(self, name, *indices, **kwargs):

        try:
            parallel = kwargs['parallel']
        except KeyError:
            parallel = False

        if parallel:
            # Create H5P_DATASET_XFER property list
            plist = h5py.h5p.create(h5py.h5p.DATASET_XFER)
            _gpaw.set_dxpl_mpio(plist.id)
        else:
            plist = None

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
