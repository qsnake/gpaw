import Numeric as num
import Scientific.IO.NetCDF as NetCDF


class Writer:
    def __init__(self, filename):
        self.nc = NetCDF.NetCDFFile(filename, 'w')
        self.dimension('unlim', None)
        var = self.nc.createVariable('FrameNumber', num.Int, ('unlim',))
        var.once = 0
        var[0] = 0

    def dimension(self, name, value):
        self.nc.createDimension(name, value)

    def __setitem__(self, name, value):
        setattr(self.nc, name, value)

    def add(self, name, shape, array=None, typecode=None, units=None):
        if array is not None:
            array = num.asarray(array)
            tc = array.typecode()
        else:
            tc = {int: num.Int,
                  float: num.Float,
                  complex: num.Complex}[typecode]
        if typecode is complex:
            if 'two' not in self.nc.dimensions:
                self.dimension('two', 2)
            var = self.nc.createVariable(name, num.Float, shape + ('two',))
        else:
            var = self.nc.createVariable(name, tc, shape)
        if units is not None:
            var.units = units
            var.once = 1
        if array is not None:
            if shape == ():
                var.assignValue(array)
            else:
                if typecode is complex:
                    var[:, 0] = array.real
                    var[:, 1] = array.imag
                else:
                    var[:] = array
        else:
            self.var = var
            self.i = 0
            self.typecode = typecode

    def fill(self, array):
        i = self.i
        indices = ()
        shape = self.var.shape
        n = len(shape) - len(array.shape)
        if self.typecode is complex:
            n -= 1
        for m in range(n - 1, 0, -1):
            j = i % shape[m]
            indices = (j,) + indices
            i = (i - j) / shape[m]
        indices = (i,) + indices
        if self.typecode is complex:
            self.var[indices + (Ellipsis, 0)] = array.real
            self.var[indices + (Ellipsis, 1)] = array.imag
        else:
            self.var[indices] = array
        self.i += 1

    def close(self):
        self.nc.sync()
        self.nc.close()

class Reader:
    def __init__(self, filename):
        self.nc = NetCDF.NetCDFFile(filename, 'r')

    def dimension(self, name):
        return self.nc.dimensions[name]
    
    def __getitem__(self, name):
        value = getattr(self.nc, name)
        if isinstance(value, str):
            return value
        else:
            return value[0]

    def has_array(self, name):
        return name in self.nc.variables
    
    def get(self, name, *indices):
        var = self.nc.variables[name]
        if var.shape == ():
            return var.getValue()
        else:
            if var.dimensions[-1] == 'two':
                x = var[indices]
                array = num.empty(x.shape[:-1], num.Complex)
                array.real = x[..., 0]
                array.imag = x[..., 1]
                return array
            else:
                return var[indices]

    def get_reference(self, name, *indices):
        return NetCDFReference(self.nc.variables[name], indices)
    
    def close(self):
        self.nc.close()


class NetCDFReference:
    def __init__(self, var, indices):
        self.var = var
        self.indices = indices
        self.cmplx = (var.dimensions[-1] == 'two')
        n = len(indices)
        if self.cmplx:
            self.shape = var.shape[n:-1]
        else:
            self.shape = var.shape[n:]

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if self.cmplx:
            x = self.var[self.indices + indices]
            array = num.zeros(x.shape[:-1], num.Complex)
            array.real = x[..., 0]
            array.imag = x[..., 1]
            return array
        else:
            return self.var[self.indices + indices]
