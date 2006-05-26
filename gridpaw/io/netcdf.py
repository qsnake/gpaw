import Scientific.IO.NetCDF as NetCDF


class Writer:
    def __init__(self, filename):
        self.nc = NetCDF.NetCDFFile(filename, 'w')

    def dimension(self, name, value):
        self.nc.createDimension(name, value)

    def __setitem__(self, name, value):
        setattr(self.nc, name, value)

    def add(self, name, typecode, shape, array=None, units=None):
        tc = {int: num.Int, float: num.Float, complex: num.Complex}[typecode]
        var = self.nc.createVariable(name, tc, shape)
        if units is not None:
            var.units = units
            var.once = 1
        if array is not None:
            var[:] = array
        else:
            self.var = var
            self.i = 0

    def fill(self, array):
        i = self.i
        indices = ()
        shape = self.var.shape
        n = len(shape) - len(array.shape)
        for m in range(n - 1, 0, -1):
            j = i % shape[m]
            indices = (j,) + indices
            i = (i - j) / shape[m]
        indices = (i,) + indices
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
        return getattr(self.nc, name)

    def get(self, name, *indices):
        return self.nc.variables['name'][indices]

    def get_reference(self, name, *indices):
        return NetCDFReference(self.nc.variables[name], indices)
    
class NetCDFReference:
    def __init__(self, var, indices):
        self.psit_unG = psit_unG
        self.u = (s, k)
        self.scale = scale
        self.cmplx = cmplx
        netcdfshape = self.psit_unG.shape
        if self.cmplx:
            self.shape = netcdfshape[2:-1]
        else:
            self.shape = netcdfshape[2:]

    def __len__(self):
        return self.shape[0]
            
    def __getitem__(self, index):
        if type(index) is not tuple:
            index = (index,)
        if self.cmplx:
            try: 
                a = self.psit_unG[self.u + index]
            except IOError: 
                raise IndexError
            w = num.zeros(a.shape[:-1], num.Complex)
            w.real = a[..., 0]
            w.imag = a[..., 1]
            w *= self.scale
            return w
        else:
            try: 
                return self.scale * self.psit_unG[self.u + index]
            except IOError:
                raise IndexError
