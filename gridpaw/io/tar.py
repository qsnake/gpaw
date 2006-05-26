import os
import xml.sax
import tarfile

import Numeric as num

    
class Writer:
    def __init__(self, name):
        self.dims = {}
        self.attrs = {}
        self.files = {}
        self.xml = ['<gpaw_io version="0.1" endianness="%s">' %
                    ('big', 'little')[num.LittleEndian]]
        if os.path.isdir(name):
            os.rename(name, name + '.old')
        self.file = tarfile.open(name, 'w')
        
    def dimension(self, name, value):
        self.dims[name] = length

    def __setitem__(self, name, value):
        self.attrs[name] = value
        
    def add(self, name, shape, array=None, typecode=None, units=None):
        if array is not None:
            typecode = {num.Int: int,
                        num.Float: float,
                        num.Complex: complex}[array.typecode()]
        self.xml += ['  <array name="%s" type="%s">' %
                     (name, typecode.__name__)]
        self.xml += ['   <dimension length="%s" name="%s"/>' %
                         (self.dims[dim], dim)
                         for dim in shape]
        self.xml += ['  </array>']
        self.shape = [self.dims[dim] for dim in shape]
        if array is not None:
            self.fill(array)

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
        self.xmk += ['  <parameter name="%s" value="%s"/>' % (name, value)
                     for name, value in self.parameters.items()]
        self.xml += ['</gpaw_io>']
        open(self.dir + 'content.xml', 'w').write('\n'.join(self.xml))

class Reader(xml.sax.handler.ContentHandler):
    def __init__(self, name):
        self.dims = {}
        self.shapes = {}
        self.typecodes = {}
        self.parameters = {}
        xml.sax.handler.ContentHandler.__init__(self)
        xml.sax.parse(name + '/content.xml', self)

    def startElement(self, tag, attrs):
        if tag == 'gpaw_io':
            self.byteswap = ((attrs['endianness'] == 'little')
                             != num.LittleEndian)
        elif tag == 'array':
            name = attrs['name']
            self.typecodes[name] = attrs['type']
            self.shapes[name] = []
            self.name = name
        elif tag == 'parameter':
            self.parameters[attrs['name']] = eval(attrs['value'])
        else:
            assert tag == 'dimension'
            n = int(attrs['length'])
            self.shapes[self.name].append(n)
            self.dims[attrs['name']] = n

    def dimension(self, name):
        return self.dims[name]
    
    def __getitem__(self, name):
        return self.attrs[name]

    def get(self, name, *indices):
        a = num.fromstring(open(self.dir + name).read(),
                           self.typecodes[name])
        if self.byteswap:
            a = a.byteswapped()
        print a.shape,self.shapes, name
        a.shape = self.shapes[name]
        return a

    def get_reference(self, name, *indices):
        return TarFileReference(self.nc.variables[name], indices)

class TarFileReference:
    def __init__(self, file, shape, typecode, byteswap, indices):
        self.file = file
        self.typecode = typecode
        self.byteswap = byteswap
        strides = [num.zeros(0, typecode).itemsize()]
        for dim in shape[:0:-1]:
            strides.insert(0, dim * strides[0])
        n = len(indices)
        self.offset = num.dot(strides[:n], indices)
        self.strides = strides[n:]
        self.shape = tuple(shape[n:])

    def __getitem__(self, indices):
        if type(indices) is not tuple:
            indices = (indices,)
        n = len(indices)
        if type(indices[-1]) is int:
            self.file.seek(self.offset + num.dot(self.strides[:n], indices))
            a = num.fromstring(self.file.read(self.strides[n - 1]),
                               self.typecode)
        else:
            start, stop = indices[-1].indices(self.shape[n - 1])[:2]
            self.file.seek(self.offset + num.dot(self.strides[:n - 1],
                                                 indices[:-1]) +
                           start * self.strides[n - 1])
            a = num.fromstring(self.file.read((stop - start) *
                                              self.strides[n - 1]),
                               self.typecode)
            n -= 1
        if self.byteswap:
            a = a.byteswapped()
        print a.shape, self.shape, n
        a.shape = self.shape[n:]
        return a
        
