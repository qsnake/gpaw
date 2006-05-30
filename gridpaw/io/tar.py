import os
import time
import tarfile
import xml.sax

import Numeric as num


itemsizes = {'int': 4, 'float': 8, 'complex': 16}

    
class Writer:
    def __init__(self, name):
        self.dims = {}
        self.files = {}
        self.xml1 = ['<gpaw_io version="0.1" endianness="%s">' %
                     ('big', 'little')[num.LittleEndian]]
        self.xml2 = []
        if os.path.isdir(name):
            os.rename(name, name[:-4] + '.old.gpw')
        self.tar = tarfile.open(name, 'w')
        self.mtime = int(time.time())
        
    def dimension(self, name, value):
        self.dims[name] = value

    def __setitem__(self, name, value):
        self.xml1 += ['  <parameter %-20s value="%s"/>' %
                      ('name="%s"' % name, value)]
        
    def add(self, name, shape, array=None, typecode=None, units=None):
        if array is not None:
            array = num.asarray(array)
            typecode = {num.Int: int,
                        num.Float: float,
                        num.Complex: complex}[array.typecode()]
        self.xml2 += ['  <array name="%s" type="%s">' %
                      (name, typecode.__name__)]
        self.xml2 += ['    <dimension length="%s" name="%s"/>' %
                      (self.dims[dim], dim)
                      for dim in shape]
        self.xml2 += ['  </array>']
        self.shape = [self.dims[dim] for dim in shape]
        size = itemsizes[typecode.__name__]
        size *= num.product([self.dims[dim] for dim in shape])
        self.write_header(name, size)
        if array is not None:
            self.fill(array)

    def fill(self, array):
        self.write(array.tostring())

    def write_header(self, name, size):
        tarinfo = tarfile.TarInfo(name)
        tarinfo.mtime = self.mtime
        tarinfo.size = size
        self.size = size
        self.n = 0
        self.tar.addfile(tarinfo)

    def write(self, string):
        self.tar.fileobj.write(string)
        self.n += len(string)
        if self.n == self.size:
            blocks, remainder = divmod(self.size, tarfile.BLOCKSIZE)
            if remainder > 0:
                self.tar.fileobj.write('\0' * (tarfile.BLOCKSIZE - remainder))
                blocks += 1
            self.tar.offset += blocks * tarfile.BLOCKSIZE
        
    def close(self):
        self.xml2 += ['</gpaw_io>']
        string = '\n'.join(self.xml1 + self.xml2)
        self.write_header('info.xml', len(string))
        self.write(string)
        self.tar.close()


class Reader(xml.sax.handler.ContentHandler):
    def __init__(self, name):
        self.dims = {}
        self.shapes = {}
        self.typecodes = {}
        self.parameters = {}
        xml.sax.handler.ContentHandler.__init__(self)
        self.tar = tarfile.open(name, 'r')
        xml.sax.parse(self.tar.extractfile('info.xml'), self)

    def startElement(self, tag, attrs):
        if tag == 'gpaw_io':
            self.byteswap = ((attrs['endianness'] == 'little')
                             != num.LittleEndian)
        elif tag == 'array':
            name = attrs['name']
            self.typecodes[name] = attrs['type']
            self.shapes[name] = []
            self.name = name
        elif tag == 'dimension':
            n = int(attrs['length'])
            self.shapes[self.name].append(n)
            self.dims[attrs['name']] = n
        else:
            assert tag == 'parameter'
            try:
                value = eval(attrs['value'])
            except (SyntaxError, NameError):
                value = attrs['value'].encode()
            self.parameters[attrs['name']] = value

    def dimension(self, name):
        return self.dims[name]
    
    def __getitem__(self, name):
        return self.parameters[name]

    def get(self, name, *indices):
        fileobj, shape, size, typecode = self.get_file_object(name, indices)
        array = num.fromstring(fileobj.read(size), typecode)
        if self.byteswap:
            array = array.byteswapped()
        array.shape = shape
        if shape == ():
            return array.toscalar()
        else:
            return array
    
    def get_reference(self, name, *indices):
        fileobj, shape, size, typecode = self.get_file_object(name, indices)
        return TarFileReference(fileobj, shape, typecode, self.byteswap)
    
    def get_file_object(self, name, indices):
        typecode = {'int': num.Int,
                    'float': num.Float,
                    'complex': num.Complex}[self.typecodes[name]]
        fileobj = self.tar.extractfile(name)
        n = len(indices)
        shape = self.shapes[name]
        size = num.product(shape[n:]) * itemsizes[self.typecodes[name]]
        offset = 0
        stride = size
        for i in range(n - 1, -1, -1):
            offset += indices[i] * stride
            stride *= shape[i]
        fileobj.seek(offset)
        return fileobj, shape[n:], size, typecode

    def close(self):
        self.tar.close()

class TarFileReference:
    def __init__(self, fileobj, shape, typecode, byteswap):
        self.fileobj = fileobj
        self.shape = shape
        self.typecode = typecode
        self.itemsize = itemsizes[{num.Int: 'int',
                                   num.Float: 'float',
                                   num.Complex: 'complex'}[typecode]]
        self.byteswap = byteswap
        self.offset = fileobj.tell()

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, indices):
        if type(indices) is slice:
            indices = ()
        elif type(indices) is int:
            indices = (indices,)
        n = len(indices)

        size = num.product(self.shape[n:]) * self.itemsize
        offset = self.offset
        stride = size
        for i in range(n - 1, -1, -1):
            offset += indices[i] * stride
            stride *= self.shape[i]
        self.fileobj.seek(offset)
        array = num.fromstring(self.fileobj.read(size), self.typecode)
        if self.byteswap:
            array = array.byteswapped()
        array.shape = self.shape[n:]
        return array
