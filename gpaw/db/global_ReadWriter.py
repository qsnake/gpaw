# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
Specifies parameters to be used to create db files
"""

import os
import sys
import time
import tarfile
import xml.sax
import random

import numpy as npy
from Numeric import *

import gpaw
import gpaw.db
import ase
from ase.version import version as ase_version


#from ase.io.pupynere import NetCDFFile

"""There are different ways to create the xml file containing the data:
   1. On the fly:
      This module provides a Writer class that can be used with python
      (Use Writer)
   2. From an nc/gpaw-file
      Reads an nc-file and extracts the parameters of interest (see 
      dacapo_params listed items below. In the future also other files than gpaw
      created should be able to be read. (Use GPAW2DB)
"""


#intsize = 4
#floatsize = npy.array([1], float).itemsize
#complexsize = npy.array([1], complex).itemsize
#itemsizes = {'int': intsize, 'float': floatsize, 'complex': complexsize}


class Writer:
    def __init__(self, name, calculator):
        """ name is the name of the file $.db
            calculator is the name of the calculator (gpaw, dacapo)
        """
        if calculator=="gpaw":
            from gpaw.db.gpaw_params import gpaw_info
            self.params = gpaw_info()
        elif calculator=="dacapo":
            from gpaw.db.dacapo_params import dacapo_info
            self.params = dacapo_info()
            
        self.no_db_copy=False  # do not copy the current file to the db
        self.ignore_mode=False # used to prevent disabled parameters 
                               # from being written to the file
        self.private="660"     # the privacy mode of the db file
        self.fname = name
        self.dims = {}
        self.print_python_type = True
        self.files = {}
        self.xml1 = ['<db version="0.1" calculator="'+calculator+'" endianness="%s">' %
                     ('big', 'little')[int(npy.little_endian)]]
        self.xml2 = []
        if os.path.isfile(name):
            os.rename(name, name[:-4] + '.old.db')
        #self.tar = tarfile.open(name, 'w')
        #self.tar = tarfile.open(name, 'w:gz')
        self.tar = tarfile.open(name, 'w:bz2')
        self.mtime = int(time.time())
        self.array_closed=True
      
        uname = os.uname()
        self['user']=os.getenv('USER', '???') + '@' + uname[1]
        self['date']=time.asctime()
        self['arch']=uname[4]
        self['pid']=os.getpid()
        self['ase_dir']=os.path.dirname(ase.__file__)
        self['ase_version']=ase_version
        self['numpy_dir']=os.path.dirname(npy.__file__)
        self.params.write_version(self)
        
        self['units']='Angstrom and eV'
        if len(sys.argv)>=1 and os.path.exists(sys.argv[0]):
           self['script_name']=sys.argv[0]
           #read the whole file to memory - can't be that big:
           f = file(sys.argv[0],"r")        
           string = f.read()
           f.close() 
           self.write_header('script.py', len(string))
           self.write(string)
        else:
           self['script_name']='unknown'
        
    def dimension(self, name, value):
        self.dims[name] = value

    def __setitem__(self, name, value):
        self.set(name, value)


    def set(self, name, value, unit=None):
        if not self.params.has_key(name):
           #print "Key \""+name+"\" is unknown or marked to be ignored and therefor not written."
           return

        #for array types use fill
        #numpy.ndarray
        if type(value)==type(array([])):
           if type(value)==type(array([])):
               shape = value.shape
           else:
               shape=None
           dtype = None
           self.add(name, shape, value, dtype,unit)
           return
        elif type(value)==npy.ndarray:
           shape = value.shape
           dtype = value.dtype
           self.add(name, shape, value, dtype,unit)
           return



        if unit!=None:
           u = ' unit="%s" '%unit
        else:
           u = ''
        if self.print_python_type:
           t = ' pythontype="%s"'%self.get_type(value)
        else:
           t = ''
        self.xml1 += ['  <parameter %-20s value="%s"%s/>' %
                        ('name="%s"' % self.params[name]["xml_name"], value, u+t)]
        

    # @return gets the type as a string without the <' 
    def get_type(self, obj):
        tp = str(type(obj))
        start = tp.find("'")
        end   = tp.rfind("'")
        return tp[start+1:end]
        
    def add(self, name, shape, array=None, dtype=None, units=None):
        "shape is either a tuple or an list of strings used to index self.dims"
        if type(shape)==tuple:
            for s in shape:
                if type(s)!=str:
                   self.dimension(str(s), s)
            shape = ["%s"%s for s in shape]

        if not self.params.has_key(name):
           #print "Key is marked to be ignored: "+name
           self.ignore_mode = True
           return

        self.close_array()
        if array is not None:
            array = npy.asarray(array)
        if dtype is None:
            dtype = array.dtype

        if dtype in [int, bool]:
            dtype = npy.int32

        dtype = npy.dtype(dtype)
        self.dtype = dtype

        type_ = dtype.name

        self.array_closed = False
        if self.print_python_type:
           self.xml2 += ['  <array name="%s" pythontype="%s">' % (self.params[name]["xml_name"], type_)]
        else:
           self.xml2 += ['  <array name="%s" type="%s">' % (self.params[name]["xml_name"], type_)]

        self.xml2 += ['    <dimension length="%s" name="%s"/>' %
                      (self.dims[dim], dim)
                      for dim in shape]
        if len(shape)==0:
           lst = array.tolist()
           if type(lst)!=type([]):
              lst = [lst]
           self.xml2 += ['    <dimension length="%s" name="%s"/>' %(len(lst), len(lst))]
        self.shape = [self.dims[dim] for dim in shape]
        if array is not None:
            self.fill(array)

    def fill(self, array, indent=""):
        if self.ignore_mode:
           return
        array = npy.asarray(array)
        shape = array.shape
        self.xml2 += [indent+"    <ar>"]
        if len(shape)>1:
           for a in xrange(len(array)):
               self.fill(array[a],indent+" ")
        else:
           if len(shape)==0:
              toprint=array.tolist()
              if type(toprint)!=type([]):
                 toprint = [toprint]
              if len(toprint)==0:
                 toprint=None
           else:
              toprint=array
           #write the individual elements
           #Please note that there is probably a loss in precision
           if toprint!=None:
              dtype = npy.dtype(array.dtype)  
              if dtype==npy.int32 or dtype==npy.bool or dtype.name.startswith("int"):
                 for a in xrange(len(toprint)):
                     self.xml2 += [indent+'       <el real="%d"/>'%toprint[a]]
              elif dtype==npy.float64 or dtype.name.startswith("float"):
                 for a in xrange(len(toprint)):
                     self.xml2 += [indent+'       <el real="%.20f"/>'%toprint[a]]
              elif dtype==npy.complex128 or dtype.name.startswith("complex"):
                 for a in xrange(len(toprint)):
                     self.xml2 += [indent+'       <el real="%.20f" imag="%.20f"/>'%(toprint[a].real,toprint[a].imag)]
              elif dtype.name.startswith("string"):
                     self.xml2 += [indent+'       <el str="%s"/>'%(''.join(toprint))]
                     #sys.exit(-1)
              else:
                 print "Array format "+str(array.dtype)+" not yet supported."
                 import sys
                 sys.exit(-1)

        self.xml2 += [indent+"    </ar>"]

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
        
    def write_additional_db_params(self, **kwargs):
           if kwargs.get('desc') is None:
              self['desc'] = ""
           if kwargs.get('desc') is None:
              self['keywords'] = ""

           #write all user-keywords to the file
           for k in kwargs.keys():
              if kwargs.get('desc') is not None:
                 self[k] = kwargs.get(k)


    def close_array(self):
        if not self.array_closed:
           self.array_closed = True
           self.ignore_mode  = False
           self.xml2 += ['  </array>']

    def set_db_copy_settings(self, make_db_copy, private):
        self.no_db_copy = not make_db_copy
        self.private = private

    def close(self):
        self.close_array()
        self.xml2 += ['</db>\n']
        string = '\n'.join(self.xml1 + self.xml2)
        self.write_header('info.xml', len(string))
        self.write(string)
        self.tar.close()

        if self.no_db_copy:
           return

        # Copy the file to the public location (if existant) if not there already
        dest = gpaw.db.get_db_path()
        if len(dest)>0 and os.path.exists(dest) and not self.fname.startswith(dest):
          outfile = dest+"/"+str(time.time())+"_"+("%0.6d"%(random.randint(0,999999)))+".db"
          os.system("cp "+self.fname+" "+outfile)
          os.system("chmod "+self.private+" "+outfile)


class Reader(xml.sax.handler.ContentHandler):
    def __init__(self, name):
        self.dims = {}
        self.shapes = {}
        self.dtypes = {}
        self.parameters = {}
        self.arrays = {}
        self.temp_array_stack = None
        self.tmp_curtype = None
        xml.sax.handler.ContentHandler.__init__(self)
        self.tar = tarfile.open(name, 'r')
        f = self.tar.extractfile('info.xml')
        xml.sax.parse(f, self)

    def endElement(self, tag):
        if tag == 'ar':
           old = self.cur_array
           self.cur_array = self.temp_array_stack.pop()
           self.cur_array.append(old)
        elif tag == 'array':
           self.arrays[self.name]=self.temp_array_stack.pop()

    def startElement(self, tag, attrs):
        if tag == 'db':
            self.byteswap = ((attrs['endianness'] == 'little')
                             != npy.little_endian)
        elif tag == 'array':
            self.name = get_inv(attrs['name'])["local_name"]
            self.tmp_curtype = self.dtypes[self.name] = attrs['pythontype']
            self.shapes[self.name] = []
            self.temp_array_stack = []
            self.cur_array = self.temp_array_stack

        elif tag == 'ar':
            #pdb.set_trace()
            t = []
            self.temp_array_stack.append(self.cur_array)
            self.cur_array = t
        elif tag == 'el':
            #pdb.set_trace()
            if self.tmp_curtype=="float":
               real = float(attrs['real'])
               self.cur_array.append(real)
            elif self.tmp_curtype=="int":
               real = int(attrs['real'])
               self.cur_array.append(real)
            elif self.tmp_curtype=="complex":
               num = complex(float(attrs['real']),float(attrs['imag']))
               self.cur_array.append(num)

        elif tag == 'dimension':
            n = int(attrs['length'])
            self.shapes[self.name].append(n)
            self.dims[self.name] = n
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

    def has_array(self, name):
        return name in self.shapes
    
    def get(self, name, *indices): 
        shape, dtype = self.get_file_object(name, indices)
        array = npy.asarray(self.arrays[get_inv(name)["local_name"]])
        if self.byteswap:
            array = array.byteswap()
        if dtype == npy.int32:
            array = npy.asarray(array, int)
        array.shape = shape
        if shape == ():
            return array.item()
        else:
            return array
    
    def get_reference(self, name, *indices):
        shape, dtype = self.get_file_object(name, indices)
        name = get_inv(name)["local_name"]
        assert dtype != npy.int32
        return npy.asarray(name)

    def get_file_object(self, name, indices):
        name = get_inv(name)["local_name"]
        dtype = npy.dtype({'int': npy.int32,
                           'float': float,
                           'complex': complex}[self.dtypes[name]])
        n = len(indices)
        shape = self.shapes[name]
        return shape[n:], dtype

    def get_parameters(self):
        return self.parameters

    def close(self):
        self.tar.close()

class TarFileReference:
    def __init__(self, fileobj, shape, dtype, byteswap):
        self.fileobj = fileobj
        self.shape = tuple(shape)
        self.dtype = dtype
        self.itemsize = dtype.itemsize
        self.byteswap = byteswap
        self.offset = fileobj.tell()

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            indices = ()
        elif isinstance(indices, int):
            indices = (indices,)
        n = len(indices)

        size = npy.prod(self.shape[n:], dtype=int) * self.itemsize
        offset = self.offset
        stride = size
        for i in range(n - 1, -1, -1):
            offset += indices[i] * stride
            stride *= self.shape[i]
        self.fileobj.seek(offset)
        array = npy.fromstring(self.fileobj.read(size), self.dtype)
        if self.byteswap:
            array = array.byteswap()
        array.shape = self.shape[n:]
        return array

    def __array__(self):
        return self[::]
