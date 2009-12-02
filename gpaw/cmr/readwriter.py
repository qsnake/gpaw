# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os
import time
import xml.sax

import numpy as np

import ase
from ase.version import version as ase_version
import gpaw
from gpaw.version import version as gversion

import cmr
from cmr.io import XMLData
from cmr.static import *

class Writer:
    """ This class is a wrapper to the db output writer
    and intended to be used with gpaw
    """
    def __init__(self, filename):
        self.verbose = False
        self.data = XMLData()
        self.data.set_calculator_name(CALCULATOR_GPAW)
        self.split_array = None #used when array is not filled at once
        self.dimensions = {}
        self.filename = filename

        uname = os.uname()
        self.data['user']=os.getenv('USER', '???')
        self.data['date']=time.asctime()

        self.data['arch']=uname[4]
        self.data['ase_dir']=os.path.dirname(ase.__file__)
        self.data['ase_version']=ase_version
        self.data['numpy_dir']=os.path.dirname(np.__file__)
        self.data['gpaw_dir']=os.path.dirname(gpaw.__file__)
        #self.data['calculator_version']=gversion
        self.data['calculator']="gpaw"
        self.data['location']=uname[1]

        
    def dimension(self, name, value):
        self.dimensions[name]=value
        if self.verbose:
            print "dimension: ", name, value

    def __setitem__(self, name, value):
        if self.verbose:
            print "name value:", name, value
        self.data[name]=value
        
    def _get_dimension(self, array):
        """retrieves the dimension of a multidimensional array
        by using then len() function until fail
        """
        indent = ""
        measured_dimensions = []
        while True:
            try:
                measured_dimensions.append(len(array))
                if self.verbose:
                    print indent+"Length:", len(array)
                indent += " "
                array = array[0]
            except TypeError:
                break
        return measured_dimensions

    def _close_array(self):
        """if an array is filled with fill then we don't know
        the exact end of the array. Therefore we check before
        adding a new variable if the end was reached."""
        if self.split_array is None:
            return

        measured_dimensions = self._get_dimension(self.split_array[2])
        if self.verbose:
            print "Dimensions:         ", self.split_array[1]
            print "Mesured Dimensions: ", measured_dimensions

        if measured_dimensions!=self.split_array[1]:
            #make the array fit (only fixes the 1st dimension)
            
            if measured_dimensions[0]==((self.split_array[1])[0]*\
                                         (self.split_array[1])[1]):
                tmp = []
                count = 0
                for a in range(0, (self.split_array[1][0])):
                    tmp.append([])
                    for b in range(0, (self.split_array[1][1])):
                        tmp[a].append(self.split_array[2][count])
                        count += 1
                nd = self._get_dimension(tmp)
                if self.verbose:
                    print "Fixed array dimensions:", nd
            else:
                print "Cannot fix dimensions."
                sys.exit(0)
            
        self.data[self.split_array[0]]=self.split_array[2]

    def add(self, name, shape, array=None, dtype=None, units=None):
        self._close_array()
        if self.verbose:
            print "add:", name, shape, array, dtype, units
        if array is None:
            dimension = []
            for a in shape:
                dimension.append(self.dimensions[a])
            self.split_array = (name, dimension, [])
        else:
            self.data[name]=array

    def fill(self, array):
        if self.verbose:
            print "fill (", len(array),"):", array
        self.split_array[2].append(array)

    def set_db_copy_settings(self, make_db_copy, private):
        if self.verbose:
            print "set_db_copy_settings", make_db_copy, private
        self.db_copy=make_db_copy
        self.private=private
        self.data.set_db_copy_settings(make_db_copy, private)

    def write(self, string, db, private, **kwargs):
        if self.verbose:
            print "write():", string
        self.data.write(string, db, private, **kwargs)

    def write_additional_db_params(self, **kwargs):
        cmr.set_params_to_xml_data(self.data, kwargs)
        
    def close(self):
        if self.verbose:
            print "close()"
        self._close_array()
        self.data.write(self.filename, self.db_copy, self.private)

class Reader(xml.sax.handler.ContentHandler):
    """ This class is a wrapper to the db input reader
        and intended to be used with gpaw
    """
    def __init__(self, name):
        self.reader = db.Reader(name)

    def startElement(self, tag, attrs):
        self.reader.startElement(tag, attrs)

    def dimension(self, name):
        return self.reader.dimension(name)
    
    def __getitem__(self, name):
        return self.reader.__getitem__(name)

    def has_array(self, name):
        return self.reader.has_array(name)
    
    def get(self, name, *indices):
        return self.reader.get(name, *indices)
    
    def get_reference(self, name, *indices):
        return self.reader.get_reference(name, *indices)
    
    def get_file_object(self, name, indices):
        return self.reader.get_file_object(name, indices)

    def get_parameters(self):
        return self.reader.get_parameters()        

    def close(self):
        self.reader.close()

