# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os
import time
import xml.sax

import numpy as np

import ase
from ase.version import version as ase_version
import gpaw

import cmr
from cmr.io import XMLData
from cmr.io import READ_DATA
from cmr.io import EVALUATE
from cmr.io import WRITE_CONVERT
from cmr.io import CONVERTION_ORIGINAL
from cmr.static import CALCULATOR_GPAW


class Writer:
    """ This class is a wrapper to the db output writer
    and intended to be used with gpaw
    """
    def __init__(self, filename):
        self.verbose = False
        self.data = XMLData()
        self.data.set_calculator_name(CALCULATOR_GPAW)
        self.data.set_write_mode(WRITE_CONVERT)
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
        self.data[name]=value
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
            
        #make the array fit (only fixes the 1st dimension)
        res = np.array(self.split_array[2]).reshape(self.split_array[1])
        self.data[self.split_array[0]] = res

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

    def fill(self, array, *indices):
        if self.verbose:
            print "fill (", len(array),"):", array
        self.split_array[2].append(array)

    def set_db_copy_settings(self, make_db_copy, private):
        """deprecated: This method is going to be removed"""
        print "Warning: gpaw.cmr.readwriter.set_db_copy_settings is deprecated."
        print "Please update gpaw and CMR"
        pass
        #if self.verbose:
        #    print "set_db_copy_settings", make_db_copy, private
        #self.db_copy=make_db_copy
        #self.private=private
        #self.data.set_db_copy_settings(make_db_copy, private)

    def write(self, string, db, private, **kwargs):
        print "Warning: gpaw.cmr.readwriter.write is deprecated."
        print "Please update gpaw and CMR"
        pass
        #if self.verbose:
        #    print "write():", string
        #self.data.write(string, db, private, **kwargs)

    def write_additional_db_params(self, cmr_params):
        """writes the user variables and also sets the write attributes for
        the output file"""
        self.cmr_params = cmr_params.copy()
        cmr.set_params_to_xml_data(self.data, cmr_params)
        
    def close(self):
        if self.verbose:
            print "close()"
        self._close_array()
        self.cmr_params["output"]=self.filename
        self.data.write(self.cmr_params)

class Reader:
    """ This class allows gpaw to access
    to read a db-file
    """
    def __init__(self, name):
        self.reader = cmr.read(name,
                               read_mode=READ_DATA,
                               evaluation_mode=EVALUATE,
                               convertion_mode=CONVERTION_ORIGINAL)
        self.parameters = self.reader

    def dimension(self, name):
        return self.reader[name]
    
    def __getitem__(self, name):
        return self.reader[name]

    def has_array(self, name):
        return self.reader.has_key(name)
    
    def get(self, name, *indices):
        print "incides", indices
        result = self.reader[name]
        if indices!=():
            for a in indices:
                result = result[a]
            return result
        
        #gpaw wants expressions evaluated
        if type(result)==str or type(result)==unicode:
            try:
                print "Converting ", result
                result = eval(result, {})
            except (SyntaxError, NameError):
                pass
        return result
    
    def get_reference(self, name, *indices):
        result = self.reader[name]
        if indices!=():
            for a in indices:
                result = result[a]
            return result
    
    def get_file_object(self, name, indices):
        result = self.reader.retrieve_file(name)
        if indices!=():
            for a in indices:
                result = result[a]
        return result
        

    def get_parameters(self):
        return self.reader.keys()

    def close(self):
        pass

