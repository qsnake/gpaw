# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import xml.sax
import gpaw.db.global_ReadWriter as db
import gpaw.db
import os, time, random

class Writer:
    """ This class is a wrapper to the db output writer 
        and intended to be used with gpaw
    """
    def __init__(self, name):
        self.writer = db.Writer(name, "gpaw")
        
    def dimension(self, name, value):
        self.writer.dimension(name, value)

    def __setitem__(self, name, value):
        self.writer.__setitem__(name, value)
        
    def add(self, name, shape, array=None, dtype=None, units=None):
        self.writer.add(name, shape, array, dtype,units)


    def fill(self, array):
        self.writer.fill(array)

    def write_header(self, name, size):
        self.writer.write_header(name, size)

    #def write(self, string):
    #    self.writer.write(string)

    def set_db_copy_settings(self, make_db_copy, private):
        self.writer.set_db_copy_settings(make_db_copy, private)

    def write(self, string, db, private, **kwargs):
        self.writer.write(string, db, private, **kwargs)

    def write_additional_db_params(self, **kwargs):
        self.writer.write_additional_db_params(**kwargs)
        
    def close(self):
        self.writer.close()

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

