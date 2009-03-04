#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.


"""
Converts gpaw nc-files to ``*.db`` files

This file is intended to be moved into a different repository!
"""



import os,sys
import gpaw.db.dacapo_ReadWriter  as dacapo_ReadWriter


class GPW2DB:
    """ Takes an gpw file and creates a db file from it """
    def __init__(self, gpw_filename, db_filename, db=True, private="660", **kwargs):
        """
           parameter 'gpw_filename': the name of the gpaw file
           parameter 'db_filename': the name of the output db-file
           parameter 'db': add the output also to the database location
           parameter 'private': the file attribute for the db-file, if db=True
           parameter 'kwargs': more arguments i.e. a list of keywords
                     for example keywords=["...","..."]
                                 desc="My description"
                                 any_name="any other arguments"
        """
        print "Warning: db and private paramters have no effect yet"
        import gpaw
        import gpaw.db
        calc = gpaw.GPAW(gpw_filename)
        calc.write(db_filename, db=db, private=private, kwargs=kwargs)
        

class NC2DB:
    """ Takes an nc file and creates a db file from it """
    
    def __init__(self, nc_filename, db_filename, db, private="", **kwargs):
        """
           parameter 'nc_filename': the name of the nc file
           parameter 'db_filename': the name of the output db-file
           parameter 'db': add the output also to the database location
           parameter 'private': the file attribute for the db-file, if db=True
           parameter 'kwargs': more arguments i.e. a list of keywords
                     for example keywords=["...","..."]
                                 desc="My description"
                                 any_name="any other arguments"
        """
        from ase.io.pupynere import NetCDFFile
        nc = NetCDFFile(nc_filename)
        calc_name = nc.history;
        if calc_name.lower().index("dacapo")!=-1:
           from Dacapo import Dacapo
           atoms=Dacapo.ReadAtoms(filename=nc_filename)
           calc=atoms.GetCalculator()
           dacapo_ReadWriter.Writer(nc, calc, atoms, db_filename, db, private, **kwargs)
        else:
           print "Could not find calculator for ",calc_name






def print_usage():
    print "Usage: "
    print "   converter.py $.gpw $.db [script.py]"
    print "   converter.py $.nc  $.db [script.py]"
    print "Parameters:"
    print "   $ is a filename"
    print "   parameter $.gpw: a gpaw file"
    print "   parameter $.nc: a NetCD-file"
    print "   parameter script.py: the python script that created the calculation"

def main():
    #the following parameters only work for nc - files currently:
    db = True         # copy the db file to the global database
    private = "660"   # file attribute for the global db - file


    argv = sys.argv
    if len(argv)==4:
        sys.argv=[sys.argv[3]] 
    else:
        sys.argv=[] #otherwise this script gets included!
    if len(argv)!=3 and len(argv)!=4:
        print_usage()
    else:
        if len(argv)==4 and not os.path.exists(sys.argv[0]):
           print "Could not find "+sys.argv[0]
           return

        if not os.path.exists(argv[1]):
           print "Could not find "+argv[1]
           return
        #if not os.path.exists(sys.argv[2]):
        #   print "File "+sys.argv[2]+" exists already and is overwritten."
        #   return
        if argv[1].endswith(".gpw"):
            GPW2DB(argv[1],argv[2], db, private)
        elif argv[1].endswith(".nc"):
            NC2DB(argv[1],argv[2], db, private)
        else:
            print "Cannot convert: Unknown format"


if __name__ == "__main__":
    main()

