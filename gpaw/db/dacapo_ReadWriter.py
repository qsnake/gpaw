# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import xml.sax
import gpaw.db.global_ReadWriter as dbrw
import gpaw.db
import os, time, random

class Writer:
    """ This class is a wrapper to the db output writer 
        and intended to be used with gpaw
    """

    def __init__(self, nc, calc, atoms, db_filename, db, private):
        """nc: the open nc-file
           calc: a Dacapo calculator
        """

        w = dbrw.Writer(db_filename, "dacapo")
        w.set_db_copy_settings(db, private)

        #1st we write the parameters that can be
        #    extracted automatically:

        for k in nc.variables.keys():
            w[k] = nc.variables[k].data

        #2nd write additional variables
        spinpol=calc.GetSpinPolarized()
        if spinpol=="True":
            magmom=calc.GetMagneticMoment()
        else:
            spinpol="False"
            magmom="N-A"
        #spinpol="Spin polarized status = "+spinpol+", Magnetic moment = "+magmom
        w["SpinPol"] = spinpol
        w["MagneticMoment"]  = magmom


        w["ElectronicTemperature"] = calc.GetElectronicTemperature()

        if calc.GetDipoleCorrection()!=[]:
            dipole="True"
        else:
            dipole="False"
        w["DipoleCorrection"] = dipole

        #######Total energy

        w["AtomicNumbers"] = atoms.GetAtomicNumbers()
        w["CartesianPositions"] = atoms.GetCartesianPositions()
        w["PotentialEnergy"] = calc.GetPotentialEnergy()

        w.close()


class Reader(xml.sax.handler.ContentHandler):
    """ This class is a wrapper to the db input reader
        and intended to be used with dacapo.
        NOT YET IMPLEMENTED
    """
    def __init__(self, name):
        self.reader = dbrw.Reader(name)
        print "Not implemented!!!!!!!!!!!!!!!"
        sys.exit(-1)

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

