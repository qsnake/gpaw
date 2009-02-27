#! /usr/bin/python
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os

import gpaw.io

from gpaw import *
from gpaw.version import version as gversion
from gpaw.db import params


class gpaw_info(params.params):

    #defines the set of interesting values
    # key: internal name
    # value: dictionary containing additional information
    #        common values are:
    #          xml_name:    the name of the xml-tag
    #          python_type: the internal type of the variable
    #          unit:        the unit of the variable
    gpaw_data = {
            'history':{"xml_name":'history'},
            'version':{"xml_name":'version'},
            'lengthunit':{"xml_name":'lengthunit'},
            'energyunit':{"xml_name":'energyunit'},

            'user':{"xml_name":'user'},
            'date':{"xml_name":'date'},
            'arch':{"xml_name":'arch'},
            'pid':{"xml_name":'pid'},
            'ase_dir':{"xml_name":'ase_dir'},
            'ase_version':{"xml_name":'ase_version'},
            'numpy_dir':{"xml_name":'numpy_dir'},
            'gpaw_dir':{"xml_name":'gpaw_dir'},
            'gpaw_version':{"xml_name":'gpaw_version'},
            'units':{"xml_name":'units'},
            'script_name':{"xml_name":'script_name'},
            'keywords':{"xml_name":'keywords'},
            'desc':{"xml_name":'desc'},
            'AluminiumFingerprint':{"xml_name":'AluminiumFingerprint'},

            'KohnShamStencil':{"xml_name":'KohnShamStencil'},
            'InterpolationStencil':{"xml_name":'InterpolationStencil'},
            'PoissonStencil' :{"xml_name":'PoissonStencil'},
            'XCFunctional' :{"xml_name":'XCFunctional'},
            'Charge' :{"xml_name":'Charge'},
            'FixMagneticMoment' :{"xml_name":'FixMagneticMoment'},
            'UseSymmetry' :{"xml_name":'UseSymmetry'},
            'ForTransport' :{"xml_name":'ForTransport'},
            'Converged' :{"xml_name":'Converged'},
            'FermiWidth' :{"xml_name":'FermiWidth'},
            'MixClass' :{"xml_name":'MixClass'},
            'MixBeta' :{"xml_name":'MixBeta'},
            'MixOld' :{"xml_name":'MixOld'},
            'MixMetric' :{"xml_name":'MixMetric'},
            'MixWeight' :{"xml_name":'MixWeight'},
            'MaximumAngularMomentum' :{"xml_name":'MaximumAngularMomentum'},
            'SoftGauss' :{"xml_name":'SoftGauss'},
            'FixDensity' :{"xml_name":'FixDensity'},
            'DensityConvergenceCriterion' :{"xml_name":'DensityConvergenceCriterion'},
            'EnergyConvergenceCriterion' :{"xml_name":'EnergyConvergenceCriterion'},
            'EigenstatesConvergenceCriterion' :{"xml_name":'EigenstatesConvergenceCriterion'},
            'NumberOfBandsToConverge' :{"xml_name":'NumberOfBandsToConverge'},
            'Ekin' :{"xml_name":'Ekin', "unit":""},
            'Epot' :{"xml_name":'Epot', "unit":""},
            'Ebar' :{"xml_name":'Ebar', "unit":""},
            'Eext' :{"xml_name":'Eext', "unit":""},
            'Exc' :{"xml_name":'Exc', "unit":""},
            'S' :{"xml_name":'S'},
            'FermiLevel' :{"xml_name":'FermiLevel'},
            'DensityError' :{"xml_name":'DensityError'},
            'EnergyError' :{"xml_name":'EnergyError'},
            'EigenstateError' :{"xml_name":'EigenstateError'},
            'DataType' :{"xml_name":'DataType'},
            'Time' :{"xml_name":'Time', "unit":""},
            'SetupTypes' :{"xml_name":'SetupTypes'},

            'AtomicNumbers':{"xml_name":'AtomicNumbers'},
            'CartesianPositions':{"xml_name":'CartesianPositions'},
            'MagneticMoments':{"xml_name":'MagneticMoments'},
            'Tags':{"xml_name":'Tags'},
            'BoundaryConditions':{"xml_name":'BoundaryConditions'},
            'UnitCell':{"xml_name":'UnitCell'},
            'PotentialEnergy':{"xml_name":'PotentialEnergy'},
            'BZKPoints':{"xml_name":'BZKPoints'},
            'IBZKPoints':{"xml_name":'IBZKPoints'},
            'IBZKPointWeights':{"xml_name":'IBZKPointWeights'},
           #'Projections':{"xml_name":'Projections', "python_type":complex},
           #'AtomicDensityMatrices':{"xml_name":'AtomicDensityMatrices'},
           #'NonLocalPartOfHamiltonian':{"xml_name":'NonLocalPartOfHamiltonian'},
            'Eigenvalues':{"xml_name":'Eigenvalues'},
            'OccupationNumbers':{"xml_name":'OccupationNumbers'},
           #'PseudoElectronDensity':{"xml_name":'PseudoElectronDensity'},
           #'PseudoPotential':{"xml_name":'PseudoPotential'},
    }

    def __init__(self):
        params.params.__init__(self, self.gpaw_data)

    def write_version(self, w):
        """parameter w is a reference to a writer
           this function writes the versions to 
           the writer
        """
        w['gpaw_dir']=os.path.dirname(gpaw.__file__)
        w['gpaw_version']=gversion

    
