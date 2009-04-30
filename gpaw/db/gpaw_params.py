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
            'lengthunit':{"xml_name":'length_unit'},
            'energyunit':{"xml_name":'energy_unit'},

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
            'AluminiumFingerprint':{"xml_name":'aluminium_fingerprint'},

            'KohnShamStencil':{"xml_name":'kohn_sham_stencil'},
            'InterpolationStencil':{"xml_name":'interpolation_stencil'},
            'PoissonStencil' :{"xml_name":'poisson_stencil'},
            'XCFunctional' :{"xml_name":'xc_functional'},
            'Charge' :{"xml_name":'charge'},
            'FixMagneticMoment' :{"xml_name":'_fix_magnetic_moment'},
            'UseSymmetry' :{"xml_name":'use_symmetry'},
            'ForTransport' :{"xml_name":'for_transport'},
            'Converged' :{"xml_name":'converged'},
            'FermiWidth' :{"xml_name":'fermi_width'},
            'MixClass' :{"xml_name":'mix_class'},
            'MixBeta' :{"xml_name":'mix_beta'},
            'MixOld' :{"xml_name":'mix_old'},
            'MixMetric' :{"xml_name":'mix_metric'},
            'MixWeight' :{"xml_name":'mix_weight'},
            'MaximumAngularMomentum' :{"xml_name":'maximum_angular_momentum'},
            'SoftGauss' :{"xml_name":'soft_gauss'},
            'FixDensity' :{"xml_name":'fix_density'},
            'DensityConvergenceCriterion' :{"xml_name":'density_convergence_criterion'},
            'EnergyConvergenceCriterion' :{"xml_name":'energy_convergence_criterion'},
            'EigenstatesConvergenceCriterion' :{"xml_name":'eigenstates_convergence_criterion'},
            'NumberOfBandsToConverge' :{"xml_name":'number_of_bands_to_converge'},
            'Ekin' :{"xml_name":'e_kin', "unit":""},
            'Epot' :{"xml_name":'e_pot', "unit":""},
            'Ebar' :{"xml_name":'e_bar', "unit":""},
            'Eext' :{"xml_name":'e_ext', "unit":""},
            'Exc' :{"xml_name":'e_xc', "unit":""},
            'S' :{"xml_name":'s'},
            'FermiLevel' :{"xml_name":'fermi_level'},
            'DensityError' :{"xml_name":'density_error'},
            'EnergyError' :{"xml_name":'energy_error'},
            'EigenstateError' :{"xml_name":'eigenstate_error'},
            'DataType' :{"xml_name":'data_type'},
            'Time' :{"xml_name":'time', "unit":""},
            'SetupTypes' :{"xml_name":'setup_types'},

            'AtomicNumbers':{"xml_name":'atomic_numbers'},
            'CartesianPositions':{"xml_name":'cartesian_positions'},
            'MagneticMoments':{"xml_name":'magnetic_moments'},
            'Tags':{"xml_name":'tags'},
            'BoundaryConditions':{"xml_name":'boundary_conditions'},
            'UnitCell':{"xml_name":'unit_cell'},
            'PotentialEnergy':{"xml_name":'potential_energy'},
            'BZKPoints':{"xml_name":'bz_k_points'},
            'IBZKPoints':{"xml_name":'ibz_k_points'},
            'IBZKPointWeights':{"xml_name":'ibz_k_point_weights'},
           #'Projections':{"xml_name":'Projections', "python_type":complex},
           #'AtomicDensityMatrices':{"xml_name":'AtomicDensityMatrices'},
           #'NonLocalPartOfHamiltonian':{"xml_name":'NonLocalPartOfHamiltonian'},
            'Eigenvalues':{"xml_name":'eigenvalues'},
            'OccupationNumbers':{"xml_name":'occupation_numbers'},
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

    
