#! /usr/bin/python
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from gpaw.db import params


class dacapo_info(params.params):

    #defines the set of interesting values
    # key: internal name
    # value: dictionary containing additional information
    #        common values are:
    #          xml_name:    the name of the xml-tag
    #          python_type: the internal type of the variable
    #          unit:        the unit of the variable
    dacapo_data = {
            #arguments in example code
            'number_of_layers':{"xml_name":'number_of_layers'},
            'code':{"xml_name":'code'},
            'electronicconv':{"xml_name":'electronicconv'},
            'symmetry':{"xml_name":'symmetry'},
            'title':{"xml_name":'title'},
            'basis_set':{"xml_name":'basis_set'},
            'coretreatment':{"xml_name":'coretreatment'},
            'othertags':{"xml_name":'othertags'},
            'runtype':{"xml_name":'runtype'},
            'charge_on_unit_cell':{"xml_name":'charge_on_unit_cell'},
            'person':{"xml_name":'person'},
            'DOS':{"xml_name":'hiDOStory'},
            'miller_indices':{"xml_name":'miller_indices'},
            'ionicconv':{"xml_name":'ionicconv'},
            'whererun':{"xml_name":'whererun'},
            'typesystem':{"xml_name":'typesystem'},
            'grid':{"xml_name":'grid'},


            'history':{"xml_name":'history'},
            'version':{"xml_name":'version'},
#            'lengthunit':{"xml_name":'lengthunit'},
#            'energyunit':{"xml_name":'energyunit'},

#           user and date are not created, but taken from the dacapo file
#            'user':{"xml_name":'user'},
#            'date':{"xml_name":'date'},
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

            'User':{"xml_name":'user'},
            'Date':{"xml_name":'date'},

            'TotalEnergy':{"xml_name":'TotalEnergy'},
            'Density_WaveCutoff':{"xml_name":'Density_WaveCutoff'},
            'ConvergenceControl':{"xml_name":'ConvergenceControl'},
            'EvaluateTotalEnergy':{"xml_name":'EvaluateTotalEnergy'},
            'InitialAtomicMagneticMoment':{"xml_name":'InitialAtomicMagneticMoment'},
            'StructureFactor':{"xml_name":'StructureFactor'},
            'ChargeMixing':{"xml_name":'ChargeMixing'},
            'TotalStress':{"xml_name":'TotalStress'},
            'OccupationNumbers':{"xml_name":'OccupationNumbers'},
            'PlaneWaveCutoff':{"xml_name":'PlaneWaveCutoff'},
            'DynamicAtomForces':{"xml_name":'DynamicAtomForces'},
            'DynamicAtomPositions':{"xml_name":'DynamicAtomPositions'},
            'EvaluateExchangeEnergy':{"xml_name":'EvaluateExchangeEnergy'},
            'KpointWeight':{"xml_name":'KpointWeight'},
            'NetCDFOutputControl':{"xml_name":'NetCDFOutputControl'},
            'EvaluateCorrelationEnergy':{"xml_name":'EvaluateCorrelationEnergy'},
            'IBZKpoints':{"xml_name":'IBZKpoints'},
            'ElectronicMinimization':{"xml_name":'ElectronicMinimization'},
            'UseSymmetry':{"xml_name":'UseSymmetry'},
            'EvalFunctionalOfDensity_XC':{"xml_name":'EvalFunctionalOfDensity_XC'},
            'BZKpoints':{"xml_name":'BZKpoints'},
            'ElectronicBands':{"xml_name":'ElectronicBands'},
            'ExcFunctional':{"xml_name":'ExcFunctional'},
            'TotalFreeEnergy':{"xml_name":'TotalFreeEnergy'},
            'AtomTags':{"xml_name":'AtomTags'},
            'NumberPlaneWavesKpoint':{"xml_name":'NumberPlaneWavesKpoint'},
            'EnsembleXCEnergies':{"xml_name":'EnsembleXCEnergies'},
            'AtomProperty_Al':{"xml_name":'AtomProperty_Al'},
            'AtomProperty_Na':{"xml_name":'AtomProperty_Na'},
            'AtomProperty_H':{"xml_name":'AtomProperty_H'},
            'AtomProperty_B':{"xml_name":'AtomProperty_B'},
            'DynamicAtomSpecies':{"xml_name":'DynamicAtomSpecies'},
            'Keywords':{"xml_name":'Keywords'},
            'EigenValues':{"xml_name":'EigenValues'},
            'UnitCell':{"xml_name":'UnitCell'},
            'FermiLevel':{"xml_name":'FermiLevel'},

            'SpinPol':{"xml_name":'SpinPol'},
            'MagneticMoment':{"xml_name":'MagneticMoment'},
            'ElectronicTemperature':{"xml_name":'ElectronicTemperature'},
            'DipoleCorrection':{"xml_name":'DipoleCorrection'},
            'PotentialEnergy':{"xml_name":'PotentialEnergy'},
            'AtomicNumbers':{"xml_name":'AtomicNumbers'},
            'CartesianPositions':{"xml_name":'CartesianPositions'},

    }

    def __init__(self):
        params.params.__init__(self, self.dacapo_data)

    def write_version(self, w):
        """parameter w is a reference to a writer
           this function writes the versions to
           the writer
        """
        #w['dacapo_dir']=os.path.dirname(gpaw.__file__)
        #w['dacapo_version']=gversion
        pass

    