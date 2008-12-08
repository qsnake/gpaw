#!/usr/bin/env python

# asses the quality of a setup X based on equilibrium distance (re) [A],
# harmonic frequency (we) [1/cm], dipole moment (m0) [D] of XY compounds,
# for fixed Y setups.

element_choices = ["ALL", "H", "F", "Cl", "Br", "I"]

mode_choices = ['calculate', 'analyse']

xc_choices = ['X_B88-C_LYP', 'PBE']

from optparse import OptionParser

parser = OptionParser(usage='%prog [options] package.\nExample of call:\n'+
                      'python %prog X\n',
                      version='%prog 0.1')
parser.add_option('--mode', dest="mode", type="choice",
                  default=mode_choices[-1],
                  choices=mode_choices,
                  help='mode: calculate or analyse.')
parser.add_option("--xc", dest="xc",
                  default=xc_choices[0],
                  help='use xc functional.')
parser.add_option("--spacings", dest="spacings",
                  default=[0.14],
                  help='test grid spacing convergence in the list [a1, a2, ...].')
parser.add_option("--widths", dest="widths",
                  default=[None],
                  help='test width convergence in the list [a1, a2, ...].')
parser.add_option('-v', '--verbose', action='store_true',
                  default=False,
                  help='verbose mode.')


opt, args = parser.parse_args()


from ase import *
from ase import units
from ase.vibrations import Vibrations

from gpaw import *
from gpaw.mixer import Mixer, MixerSum
from gpaw import setup_paths

from glob import glob
from os import remove

import numpy as npy

from math import sqrt

from gpaw.mpi import rank, world

def round2(value):
    return '%.2f' % round(value,2)

def round4(value):
    return '%.2f' % round(value,4)

Debye = 1.E-21/units._c/1.E-10/units._e

ref_data = {
    'X_B88-C_LYP':
    {
    'H-F':   {'re':  93.5/100, 'we': 3912.2, 'm0': 1.797},
    'H-Cl':  {'re': 129.4/100, 'we': 2840.4, 'm0': 1.086},
    'H-Br':  {'re': 143.6/100, 'we': 2512.9, 'm0': 0.750},
    'H-I':   {'re': 163.5/100, 'we': 2189.3, 'm0': 0.344},
    'F-F':   {'re': 143.2/100, 'we':  962.8, 'm0': 0.0},
    'Cl-Cl': {'re': 205.6/100, 'we':  501.3, 'm0': 0.0},
    'Br-Br': {'re': 235.2/100, 'we':  290.0, 'm0': 0.0},
    'I-I':   {'re': 276.3/100, 'we':  182.9, 'm0': 0.0},
    'Cl-F':  {'re': 168.2/100, 'we':  724.2, 'm0': 0.945},
    'Br-F':  {'re': 181.3/100, 'we':  612.8, 'm0': 1.483},
    #'I-F':   {'re': 198.2/100, 'we':  540.7, 'm0': 2.080}, # \mu_{0} not converged
    'I-F':   {'re': 198.2/100, 'we':  540.7}, # \mu_{0} not converged
    'Br-Cl': {'re': 220.5/100, 'we':  395.3, 'm0': 0.576},
    #'I-Cl':  {'re': 240.5/100, 'we':  335.9, 'm0': 1.364}, # \mu_{0} not converged
    'I-Cl':  {'re': 240.5/100, 'we':  335.9}, # \mu_{0} not converged
    #'I-Br':  {'re': 255.5/100, 'we':  233.3, 'm0': 0.831}, # \mu_{0} not converged
    'I-Br':  {'re': 255.5/100, 'we':  233.3}, # \mu_{0} not converged
    }
    }

# Use setups from the $PWD first
setup_paths.insert(0, '.')

# Use setups from the $PWD first
setup_paths.insert(0, '../')

a = 16.0
#a = 8.0
b = a
c = a

def calculate(element, ref_data, p):
    values_dict = {}
    values_dict[p['xc']] = {}
    for XY, data in ref_data[p['xc']].items():
        X = XY.split('-')[0]
        Y = XY.split('-')[1]
        if (X == Y and X == element) or (X != Y and (X == element or Y == element)):
            # compound contains the requested element
            re_ref = data['re']
            we_ref = data['we']
            m0_ref = data.get('m0', 0.0)
            #
            compound = Atoms(X+Y,
                             [
                (0,       0,     0.5  ),
                (0,       0,     0.5+re_ref/a),
                ],
                             pbc=0)
            compound.set_cell([a, b, c], scale_atoms=1)
            compound.center()

            # calculation on the reference geometry
            calc = Calculator(**p)
            compound.set_calculator(calc)
            e_compound = compound.get_potential_energy()
            dip = calc.finegd.calculate_dipole_moment(calc.density.rhot_g)*calc.a0
            vib = Vibrations(compound)
            vib.run()
            vib_compound = vib.get_frequencies(method='frederiksen').real[-1]
            world.barrier()
            vib_pckl = glob('vib.*.pckl')
            if rank == 0:
                for file in vib_pckl: remove(file)

            # calculation on the relaxed geometry
            qn = QuasiNewton(compound)
            #qn.attach(PickleTrajectory('compound.traj', 'w', compound).write)
            qn.run(fmax=0.05)
            e_compound_r = compound.get_potential_energy()
            dist_compound_r = compound.get_distance(0,1)
            dip_r = calc.finegd.calculate_dipole_moment(calc.density.rhot_g)*calc.a0
            vib = Vibrations(compound)
            vib.run()
            vib_compound_r = vib.get_frequencies(method='frederiksen').real[-1]
            world.barrier()
            vib_pckl = glob('vib.*.pckl')
            if rank == 0:
                for file in vib_pckl: remove(file)

            del compound
            e = e_compound
            we = vib_compound
            m0 = dip
            e_r = e_compound_r
            we_r = vib_compound_r
            re_r = dist_compound_r
            m0_r = dip_r
            #
            values_dict[p['xc']][XY] = {'re': re_r, 'we': (we_r, we), 'm0': (m0_r, m0)}
    #
    return values_dict

def get_vector_name(vector, format):
    """Get the name of the vector based
    on the coordinates and unit format, e.g.:
    ([1, 2.01, 3.001], '%02d') -> 010203
    """
    assert len(vector) >= 1
    name = ''
    for unit in vector:
        name += format % (unit)
    return name


if __name__ == '__main__':
    assert (len(args) >= 1)
    element = args[0]
    assert element in element_choices, element+' not in '+str(element_choices)
    if element == 'ALL':
        elements = element_choices[1:]
    else:
        elements = [element]

    import pickle

    spacings = eval(str(opt.spacings))
    widths = eval(str(opt.widths))
    for element in elements:
        for spacing in spacings:
            for width in widths:
                # pickle filename
                pickle_filename = element+'h'+get_vector_name([spacing], '%.3f')
                parameters = {
                    'xc': opt.xc,
                    'h': spacing,
                    'txt': '-'}
                # use widht only if requested
                if width != None:
                    parameters = {
                        'width': width}
                    pickle_filename = pickle_filename+'w'+get_vector_name([width], '%.3f')
                if opt.mode == 'calculate':
                    data = calculate(element, ref_data, parameters)
                    pickle.dump(data, open(pickle_filename, 'w'))
                elif opt.mode == 'analyse':
                    data = pickle.load(open(pickle_filename))
                if opt.verbose:
                    if rank == 0:
                        print 'data ',pickle_filename, data
                # analyse the results
                result = {}
                error = {}
                count = {'re': 0, 'we':  0, 'm0': 0}
                cumulative_abs_relative_error = {'re': 0.0, 'we':  0.0, 'm0': 0.0}
                max_relative_error = {'re': (None, 0.0), 'we':  (None, 0.0), 'm0': (None, 0.0)}
                for XY, data in data[parameters['xc']].items():
                    result[XY] = {}
                    error[XY] = {}
                    X = XY.split('-')[0]
                    Y = XY.split('-')[1]
                    if X == element or Y == element:
                        for key, value in data.items():
                            ref = ref_data[parameters['xc']][XY].get(key, -1)
                            if ref > 0.0:
                                if key == 'we': value = value[0]
                                if key == 'm0': value = sqrt(npy.dot(value[0],value[0]))/Debye
                                relative_error = (value-ref)/ref
                                result[XY][key] = value
                                error[XY][key] = relative_error
                                cumulative_abs_relative_error[key] += abs(relative_error)
                                if abs(relative_error) > abs(max_relative_error[key][1]):
                                    max_relative_error[key] = (XY, relative_error)
                                count[key] += 1
                            else:
                                result[XY][key] = 0.0
                                error[XY][key] = 0.0

                for key, value in cumulative_abs_relative_error.items():
                    cumulative_abs_relative_error[key] = cumulative_abs_relative_error[key]/count[key]

                if opt.verbose:
                    if rank == 0:
                        print 'error ',pickle_filename, error
                        print 'cumulative_abs_relative_error ',pickle_filename, cumulative_abs_relative_error
                        print 'max_relative_error ',pickle_filename, max_relative_error

                if rank == 0:
                    print pickle_filename
                    print '  X-Y| r [A]|   ref|   [%]| w [1/cm]|    ref|   [%]| m [D]|   ref|   [%]|'
                    keys = []
                    for key, value in result.items():
                        keys.append(key)
                    keys.sort()
                    for key in keys:
                        value = result.get(key)
                        value_ref = ref_data[parameters['xc']][key]
                        if abs(value_ref.get('m0', 0.0)) < 0.001:
                            m0_error = 0.0
                        else:
                            m0_error = ((value['m0']-value_ref['m0'])/value_ref['m0'])*100
                        print "%5s| %5.3f| %5.3f| %5.1f|   %6.1f| %6.1f| %5.1f| %5.3f| %5.3f| %5.1f|" % (
                            key, value['re'], value_ref['re'], ((value['re']-value_ref['re'])/value_ref['re'])*100,
                            value['we'], value_ref['we'], ((value['we']-value_ref['we'])/value_ref['we'])*100,
                            value['m0'], value_ref.get('m0', 0.0), m0_error
                            )
