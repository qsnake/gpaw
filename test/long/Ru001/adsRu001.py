#!/usr/bin/env python

from optparse import OptionParser

code_choices = ['gpaw', 'dacapo', 'abinit', 'elk']
adsorbate_choices = ['None', 'N', 'O']
geometry_choices = ['fix', 'relax']
mode_choices = ['molecule', 'slab']
xc_choices = ['PW91', 'LDA', 'PBE']

parser = OptionParser(usage='%prog [options] package.\nExample of call:\n'+
                      'Calculate adsorption on Ru001 system'
                      'python %prog --code=dacapo\n'+
                      'python %prog --code=dacapo --mode=slab\n'+
                      'python %prog --code=dacapo --mode=slab --adsorbate=N\n'+
                      'python %prog --code=dacapo --mode=slab --adsorbate=O\n',
                      version='%prog 0.1')
parser.add_option('--code', dest="code", type="choice",
                  default=code_choices[0],
                  choices=code_choices,
                  help='code: which code to use.')
parser.add_option('--adsorbate', dest="adsorbate", type="choice",
                  default=adsorbate_choices[0],
                  choices=adsorbate_choices,
                  help='adsorbate.')
parser.add_option('--geometry', dest="geometry", type="choice",
                  default=geometry_choices[-1],
                  choices=geometry_choices,
                  help='geometry: fix geometry (read from traj file) or relax.')
parser.add_option('--mode', dest="mode", type="choice",
                  default=mode_choices[0],
                  choices=mode_choices,
                  help='mode: calculate molecules or slab w/wo molecules.')
parser.add_option('--xc', dest="xc", type="choice",
                  default=xc_choices[-1],
                  choices=xc_choices,
                  help='XC functional.')
parser.add_option('-v', '--verbose', action='store_true',
                  default=False,
                  help='verbose mode.')

opt, args = parser.parse_args()

from os import remove
from os.path import exists, join

try:
    import numpy as npy
except ImportError:
    raise SystemExit('numpy is not installed!')

try:
    import gpaw
except ImportError:
    raise SystemExit('gpaw is not installed!')

from gpaw.utilities.tools import gridspacing2cutoff

try:
    import ase
except ImportError:
    raise SystemExit('ase is not installed!')

from ase import Atoms, Atom
from ase import molecule
from ase import QuasiNewton
from ase.lattice.surface import hcp0001, add_adsorbate
from ase.constraints import FixAtoms
from ase.io.xyz import read_xyz
from ase.io.trajectory import PickleTrajectory
from ase.io.trajectory import read_trajectory, write_trajectory

import time

from gpaw import setup_paths
setup_paths.insert(0, '.')

h = 0.2

fmax= 0.07

add_vacuum = 2.0 # additional vacuum

def initialize_parameters(code, width, h):

    parameters = {}

    if code == 'gpaw':
        from gpaw import GPAW as Calculator
        from gpaw.mpi import rank
        parameters['h'] = h
        parameters['width'] = width
        parameters['stencils'] = (3,3)
        #parameters['eigensolver'] = 'cg'
        conv_param = parameters['h']
    elif code == 'dacapo':
        from ase.calculators.dacapo import Dacapo as Calculator
        rank = 0
        parameters['kT'] = width
        parameters['planewavecutoff'] = gridspacing2cutoff(h)
        parameters['densitycutoff'] = parameters['planewavecutoff']*1.2
        conv_param = parameters['planewavecutoff']
    elif code == 'abinit':
        from ase.calculators.abinit import Abinit as Calculator
        rank = 0
        parameters['width'] = width
        parameters['ecut'] = gridspacing2cutoff(h)*1.4
        conv_param = parameters['ecut']
    elif code == 'elk':
        parameters['swidth'] = width
        parameters['stype'] = 3
        parameters['autormt'] = True
        parameters['tforce'] = True # calulate forces
        parameters['nempty'] = 20 # default 5
        #parameters['fixspin'] = -1 # default 0
        #parameters['evalmin'] = -15.0 # default -4.5
        #parameters['autokpt'] = True
        #parameters['nosym'] = True
        #parameters['radkpt'] = 10.0 # default 40.0
        parameters['gmaxvr'] = 16 # default 12
        parameters['rgkmax'] = 7.5 # default 7
        #parameters['gmaxvr'] = 18 # default 12
        #parameters['rgkmax'] = 9.5 # default 7
        parameters['beta0'] = 0.02 # default 0.05
        parameters['betamax'] = 0.05 # default 0.5
        parameters['maxscl'] = 500 # default 200
        #parameters['mixtype'] = 2 # Pulay # default 1
        #parameters['deband'] = 0.005 # default 0.0025
        #parameters['rmtapm'] = '0.25 0.90' # default (0.25,0.95)
    return parameters

def run_molecule(geometry, xc, code):

    parameters = initialize_parameters(code, 0.01, h)
    parameters['xc'] = xc

    for name, nbands in [('N2', 8), ('O2', 8), ('NO', 8)]:
        if code != 'elk':
            parameters['nbands'] = nbands
        if geometry == 'fix':
            mol = read_trajectory(code+'_'+name+'.traj')
        else:
            mol = molecule(name)
        mol.center(vacuum=3.0+add_vacuum)
        if name == 'NO':
            mol.translate((0, 0.1, 0))
        #
        if code == 'gpaw':
            from gpaw import GPAW as Calculator
            from gpaw.mpi import rank
            parameters['txt'] = code+'_'+name+'.txt'
            from gpaw.mixer import Mixer, MixerSum
            #if name == 'N2':
            #    parameters['mixer'] = Mixer(beta=0.1, nmaxold=5, metric='new', weight=100.0)
            #else:
            #    #parameters['eigensolver'] = 'cg'
            #    parameters['mixer'] = MixerSum(beta=0.2, nmaxold=5, metric='new', weight=100.0)
        if code == 'dacapo':
            from ase.calculators.dacapo import Dacapo as Calculator
            rank = 0
            parameters['txtout'] = code+'_'+name+'.txt'
        if code == 'abinit':
            from ase.calculators.abinit import Abinit as Calculator
            rank = 0
            parameters['label'] = code+'_'+name
        if code == 'elk':
            from ase.calculators.elk import ELK as Calculator
            rank = 0
            elk_dir = 'elk_'+str(parameters['rgkmax'])
            conv_param = 1.0
            parameters['dir'] = elk_dir+'_'+name
        #
        calc = Calculator(
            **parameters)
        #
        mol.set_calculator(calc)
        try:
            if geometry == 'fix':
                mol.get_potential_energy()
                traj = PickleTrajectory(code+'_'+name+'.traj', mode='w')
                traj.write(mol)
            else:
                opt = QuasiNewton(mol, logfile=code+'_'+name+'.qn', trajectory=code+'_'+name+'.traj')
                opt.run(fmax=fmax)
        except:
            raise

def run_slab(adsorbate, geometry, xc, code):

    parameters = initialize_parameters(code, 0.1, h)
    parameters['xc'] = xc

    tag = 'Ru001'

    if adsorbate != 'None':
        name = adsorbate + tag
    else:
        name = tag

    if geometry == 'fix':
        slab = read_trajectory(code+'_'+name+'.traj')
    else:
        adsorbate_heights = {'N': 1.108, 'O': 1.257}

        slab = hcp0001('Ru', size=(2, 2, 4), a=2.72, c=1.58*2.72,
                       vacuum=5.0+add_vacuum,
                       orthogonal=True)
        slab.center(axis=2)

        if adsorbate != 'None':
            add_adsorbate(slab, adsorbate, adsorbate_heights[adsorbate], 'hcp')

    slab.set_constraint(FixAtoms(mask=slab.get_tags() >= 3))

    if code != 'elk':
        parameters['nbands'] = 80
        parameters['kpts'] = [4, 4, 1]
    #
    if code == 'gpaw':
        from gpaw import GPAW as Calculator
        from gpaw.mpi import rank
        parameters['txt'] = code+'_'+name+'.txt'
        from gpaw.mixer import Mixer, MixerSum
        parameters['mixer'] = Mixer(beta=0.2, nmaxold=5, metric='new', weight=100.0)
    if code == 'dacapo':
        from ase.calculators.dacapo import Dacapo as Calculator
        rank = 0
        parameters['txtout'] = code+'_'+name+'.txt'
    if code == 'abinit':
        from ase.calculators.abinit import Abinit as Calculator
        rank = 0
        parameters['label'] = code+'_'+name
    if code == 'elk':
        from ase.calculators.elk import ELK as Calculator
        rank = 0
        parameters['autokpt'] = True
        elk_dir = 'elk_'+str(parameters['rgkmax'])
        conv_param = 1.0
        parameters['dir'] = elk_dir+'_'+name
    #
    calc = Calculator(
        **parameters)
    #
    slab.set_calculator(calc)
    try:
        if geometry == 'fix':
            slab.get_potential_energy()
            traj = PickleTrajectory(code+'_'+name+'.traj', mode='w')
            traj.write(slab)
        else:
            opt = QuasiNewton(slab, logfile=code+'_'+name+'.qn', trajectory=code+'_'+name+'.traj')
            opt.run(fmax=fmax)
    except:
        raise

if __name__ == '__main__':

    assert len(args) == 0, 'Error: arguments not accepted'

    assert opt.code in code_choices, opt.code+' not in '+str(code_choices)
    assert opt.adsorbate in adsorbate_choices, opt.adsorbate+' not in '+str(adsorbate_choices)
    assert opt.geometry in geometry_choices, opt.geometry+' not in '+str(geometry_choices)
    assert opt.mode in mode_choices, opt.mode+' not in '+str(mode_choices)
    assert opt.xc in xc_choices, opt.xc+' not in '+str(xc_choices)
    if opt.mode == 'molecule':
        assert opt.adsorbate == 'None', 'adsorbate in molecule: not implemented yet'

    if opt.code == 'dacapo':
        try:
            import ASE
        except ImportError:
            raise SystemExit('ASE (2) is not installed!')

    if opt.mode == 'molecule':
        run_molecule(opt.geometry, opt.xc, opt.code)
    elif opt.mode == 'slab':
        run_slab(str(opt.adsorbate), opt.geometry, opt.xc, opt.code)
