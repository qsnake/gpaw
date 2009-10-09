#!/usr/bin/env python

from optparse import OptionParser

code_choices = ['gpaw', 'dacapo', 'abinit', 'elk']
xc_choices = ['PW91', 'LDA', 'PBE']

parser = OptionParser(usage='%prog [options] package.\nExample of call:\n'+
                      'Report adsorption on Ru001 system'
                      'python %prog --code=dacapo --xc=PBE',
                      version='%prog 0.1')
parser.add_option('--code', dest="code", type="choice",
                  default=code_choices[0],
                  choices=code_choices,
                  help='code: which code to use.')
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
    import ase
except ImportError:
    raise SystemExit('ase is not installed!')

from ase.io.trajectory import read_trajectory

# J.Phys.: Condens. Matter 18 (2006) 41-54
pw91vasp = {'NO':     -0.95,
            'N2':      0.00,
            'O2':      0.00,
            'NRu001': -0.94,
            'ORu001': -2.67,
            'Ru001':   0.00}

def report_results(xc, code):

    energy = {}
    for name in ['NO', 'O2', 'N2', 'Ru001', 'NRu001', 'ORu001']:
        a = read_trajectory(code+'_'+name+'.traj')
        energy[name] = a.get_potential_energy()

    for data, text in [(energy, xc),
                       (pw91vasp, 'PW91 (VASP)')]:
        print ('%22s %.3f %.3f %.3f' %
               (text,
                data['NRu001'] - data['Ru001'] - data['N2'] / 2,
                data['ORu001'] - data['Ru001'] - data['O2'] / 2,
                -(data['NO'] - data['N2'] / 2 - data['O2'] / 2)))

if __name__ == '__main__':

    assert len(args) == 0, 'Error: arguments not accepted'

    assert opt.code in code_choices, opt.code+' not in '+str(code_choices)
    assert opt.xc in xc_choices, opt.xc+' not in '+str(xc_choices)

    print opt.code
    report_results(opt.xc, opt.code)
