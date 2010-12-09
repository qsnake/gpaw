import sys
from optparse import OptionParser

import numpy as np

from gpaw.atom.atompaw import AtomPAW
from gpaw.atom.basis import rsplit_by_norm, QuasiGaussian,\
     get_gaussianlike_basis_function
from gpaw.basis_data import BasisFunction
from gpaw.hgh import setups as hgh_setups, sc_setups as hgh_sc_setups,\
     HGHSetupData


def generate_basis(opts, setup):
    h = opts.grid
    rcut = opts.rcut
    
    calc = AtomPAW(setup.symbol, [setup.f_ln], h=h, rcut=rcut,
                   setups={setup.symbol : setup},
                   lmax=0)
    bfs = calc.extract_basis_functions(basis_name=opts.name)
    ldict = dict([(bf.l, bf) for bf in bfs.bf_j])

    rgd = bfs.get_grid_descriptor()

    def get_rsplit(bf, splitnorm):
        if opts.s_approaches_zero and bf.l == 0:
            l = 1 # create a function phi(r) = A * r + O(r^2)
        else:
            l = bf.l
        return rsplit_by_norm(rgd,
                              l,
                              bf.phit_g * rgd.r_g,
                              splitnorm**2,
                              sys.stdout)

    splitvalence_bfs = []
    for splitnorm in opts.splitnorm:
        splitnorm = float(splitnorm)
        for orbital_bf in bfs.bf_j:
            rsplit, normsqr, phit_g = get_rsplit(orbital_bf, splitnorm)
            phit_g[1:] /= rgd.r_g[1:]
            gcut = rgd.r2g_ceil(rsplit)
            #tailnorm = np.dot(rgd.dr_g[gcut:],
            #                  (rgd.r_g[gcut:] * orbital_bf.phit_g[gcut:])**2)**0.5
            #print 'tailnorm', tailnorm
            dphit_g = orbital_bf.phit_g[:gcut+1] - phit_g[:gcut+1]
            bf = BasisFunction(l=orbital_bf.l,
                               rc=rgd.r_g[gcut],
                               phit_g=dphit_g,
                               type='%s split-valence' % 'spd'[orbital_bf.l])
            splitvalence_bfs.append(bf)
    bfs.bf_j.extend(splitvalence_bfs)

    #rpol = None
    for l in range(3):
        if not l in ldict:
            lpol = l
            source_bf = ldict[lpol - 1]
            break
    else:
        raise NotImplementedError('f-type polarization not implemented')
    
    for splitnorm in opts.polarization:
        splitnorm = float(splitnorm)
        rchar, normsqr, phit_g = get_rsplit(source_bf, splitnorm)
        gcut = rgd.r2g_ceil(3.5 * rchar)
        rcut = rgd.r_g[gcut]
        phit_g = get_gaussianlike_basis_function(rgd, lpol, rchar, gcut)
        N = len(phit_g)
        x = np.dot(rgd.dr_g[:N], (phit_g * rgd.r_g[:N])**2)**0.5
        print 'x', x
        bf = BasisFunction(lpol,
                           rc=rcut,
                           phit_g=phit_g,
                           type='%s polarization' % 'spd'[lpol])
        bf.phit_g
        bfs.bf_j.append(bf)
    bfs.write_xml()


def build_parser():
    usage = '%prog [OPTION] [SYMBOL...]'
    description = 'generate basis set from existing setup.  Values are in'\
                  ' atomic units.'
    p = OptionParser(usage=usage, description=description)
    p.add_option('--grid', default=0.05, type=float, metavar='DR',
                 help='grid spacing in atomic calculation')
    p.add_option('--rcut', default=10.0, type=float,
                 help='radial cutoff for atomic calculation')
    p.add_option('-n', '--name', default='apaw',
                 help='name of basis set')
    p.add_option('--splitnorm', action='append', default=[], metavar='NORM',
                 help='add split-valence basis functions at this'
                 ' tail norm.  Multiple options accumulate')
    p.add_option('--polarization', action='append', default=[],
                 metavar='NORM',
                 help='add polarization function with characteristic radius'
                 ' determined by tail norm.  Multiple options accumulate')
    p.add_option('--s-approaches-zero', action='store_true',
                 help='force s-type split-valence functions to 0 at origin')
    #p.add_option('-t', '--type',
    #             help='string describing extra basis functions')
                 
    return p


def main():
    p = build_parser()
    opts, args = p.parse_args()
    
    for arg in args:
        setup = hgh_setups.get(arg)
        if setup is None:
            setup = hgh_sc_setups.get(arg.split('.')[0])
        if setup is None:
            raise ValueError('Unknown setup %s' % arg)
        print setup
        generate_basis(opts, HGHSetupData(setup))
