#!/usr/bin/env python

import os.path

from ase import Atoms,Atom
from gpaw import GPAW
from gpaw.mixer import Mixer, MixerSum
from gpaw.mpi import rank, world
from gpaw.utilities import equal

# -------------------------------------------------------------------

def create_calc(name, spinpol, pbc):
    # Bond lengths between H-C and C-C for ethyne (acetylene) cf.
    # CRC Handbook of Chemistry and Physics, 87th ed., p. 9-28
    dhc = 1.060
    dcc = 1.203

    atoms = Atoms([Atom('H', (0, 0, 0)),
                   Atom('C', (dhc, 0, 0)),
                   Atom('C', (dhc+dcc, 0, 0)),
                   Atom('H', (2*dhc+dcc, 0, 0))],
                  pbc=pbc)

    atoms.center(vacuum=2.0)

    # Number of occupied and unoccupied bands to converge
    nbands = int(10/2.0)
    nextra = 3

    #TODO use pbc and cell to calculate nkpts
    kwargs = {}

    if spinpol:
        kwargs['mixer'] = MixerSum(nmaxold=5, beta=0.1, weight=100)
    else:
        kwargs['mixer'] = Mixer(nmaxold=5, beta=0.1, weight=100)

    calc = GPAW(h=0.3,
                nbands=nbands+nextra,
                xc='PBE',
                spinpol=spinpol,
                eigensolver='cg',
                convergence={'energy':1e-4/len(atoms), 'density':1e-5, \
                             'eigenstates': 1e-9, 'bands':-1},
                txt=name+'.txt', **kwargs)

    atoms.set_calculator(calc)
    atoms.get_potential_energy()

    calc.write(name+'.gpw', mode='all')

def get_calc(name, spinpol, parsize, pbc):
    if not os.path.isfile(name+'.gpw'):
       create_calc(name, spinpol, pbc)

    name_parsize = 'par%d%d%d' % parsize[:]

    calc = GPAW(name+'.gpw',
                parsize=parsize,
                txt=name+'_'+name_parsize+'.txt')

    return calc


if __name__ in ['__main__', '__builtin__']:

    import sys

    def write(text):
        if rank == 0:
            sys.stdout.write(text)
            sys.stdout.flush()

    pbcs_c = [tuple([bool(p & 2**c) for c in range(3)]) for p in range(8)]
    #pbcs_c = [(False, False, False)]

    for spinpol in [True,False]:

        parsizes_dc = {}

        parsizes_dc[1] = [(1,1,1)]
        parsizes_dc[2] = [(1,1,2),(1,2,1),(2,1,1)]
        parsizes_dc[3] = [(1,1,3),(1,3,1),(3,1,1)]
        parsizes_dc[4] = [(1,1,4),(1,2,2),(1,4,1),(2,1,2),(2,2,1),(4,1,1)] #ign.
        parsizes_dc[5] = [(1,1,5),(1,5,1),(5,1,1)] #ign.
        parsizes_dc[6] = [(1,2,3),(1,3,2),(2,3,1),(2,1,3),(3,1,2),(3,2,1)] #more
        parsizes_dc[7] = [(1,1,7),(1,7,1),(7,1,1)]
        parsizes_dc[8] = [(1,2,4),(1,4,2),(2,1,4),(2,2,2),(2,4,1),(4,1,2),(4,2,1)] #more

        if spinpol:
            ndomains = max(1, world.size//2)
        else:
            ndomains = world.size

        for pbc_c in pbcs_c:
            name_spin = 'spin%s' % (spinpol and 'polarized' or 'paired')
            name_pbc = 'pbc%d%d%d' % pbc_c[:]
            name = 'resume_%s_%s' % (name_spin, name_pbc)

            for parsize_c in parsizes_dc[ndomains]:
                calc = get_calc(name, spinpol, parsize_c, pbc_c)
                write('spinpol=%s, pbc=(%d,%d,%d), parsize=(%d,%d,%d): ' \
                       % ((spinpol,) + tuple(pbc_c) + tuple(parsize_c)))
                try:
                    E1 = calc.get_potential_energy()
                    calc.scf.reset()
                    calc.scf.niter_fixdensity = -1 #a hack!
                    calc.calculate(calc.atoms)
                    E2 = calc.get_potential_energy()
                except ValueError, e:
                    if e.message != 'Grid too small!':
                        raise e
                    write('ignored\n')
                else:
                    equal(E1, E2, 1e-4)
                    write('ok\n')

