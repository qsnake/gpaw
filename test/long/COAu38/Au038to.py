#!/usr/bin/env python
from ase import *
from ase.utils.eos import *
from gpaw import * 
x = 2 / 0

kwargs = dict(mode='lcao',
              #basis='dzp',
              gpts=(112, 112, 112),
              convergence={'density':0.1, 'energy':0.1}
              )#poissonsolver=poissonsolver)

cobocta = read('A2.gpe10.traj')
x = 'A7c'
calc = GPAW(#h      = 0.18,
            nbands = 250,
            xc     = 'PBE',
            width  = 0.01,
            mixer  = Mixer(0.1, 5, metric='new', weight=100.0),
            txt    = x + '.out',
            **kwargs)
cobocta.set_calculator(calc)
cobocta.get_potential_energy()
cobocta.positions = read('A2gf11.traj').positions
cobocta.get_potential_energy()
