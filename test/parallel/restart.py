#!/usr/bin/env python
from ASE import Atom, ListOfAtoms
from gpaw import Calculator
import numpy as npy
from gpaw.utilities import equal
from gpaw.mpi import rank, world
import time

magmom = 0.0
ng = 16

nhostswrite = [8]
nhostsread = [8]

tests = []
for nkpt in [4]:
    for magmom in [3.0]:
        test = 'test:  nkpt = %d magmom = %1.1f' % (nkpt, magmom)
        for nhosts in nhostswrite: 
       	    print test
            file_prefix = 'Fe_%d_%1.1f_par%d'%(nkpt,magmom,nhosts)

            fcc = ListOfAtoms([Atom('Fe', (0, 0, 0.0001) ,magmom=magmom)],
                              periodic=True,
                              cell = (2.55,2.55,2.55))

            calc = Calculator(nbands=6,
                              gpts=(ng,ng,ng),
                              kpts=(4, 2, 2),
                              txt=file_prefix+'.txt',
                              tolerance = 1e-10)
	
            fcc.SetCalculator(calc)
            fcc[0].SetMagneticMoment(magmom)
            e = fcc.GetPotentialEnergy()
            calc.write(file_prefix+'.gpw')
            del calc,fcc

        for nhosts in nhostsread: 
            file_prefix = 'Fe_%d_%1.1f_par%d'%(nkpt,magmom,nhosts)
            print '------ restart calculation  ',file_prefix, rank*111111111
            calc = Calculator(file_prefix+'.gpw',
                              txt=file_prefix+'_restart.txt',
                              tolerance = 1e-10)
            atoms = calc.get_atoms()
            atoms[0].SetCartesianPosition([0, 0, -0.0001])
            erg = atoms.GetPotentialEnergy()


            result = 'ok'
            equal(e,erg,1e-4)

            niter = calc.niter
            tests.append((test,result,niter,nhosts))

del calc,atoms

for test in tests: 
    print "%s ---- %10s --- %d ---- %d "%(test[0],test[1],test[2],test[3])


nhosts = 8
d = 2.0
if 1: 
    a = 5.0
    O2 = ListOfAtoms([Atom('O',(0+d,d,d  ), magmom=1),
                      Atom('O',(1.2+d,d,d), magmom=1)],
                     periodic=1,
                     cell=(a, a, a))
    calc = Calculator(nbands=8, h=0.2, txt = 'O2.txt',tolerance=1e-9)
    O2.SetCalculator(calc)
    e0 = O2.GetPotentialEnergy()
    f  = O2.GetCartesianForces()
    #equal(2.1062, sum(abs(f.ravel())), 1e-2)
    calc.write('O2.gpw')
    print e0, f
    O2[1].SetCartesianPosition((1.21+d,d,d))
    e2 = O2.GetPotentialEnergy()
    niter2 = calc.niter
    f2 = O2.GetCartesianForces()

    del calc,O2

if 1: 
    atoms = Calculator('O2.gpw', txt='O2-restart.txt',
                       tolerance=1e-9).get_atoms()
    e = atoms.GetPotentialEnergy()
    atoms[1].SetCartesianPosition((1.21+d,d,d))
    e1 = atoms.GetPotentialEnergy()
    f1 = atoms.GetCartesianForces()
    niter1 = atoms.GetCalculator().niter

    print e1,e2
    print niter1,niter2
    print sum(abs(f1.ravel()-f2.ravel()))
    print f1,f2, f1-f2
    equal(e1,e2,3e-5)
    equal(niter1,niter2,0)
    equal(sum(abs(f1.ravel()-f2.ravel())),0.0,0.002)


