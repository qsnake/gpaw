#!/usr/bin/env python
from ASE import Atom, ListOfAtoms
from gpaw import Calculator
import Numeric as num
from gpaw.utilities import equal
import Scientific.IO.NetCDF as NetCDF
from ASE.IO.Cube import WriteCube

import time

magmom = 0.0
ng = 16

nhostswrite = [8]
nhostsread = [8]

tests = []
for nkpt in [4]:
    for magmom in [3.0]:
        test =  'test:  nkpt = %d magmom = %1.1f'%(nkpt,magmom)
        for nhosts in nhostswrite: 
       	    print test
            file_prefix = 'Fe_%d_%1.1f_par%d'%(nkpt,magmom,nhosts)

            fcc = ListOfAtoms([Atom('Fe', (0, 0, 0.0001) ,magmom=magmom)],
                              periodic=True,
                              cell = (2.55,2.55,2.55))

            calc = Calculator(nbands=6,
                              gpts=(ng,ng,ng),
                              kpts=(4, 2, 2),
                              out=file_prefix+'.txt',
                              tolerance = 1e-10, 
                              hosts=nhosts)
	
            fcc.SetCalculator(calc)
            fcc[0].SetMagneticMoment(magmom)
            e = fcc.GetPotentialEnergy()
            calc.Write(file_prefix+'.nc')

            del calc,fcc

        for nhosts in nhostsread: 
            file_prefix = 'Fe_%d_%1.1f_par%d'%(nkpt,magmom,nhosts)
            print '------ restart calculation  ',file_prefix
            atoms = Calculator.ReadAtoms(file_prefix+'.nc',
                                         out=file_prefix+'_restart.txt',
                                         tolerance = 1e-10,
                                         hosts=nhosts)
            calc = atoms.GetCalculator()
            atoms[0].SetCartesianPosition([0, 0, -0.0001])
            erg = atoms.GetPotentialEnergy()


            result = 'ok'
            equal(e,erg,1e-4)

            niter = calc.GetNumberOfIterations() 
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
    calc = Calculator(nbands=8, h=0.2,out = 'O2.txt',tolerance=1e-9,
                      hosts = nhosts)
    O2.SetCalculator(calc)
    e0 = O2.GetPotentialEnergy()
    f  = O2.GetCartesianForces()
    equal(2.1062, sum(abs(f.flat)), 1e-2)
    calc.Write('O2.nc')

    O2[1].SetCartesianPosition((1.21+d,d,d))
    e2 = O2.GetPotentialEnergy()
    niter2 = calc.GetNumberOfIterations()
    f2 = O2.GetCartesianForces()

    del calc,O2

if 1: 
    atoms = Calculator.ReadAtoms('O2.nc',out='O2-restart.txt',hosts=nhosts,
                                 tolerance=1e-9)
    e = atoms.GetPotentialEnergy()
    atoms[1].SetCartesianPosition((1.21+d,d,d))
    e1 = atoms.GetPotentialEnergy()
    f1 = atoms.GetCartesianForces()
    niter1 = atoms.GetCalculator().GetNumberOfIterations()

    print e1,e2
    print niter1,niter2
    print sum(abs(f1.flat-f2.flat))
    print f1,f2, f1-f2
    equal(e1,e2,3e-5)
    equal(niter1,niter2,0)
    equal(sum(abs(f1.flat-f2.flat)),0.0,0.002)


