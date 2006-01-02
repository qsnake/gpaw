#!/usr/bin/env python
from ASE import Atom, ListOfAtoms
from gridpaw import Calculator
import Numeric as num
from gridpaw.utilities import equal
import Scientific.IO.NetCDF as NetCDF
from ASE.IO.Cube import WriteCube

import time

magmom = 0.0
ng = 16

nhostswrite = [2]
nhostsread = [2]

tests = []
for nkpt in [4]:
   for magmom in [1.0]:
      test =  'test:  nkpt = %d magmom = %1.1f'%(nkpt,magmom)
      for nhosts in nhostswrite: 
       	print test
       	file_prefix = 'Fe_%d_%1.1f_par%d'%(nkpt,magmom,nhosts)

        fcc = ListOfAtoms([Atom('Fe',magmom=magmom)],
                           periodic=True,
                           cell = (2.55,2.55,2.55))

       	calc = Calculator(nbands=6,
                         gpts=(ng,ng,ng),
                         kpts=(2, 2, nkpt),
                         out=file_prefix+'.txt',
                         tolerance = 1e-6, 
                         hosts=nhosts)
	
       	fcc.SetCalculator(calc)
       	fcc[0].SetMagneticMoment(magmom)
       	e = fcc.GetPotentialEnergy()
       	calc.Write(file_prefix+'.nc')

        del calc,fcc
       	time.sleep(10)

      for nhosts in nhostsread: 
       	file_prefix = 'Fe_%d_%1.1f_par%d'%(nkpt,magmom,nhosts)
        print '------ restart calculation  ',file_prefix
        atoms = Calculator.ReadAtoms(file_prefix+'.nc',
                                    out=file_prefix+'_restart.txt',
				    mix = 0.1,
                                    old = 1,
                                    hosts=nhosts)
        # atoms.SetCartesianPositions(atoms.GetCartesianPositions()+0.0000001)
        atoms.SetCartesianPositions(atoms.GetCartesianPositions()+0.01)
        atoms.SetCartesianPositions(atoms.GetCartesianPositions()-0.01)
        e = erg = atoms.GetPotentialEnergy()

        result = 'ok'
        try: 
           equal(e,e,1e-4)
        except: 
           result = 'failed'

        niter = atoms.GetCalculator().GetNumberOfIterations() 
        tests.append((test,result,niter,nhosts))


for test in tests: 
   print "%s ---- %10s --- %d ---- %d "%(test[0],test[1],test[2],test[3])
