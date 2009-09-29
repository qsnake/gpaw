import os

from ase import *
from ase.parallel import size, rank

from gpaw import GPAW
from gpaw.cluster import Cluster 
from gpaw.analyse.simple_stm import SimpleStm
from gpaw.utilities import equal

from gpaw.output import eigenvalue_string

load=True
load=False
txt = '/dev/null'
#txt='-'

me = ''
if size > 1:
    me += 'rank ' + str(rank) + ': '

BH = Atoms([Atom('B', [.0, .0, .41]),
             Atom('H', [.0, .0, -1.23]),
             ], cell=[5, 5, 6.5])
BH.center()

f3dname = 'stm3d.plt'
def testSTM(calc):
    stm = SimpleStm(calc)
    stm.write_3D([1,0,0], f3dname) # single wf
    wf = stm.gd.integrate(stm.ldos)
##    print "wf=", wf

    if size == 1: # XXXX we have problem with reading plt in parallel
        stm2 = SimpleStm(f3dname)
        wf2 = stm2.gd.integrate(stm2.ldos)
        print 'Integrals: written, read=', wf, wf2
        equal(wf, wf2, 1.e-7)

##    print eigenvalue_string(calc)
    stm.write_3D(3.1, f3dname)
    wf2 = stm.gd.integrate(stm.ldos)
##    print "wf2=", wf2
    equal(wf2, 2, 0.03)

    return wf

# finite system without spin and width
fname='BH-nospin_wfs.gpw'
if not load:
    BH.set_pbc(False)
    cf = GPAW(nbands=3, h=.3, txt=txt)
    BH.set_calculator(cf)
    BH.get_potential_energy()
    cf.write(fname, 'all')
else:
    cf = GPAW(fname, txt=txt)
wf = testSTM(cf)

# finite system with spin
fname='BH-spin_Sz2_wfs.gpw'
BH.set_initial_magnetic_moments([1, 1])
if not load:
    BH.set_pbc(False)
    cf = GPAW(spinpol=True, fixmom=True,
              nbands=5, h=.3, txt=txt, width=.1)
    BH.set_calculator(cf)
    BH.get_potential_energy()
    cf.write(fname, 'all')
else:
    cf = GPAW(fname, txt=txt)
testSTM(cf)

# periodic system
if not load:
    BH.set_pbc(True)
    cp = GPAW(spinpol=True, nbands=3, h=.3, kpts=(2,1,1), txt=txt)
    BH.set_calculator(cp)
    BH.get_potential_energy()
    cp.write('BH-8kpts_wfs.gpw', 'all')
else:
    cp = GPAW('BH-8kpts_wfs.gpw', txt=txt)

stmp = SimpleStm(cp)

stmp.write_3D(-4., f3dname)
print me + 'Integrals(occ): 2 * wf, bias=', 2 * wf, stmp.gd.integrate(stmp.ldos)
equal(2 * wf, stmp.gd.integrate(stmp.ldos), 0.02)

stmp.write_3D(+4., f3dname)
print me + 'Integrals(unocc): 2 * wf, bias=',
print 2 * wf, stmp.gd.integrate(stmp.ldos)
equal(2 * wf, stmp.gd.integrate(stmp.ldos), 0.02)