import os

from ase import *

from gpaw import GPAW
from gpaw.cluster import Cluster 
from gpaw.analyse.simple_stm import SimpleStm
from gpaw.utilities import equal

load=True
load=False
txt = '/dev/null'

LiH = Atoms([Atom('Li', [.0, .0, .41]),
             Atom('H', [.0, .0, -1.23]),
             ], cell=[5, 5, 6.5])
LiH.center()

# finite system
if not load:
    LiH.set_pbc(False)
    cf = GPAW(spinpol=True, nbands=2, h=.3, txt=txt)
    LiH.set_calculator(cf)
    LiH.get_potential_energy()
#    cf.write('LiH_wfs.gpw', 'all')
else:
    cf = GPAW('LiH-wfs.gpw', txt=txt)

f3dname = 'stm3d.plt'

stmf = SimpleStm(cf)
stmf.write_3D([1,0,0], f3dname)
wf = stmf.gd.integrate(stmf.ldos)

stm2 = SimpleStm(f3dname)
wf2 = stm2.gd.integrate(stm2.ldos)
print 'Integrals: written, read=', wf, wf2
equal(wf, wf2, 1.e-7)

#stm2.scan_const_density(dens, [1,0,0])
#stm2.write('stm2_raw.dat')

# periodic system
if not load:
    LiH.set_pbc(True)
    cp = GPAW(spinpol=True, nbands=2, h=.3, kpts=(2,2,2), txt=txt)
    LiH.set_calculator(cp)
    LiH.get_potential_energy()
#    cp.write('LiH-8kpts_wfs.gpw', 'all')
else:
    cp = GPAW('LiH-8kpts_wfs.gpw', txt=txt)

stmp = SimpleStm(cp)

stmp.write_3D(-4., f3dname)
print 'Integrals(occ): 2 * wf, bias=', 2 * wf, stmp.gd.integrate(stmp.ldos)
equal(2 * wf, stmp.gd.integrate(stmp.ldos), 0.02)

stmp.write_3D(+4., f3dname)
print 'Integrals(unocc): 2 * wf, bias=', 2 * wf, stmp.gd.integrate(stmp.ldos)
equal(2 * wf, stmp.gd.integrate(stmp.ldos), 0.02)
