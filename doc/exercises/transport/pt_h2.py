# creates: pt_h2.png

from ase import *
import os

a = 2.41
b = 0.9
c = 1.7
L = 7
l = L * 0.5
N = 5

atoms1 = Atoms('Pt',[(0.0,l,l)],pbc=True,cell=(a,L,L))
atoms1 *= (N,1,1)
atoms2 = atoms1.copy()
atoms2.translate(((N-1)*a+b+2*c,0,0))
h2 = Atoms('H2',[((N-1)*a+c,l,l),((N-1)*a+c+b,l,l)],pbc=True)

atoms = atoms1 + h2 + atoms2
atoms.set_cell([(2*N-1)*a+b+2*c,L,L])
pov_options = {'display' : False}
write('pt_h2.pov', atoms, show_unit_cell=2, pause=False,**pov_options)
os.system('povray pt_h2.ini')


