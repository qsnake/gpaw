import os
import sys
import pylab
n = 1
def show():
    global n
    pylab.savefig('x%d.png' % n)
    n += 1
pylab.show = show
os.system('rm -rf test; mkdir test')
os.chdir('test')
for dir, script in [
    ('aluminium', 'Al_fcc.py'),
    ('aluminium', 'Al_fcc_convergence.py'),
    ('surface', 'Al100.py'),
    ('surface', 'work_function.py'),
    ('diffusion', 'initial.py'),
    ('diffusion', 'solution.py'),
    ('diffusion', 'densitydiff.py'),
    ('vibrations', 'h2o.py'),
    ('vibrations', 'H2O_vib.py'),
    ('iron', 'ferro.py'),
    ('iron', 'anti.py'),
    ('iron', 'non.py'),
    ('iron', 'PBE.py'),
    ('band_structure', 'Na_band.py'),
    ('band_structure', 'plot_band.py'),
    ('wannier', 'si.py'),
    ('wannier', 'wannier-si.py'),
    ('wannier', 'benzene.py'),
    ('wannier', 'wannier-benzene.py'),
    ('stm', 'HAl100.py'),
    ]:
    execfile('../' + dir + '/' + script, {'k': 6})
for dir, script, arg in [
    ('stm', 'stm.py', 'HAl100.gpw')]:
    os.chdir(dir)
    sys.argv = ['', arg]
    execfile('../' + dir + '/' + script)
