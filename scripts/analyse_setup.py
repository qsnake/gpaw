import pylab as pl
#import numpy as npy

from ase.data import atomic_names as names
#from ASE.ChemicalElements.name import names
from gpaw.read_setup import PAWXMLParser

symbol = 'C'
setupname = 'PBE'

# Load setup data from XML file:
(Z, Nc, Nv,
 e_total, e_kinetic, e_electrostatic, e_xc,
 e_kinetic_core,
 n_j, l_j, f_j, eps_j, rcut_j, id_j,
 ng, beta,
 nc_g, nct_g, vbar_g, rcgauss,
 phi_jg, phit_jg, pt_jg,
 e_kin_jj, X_p, ExxC,
 tauc_g, tauct_g,
 fingerprint,
 filename,
 core_hole_state,
 fcorehole,
 core_hole_e,
 core_hole_e_kin,
 core_response) = PAWXMLParser().parse(symbol, setupname)

colors = []
for j, n in enumerate(n_j):
    if n == -1: n = '*'
    id_j[j] = str(n) + 'spdf'[l_j[j]]
    colors.append('kbrgymc'[j])

g = pl.arange(ng, dtype=float)
r_g = beta * g / (ng - g)

print 'Element    :', symbol, names[Z].title()
print 'Setup      :', setupname
print 'Filename   :', filename
print 'Fingerprint:', fingerprint
print 'Electrons  : %d (%d core and %d valence)' % (Z, Nc, Nv)
print ''
print 'Energy'
print '---------------------'
print 'Kinetic  : %10.4f' % e_kinetic
print 'Potential: %10.4f' % e_electrostatic
print 'XC       : %10.4f' % e_xc
#print 'Core-kin : %10.4f' % e_kinetic_core
print '---------------------'
print 'Total    : %10.4f' % e_total
print ''
print 'Partial Waves'
print 'id occ   eps  rcut'
for id, f, eps, rcut in zip(id_j, f_j, eps_j, rcut_j):
    print '%s  %s  %6.3f %4.2f' % (id, f, eps, rcut)
print ''
print 'Compensation charge cutoff: %0.4f Bohr' % rcgauss
#print 'Kinetic energy differences:\n', e_kin_jj

rmax = max(rcut_j)
fig = pl.figure(1, figsize=(13, 6))
fig.subplots_adjust(left=.05, right=.95)

pl.subplot(131)
for phi_g, phit_g, id, color in zip(phi_jg, phit_jg, id_j, colors):
    pl.plot(r_g, phi_g, color + '-', label=id)
    pl.plot(r_g, phit_g, color + '--', label='_nolegend_')
pl.legend()
pl.axis('tight')
lim = pl.axis(xmin=0, xmax=rmax*1.2)
pl.plot([rmax, rmax], lim[2:], 'k--', label='_nolegend_')
pl.text(rmax, lim[2], r'$r_c$', ha='left', va='bottom', size=17)
pl.title('Partial Waves')
pl.xlabel('r [Bohr]')
pl.ylabel(r'$r^l\phi, r^l\tilde{\phi}, \rm{[Bohr}^{-3/2}\rm{]}$')

pl.subplot(132)
for pt_g, id, color in zip(pt_jg, id_j, colors):
    pl.plot(r_g, pt_g, color + '-', label=id)
pl.axis('tight')
lim = pl.axis(xmin=0, xmax=rmax*1.2)
pl.plot([rmax, rmax], lim[2:], 'k--', label='_nolegend_')
pl.text(rmax, lim[2], r'$r_c$', ha='left', va='bottom', size=17)
pl.legend()
pl.title('Projectors')
pl.xlabel('r [Bohr]')
pl.ylabel(r'$r^l\tilde{p}, \rm{[Bohr}^{-3/2}\rm{]}$')

pl.subplot(133)
pl.plot(r_g, nc_g, colors[0], label=r'$n_c$')
pl.plot(r_g, nct_g, colors[1], label=r'$\tilde{n}_c$')
pl.plot(r_g, vbar_g, colors[2], label=r'$\bar{v}$')
pl.axis('tight')
lim = pl.axis(xmin=0, xmax=rmax*1.2,ymax=max([max(nct_g), max(vbar_g)]))
pl.plot([rmax, rmax], lim[2:], 'k--', label='_nolegend_')
pl.text(rmax, lim[2], r'$r_c$', ha='left', va='bottom', size=17)
pl.legend()
pl.title('Other stuff')
pl.xlabel('r [Bohr]')

pl.show()
