import pylab as pl
from sys import argv
from ase.data import atomic_names as names
from gpaw.setup_data import SetupData
from gpaw.atom.all_electron import shoot

input = argv[1].split('.')
symbol = input[0]
xcname = input[-1]
setupname = 'paw'
if len(input) > 2:
    setupname = '.'.join(input[1:-1])

setup = SetupData(symbol, xcname, setupname)

colors = []
id_j = []
for j, n in enumerate(setup.n_j):
    if n == -1: n = '*'
    id_j.append(str(n) + 'spdf'[setup.l_j[j]])
    colors.append('kbrgymc'[j])

g = pl.arange(setup.ng, dtype=float)
r_g = setup.beta * g / (setup.ng - g)

print 'Element    :', symbol, names[setup.Z].title()
print 'Setup      :', '.'.join([setupname, xcname])
print 'Filename   :', setup.filename
print 'Fingerprint:', setup.fingerprint
print 'Electrons  : %d (%d core and %d valence)' % (setup.Z, setup.Nc, 
                                                    setup.Nv)
print ''
print 'Energy'
print '---------------------'
print 'Kinetic  : %10.4f' % setup.e_kinetic
print 'Potential: %10.4f' % setup.e_electrostatic
print 'XC       : %10.4f' % setup.e_xc
#print 'Core-kin : %10.4f' % e_kinetic_core
print '---------------------'
print 'Total    : %10.4f' % setup.e_total
print ''
print 'Partial Waves'
print 'id occ   eps  rcut'
for id, f, eps, rcut in zip(id_j, setup.f_j, setup.eps_j, setup.rcut_j):
    print '%s  %s  %6.3f %4.2f' % (id, f, eps, rcut)
print ''
print 'Compensation charge cutoff: %0.4f Bohr' % setup.rcgauss
#print 'Kinetic energy differences:\n', e_kin_jj


rmax = max(setup.rcut_j)
fig = pl.figure(1, figsize=(13, 6))
fig.subplots_adjust(left=.05, right=.95)

pl.subplot(131)
for phi_g, phit_g, id, color in zip(setup.phi_jg, setup.phit_jg, id_j, colors):
    pl.plot(r_g, phi_g, color + '-', label=id)
    pl.plot(r_g, phit_g, color + '--', label='_nolegend_')
pl.legend()
pl.axis('tight')
lim = pl.axis(xmin=0, xmax=rmax*1.2)
pl.plot([rmax, rmax], lim[2:], 'k--', label='_nolegend_')
pl.text(rmax, lim[2], r'$r_c$', ha='left', va='bottom', size=17)
pl.title('Partial Waves')
pl.xlabel('r [Bohr]')
pl.ylabel(r'$r^l\phi,\ r^l\tilde{\phi},\ \rm{[Bohr}^{-3/2}\rm{]}$')

pl.subplot(132)
for pt_g, id, color in zip(setup.pt_jg, id_j, colors):
    pl.plot(r_g, pt_g, color + '-', label=id)
pl.axis('tight')
lim = pl.axis(xmin=0, xmax=rmax*1.2)
pl.plot([rmax, rmax], lim[2:], 'k--', label='_nolegend_')
pl.text(rmax, lim[2], r'$r_c$', ha='left', va='bottom', size=17)
pl.legend()
pl.title('Projectors')
pl.xlabel('r [Bohr]')
pl.ylabel(r'$r^l\tilde{p},\ \rm{[Bohr}^{-3/2}\rm{]}$')

pl.subplot(133)
pl.plot(r_g, setup.nc_g, colors[0], label=r'$n_c$')
pl.plot(r_g, setup.nct_g, colors[1], label=r'$\tilde{n}_c$')
pl.plot(r_g, setup.vbar_g, colors[2], label=r'$\bar{v}$')
pl.axis('tight')
lim = pl.axis(xmin=0, xmax=rmax*1.2,ymax=max([max(setup.nct_g), 
                                              max(setup.vbar_g)]))
pl.plot([rmax, rmax], lim[2:], 'k--', label='_nolegend_')
pl.text(rmax, lim[2], r'$r_c$', ha='left', va='bottom', size=17)
pl.legend()
pl.title('Other stuff')
pl.xlabel('r [Bohr]')

pl.show()
