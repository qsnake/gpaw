from math import sqrt, pi

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
dr_g = setup.beta * setup.ng / (setup.ng - g)**2
d2gdr2 = -2 * setup.ng * setup.beta / (setup.beta + r_g)**3
rmax = max(setup.rcut_j)
gmax = 1 + int(rmax * setup.ng / (rmax + setup.beta))
rcutcomp = sqrt(10) * setup.rcgauss

# Find Fourier-filter cutoff radius:
g = setup.ng - 1
while setup.pt_jg[0][g] == 0.0:
    g -= 1
gcutfilter = g + 1
rcutfilter = r_g[gcutfilter]

# Find cutoff for core density:
if setup.Nc == 0:
    rcore = 0.5
else:
    N = 0.0
    g = setup.ng - 1
    while N < 1e-7:
        N += sqrt(4 * pi) * setup.nc_g[g] * r_g[g]**2 * dr_g[g]
        g -= 1
    rcore = r_g[g]

# Construct logarithmic derivatives


print 'Element    : %s (%s)' % (names[setup.Z].title(), symbol)
print 'Setup      :', '.'.join([setupname, xcname])
print 'Filename   :', setup.filename
print 'Fingerprint:', setup.fingerprint
print 'Electrons  : %d (%d core and %d valence)' % (setup.Z, setup.Nc, 
                                                    setup.Nv)
print ''
print 'Energy'
print '-----------------------'
print 'Kinetic  : %12.4f' % setup.e_kinetic
print 'Potential: %12.4f' % setup.e_electrostatic
print 'XC       : %12.4f' % setup.e_xc
print '-----------------------'
print 'Total    : %12.4f' % setup.e_total
print ''
print 'Partial Wave Valence States'
print 'id occ eigenvals cutoff'
for id, f, eps, rcut in zip(id_j, setup.f_j, setup.eps_j, setup.rcut_j):
    print '%s  %s  %6.3f Ha %4.2f Bohr' % (id, f, eps, rcut)
print ''
print 'Cutoffs: %4.2f(comp), %4.2f(filt), %4.2f(core) Bohr' % (
    rcutcomp, rcutfilter, rcore)


fig = pl.figure(1, figsize=(13, 11))
fig.subplots_adjust(left=.06, right=.98, top=.97, bottom=.04)

pl.subplot(221)
for phi_g, phit_g, id, color in zip(setup.phi_jg, setup.phit_jg, id_j, colors):
    pl.plot(r_g, r_g * phi_g, color + '-', label=id)
    pl.plot(r_g, r_g * phit_g, color + '--', label='_nolegend_')
pl.legend()
pl.axis('tight')
lim = pl.axis(xmin=0, xmax=rmax*1.2)
pl.plot([rmax, rmax], lim[2:], 'k--', label='_nolegend_')
pl.text(rmax, lim[2], r'$r_c$', ha='left', va='bottom', size=17)
pl.title('Partial Waves')
pl.xlabel('r [Bohr]')
pl.ylabel(r'$r^{l+1}\phi,\ r^{l+1}\tilde{\phi},\ \rm{[Bohr}^{-3/2}\rm{]}$')

pl.subplot(222)
for pt_g, id, color in zip(setup.pt_jg, id_j, colors):
    pl.plot(r_g, r_g * pt_g, color + '-', label=id)
pl.axis('tight')
lim = pl.axis(xmin=0, xmax=rmax*1.2)
pl.plot([rmax, rmax], lim[2:], 'k--', label='_nolegend_')
pl.text(rmax, lim[2], r'$r_c$', ha='left', va='bottom', size=17)
pl.legend()
pl.title('Projectors')
pl.xlabel('r [Bohr]')
pl.ylabel(r'$r^{l+1}\tilde{p},\ \rm{[Bohr}^{-3/2}\rm{]}$')

pl.subplot(223)
#lim = pl.axis('tight')
pl.legend(('s', 'p', 'd'))
pl.title('Logarithmic Derivatives')
pl.ylabel('logarithmic Derivative [arb. units]')
pl.xlabel('Energy [Hartree]')


pl.subplot(224)
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
