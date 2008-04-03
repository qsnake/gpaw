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
logr = 1.475
## logd = {}
## loge = pl.load('B.ae.ld.s')[:, 0]
## logd['sae'] = pl.load('B.ae.ld.s')[:, 1]
## logd['pae'] = pl.load('B.ae.ld.p')[:, 1]
## logd['dae'] = pl.load('B.ae.ld.d')[:, 1]
## logd['sps'] = pl.load('B.ps.ld.s')[:, 1]
## logd['pps'] = pl.load('B.ps.ld.p')[:, 1]
## logd['dps'] = pl.load('B.ps.ld.d')[:, 1]
## eig = [[-0.347, 0.653], # s
##        [-0.133, 0.867], # p
##        [-0.000,]]       # d

def pfill(s, l):
    print s + (l-len(s)) * ' ' + '|'

print '+-------------+------------------+'
print '| Setup Details                  |'
print '+=============+==================+'
pfill('| Element     | %s (%s)' % (names[setup.Z].title(), symbol), 33)
print '+-------------+------------------+'
pfill('| Setup       | %s' % '.'.join([setupname, xcname]), 33)
print '+-------------+------------------+'
## pfill('| Filename    | %s' % setup.filename, 33)
## print '+-------------+------------------+'
## pfill('| Fingerprint | %s' % setup.fingerprint, 33)
## print '+-------------+------------------+'
pfill('| |           | %d (total)' % setup.Z, 33)
print '+ |           +------------------+'
pfill('| | Electrons | %d (valence)' % setup.Nv, 33)
print '+ |           +------------------+'
pfill('| |           | %d (core)' % setup.Nc, 33)
print '+-------------+------------------+'
print '| |           | %4.2f Bohr (comp) |' % rcutcomp
print '+ |           +------------------+'
print '| | Cutoffs   | %4.2f Bohr (filt) |' % rcutfilter
print '+ |           +------------------+'
print '| |           | %4.2f Bohr (core) |' % rcore
print '+-------------+------------------+'
print ''
print '========= ==============='
print 'Energy Contributions '
print '========================='
print 'Kinetic   %12.4f Ha' % setup.e_kinetic
print 'Potential %12.4f Ha' % setup.e_electrostatic
print 'XC        %12.4f Ha' % setup.e_xc
print '--------- ---------------'
print 'Total     %12.4f Ha' % setup.e_total
print '========= ==============='
print ''
print '=== === ========= ========='
print 'Partial Wave Valence States'
print '==========================='
print 'id  occ eigenvals cutoff'
print '--- --- --------- ---------'
for id, f, eps, rcut in zip(id_j, setup.f_j, setup.eps_j, setup.rcut_j):
    print '%3s  %s  %6.3f Ha %4.2f Bohr' % (id.replace('*','\*'), f, eps, rcut)
print '=== === ========= ========='
print
print '.. figure:: waves.png'
print 
print 'Back to `periodic table`_.'
print 
print '.. _periodic table: Setups_'
print

dpi = 80
fig = pl.figure(1, figsize=(9.2, 8.4), dpi=dpi)
fig.subplots_adjust(left=.075, right=.99, top=.96, bottom=.06)

pl.subplot(221)
for phi_g, phit_g, id, color in zip(setup.phi_jg, setup.phit_jg, id_j, colors):
    pl.plot(r_g, r_g * phi_g, color + '-', label=id)
    pl.plot(r_g, r_g * phit_g, color + '--', label='_nolegend_')

pl.legend()
pl.axis('tight')
lim = pl.axis(xmin=0, xmax=rmax*1.2, ymin=-.6, ymax=.7)
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
## for id, color, es in zip('spd', colors, eig):
##     refx = []
##     refy = []
##     for e in es:
##         i = loge.searchsorted(e)
##         refx.append(loge[i])
##         refy.append(logd[id+'ae'][i])
    
##     pl.plot(refx, refy, 'ko', label='_nolegend_')
##     pl.plot(loge, logd[id+'ae'],linestyle='-',color=color,label=id)
##     pl.plot(loge, logd[id+'ps'],linestyle='--',color=color,label='_nolegend_')

## lim = pl.axis('tight')
pl.legend()
pl.title('Logarithmic Derivatives')
pl.ylabel('log. deriv. at r=%s Bohr' % logr)
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

pl.savefig('waves.png', dpi=dpi)
pl.show()
