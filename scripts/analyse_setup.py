from math import sqrt, pi

import pylab as pl
from ase.data import atomic_names as names
from gpaw.setup_data import SetupData


class TxtWriter:
    def __init__(self, symbol):
        self.file = open(symbol + '.rst', 'w')

    def __call__(self, s, fill=None):
        if fill is not None:
            s = s + (fill - len(s)) * ' ' + '|'
        print >> self.file, s

def make_page(symbol, xcname, setupname, show=False):
    txt = TxtWriter(symbol)
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
    rlog = r_g[gmax + 10]
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
    eig = {}
    for l, e in zip(setup.l_j, setup.eps_j):
        eig['spdf'[l]] = eig.get('spdf'[l], []) + [e,]
    logd = {}
    ref = []
    try: # Only works if logarithmic derivatives have been predetermined
        loge = pl.load(symbol + '.ae.ld.s')[:, 0]
        logd['sae'] = pl.load(symbol + '.ae.ld.s')[:, 1]
        logd['pae'] = pl.load(symbol + '.ae.ld.p')[:, 1]
        logd['dae'] = pl.load(symbol + '.ae.ld.d')[:, 1]
        logd['sps'] = pl.load(symbol + '.ps.ld.s')[:, 1]
        logd['pps'] = pl.load(symbol + '.ps.ld.p')[:, 1]
        logd['dps'] = pl.load(symbol + '.ps.ld.d')[:, 1]

        for l, e in zip(setup.l_j, setup.eps_j):
            i = loge.searchsorted(e)
            ref.append([loge[i], logd['spdf'[l] + 'ae'][i]])
        ref = pl.array(ref)
    except:
        pass

    txt('+-------------+------------------+')
    txt('| Setup Details                  |')
    txt('+=============+==================+')
    txt('| Element     | %s (%s)' % (names[setup.Z].title(), symbol), 33)
    txt('+-------------+------------------+')
    txt('| Setup       | %s' % '.'.join([setupname, xcname]), 33)
    txt('+-------------+------------------+')
    ## txt('| Filename    | %s' % setup.filename, 33)
    ## txt('+-------------+------------------+')
    ## txt('| Fingertxt(| %s' % setup.fingerprint, 33)
    ## txt('+-------------+------------------+')
    txt('| |           | %d (total)' % setup.Z, 33)
    txt('+ |           +------------------+')
    txt('| | Electrons | %d (valence)' % setup.Nv, 33)
    txt('+ |           +------------------+')
    txt('| |           | %d (core)' % setup.Nc, 33)
    txt('+-------------+------------------+')
    txt('| |           | %4.2f Bohr (comp) |' % rcutcomp)
    txt('+ |           +------------------+')
    txt('| | Cutoffs   | %4.2f Bohr (filt) |' % rcutfilter)
    txt('+ |           +------------------+')
    txt('| |           | %4.2f Bohr (core) |' % rcore)
    txt('+-------------+------------------+')
    txt('')
    txt('========= ===============')
    txt('Energy Contributions ')
    txt('=========================')
    txt('Kinetic   %12.4f Ha' % setup.e_kinetic)
    txt('Potential %12.4f Ha' % setup.e_electrostatic)
    txt('XC        %12.4f Ha' % setup.e_xc)
    txt('--------- ---------------')
    txt('Total     %12.4f Ha' % setup.e_total)
    txt('========= ===============')
    txt('')
    txt('=== === ========= =========')
    txt('Partial Wave Valence States')
    txt('===========================')
    txt('id  occ eigenvals cutoff')
    txt('--- --- --------- ---------')
    for id, f, eps, rcut in zip(id_j, setup.f_j, setup.eps_j, setup.rcut_j):
        txt('%3s  %s  %6.3f Ha %4.2f Bohr' % (id.replace('*','\*'), f,
                                              eps, rcut))
    txt('=== === ========= =========')
    txt('')
    txt('.. figure:: %s_waves.png' % symbol)
    txt('')
    txt('Back to the `periodic table`_.')
    txt('')
    txt('.. _periodic table: Setups_')
    txt('')

    dpi = 80
    fig = pl.figure(1, figsize=(9.2, 8.4), dpi=dpi)
    fig.subplots_adjust(left=.075, right=.99, top=.96, bottom=.06)

    pl.subplot(221)
    for phi_g, phit_g, id, color in zip(setup.phi_jg, setup.phit_jg,
                                        id_j, colors):
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
    if len(logd) != 0:
        pl.plot(ref[:, 0], ref[:, 1], 'ko', label='_nolegend_')
        for id, color in zip('spd', colors):
            pl.plot(loge, logd[id+'ae'], linestyle='-', color=color, label=id)
            pl.plot(loge, logd[id+'ps'], linestyle='--', color=color,
                    label='_nolegend_')
    lim = pl.axis('tight')
    pl.legend()
    pl.title('Logarithmic Derivatives')
    pl.ylabel('log. deriv. at r=%s Bohr' % rlog)
    pl.xlabel('Energy [Hartree]')

    pl.subplot(224)
    pl.plot(r_g, setup.nc_g, colors[0], label=r'$n_c$')
    pl.plot(r_g, setup.nct_g, colors[1], label=r'$\tilde{n}_c$')
    pl.plot(r_g, setup.vbar_g, colors[2], label=r'$\bar{v}$')
    pl.axis('tight')
    lim = pl.axis(xmin=0, xmax=rmax * 1.2, ymax=max([max(setup.nct_g), 
                                                     max(setup.vbar_g)]))
    pl.plot([rmax, rmax], lim[2:], 'k--', label='_nolegend_')
    pl.text(rmax, lim[2], r'$r_c$', ha='left', va='bottom', size=17)
    pl.legend()
    pl.title('Other stuff')
    pl.xlabel('r [Bohr]')

    pl.savefig('%s_waves.png' % symbol, dpi=dpi)
    
    if show:
        pl.show()


if __name__ == '__main__':
    from sys import argv
    input = argv[1].split('.')
    symbol = input[0]
    xcname = input[-1]
    setupname = 'paw'
    if len(input) > 2:
        setupname = '.'.join(input[1:-1])

    make_page(symbol, xcname, setupname, show=True)
