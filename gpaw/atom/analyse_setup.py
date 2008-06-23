from math import sqrt, pi

import pylab as plt
import numpy as np

from ase.data import atomic_names as names


def analyse(generator, show=False):
    gen = generator
    symbol = gen.symbol
    
    fd = open(symbol + '.rst', 'w')
    def txt(s):
        fd.write(s + '\n')

    colors = []
    id_j = []
    for j, n in enumerate(gen.vn_j):
        if n == -1: n = '*'
        id_j.append(str(n) + 'spdf'[gen.vl_j[j]])
        colors.append('kbrgymc'[j])

    r_g = gen.r
    g = np.arange(gen.N)
    dr_g = gen.beta * gen.N / (gen.N - g)**2
    rmax = max(gen.rcut_l)
    rcutcomp = gen.rcutcomp
    rcutfilter = gen.rcutfilter

    # Find cutoff for core density:
    if gen.Nc == 0:
        rcore = 0.5
    else:
        N = 0.0
        g = gen.N - 1
        while N < 1e-7:
            N += sqrt(4 * pi) * gen.nc[g] * r_g[g]**2 * dr_g[g]
            g -= 1
        rcore = r_g[g]

    # Construct logarithmic derivatives
    if len(gen.logd) > 0:
        rlog = gen.rlog
        logd = gen.logd
        elog = gen.elog
        ref = []
        for l, e in zip(gen.vl_j, gen.ve_j):
            i = gen.loge.searchsorted(e)
            ref.append((elog[i], logd[l][1][i]))
        ref = np.array(ref)

    txt('=========  =========')
    txt('Electrons')
    txt('=========  =========')
    txt('total      %d' % gen.Z)
    txt('valence    %d' % gen.Nv)
    txt('core       %d' % gen.Nc)
    txt('=========  =========')
    txt('')
    txt('====================  ==============')
    txt('Cutoffs')
    txt('====================  ==============')
    txt('compensation charges  %4.2f Bohr' % rcutcomp)
    txt('filtering             %4.2f Bohr' % rcutfilter)
    txt('core density          %4.2f Bohr' % rcore)
    txt('====================  ==============')
    txt('')
    txt('========= ===============')
    txt('Energy Contributions ')
    txt('=========================')
    txt('Kinetic   %12.4f Ha' % gen.Ekin)
    txt('Potential %12.4f Ha' % gen.Epot)
    txt('XC        %12.4f Ha' % gen.Exc)
    txt('--------- ---------------')
    txt('Total     %12.4f Ha' % gen.ETotal)
    txt('========= ===============')
    txt('')
    txt('=== === ========= =========')
    txt('Partial Wave Valence States')
    txt('===========================')
    txt('id  occ eigenvals cutoff')
    txt('--- --- --------- ---------')
    for id, f, eps, l in zip(id_j, gen.vf_j, gen.ve_j, gen.vl_j):
        txt('%3s  %s  %6.3f Ha %4.2f Bohr' % (id.replace('*','\*'), f,
                                              eps, gen.rcut_l[l]))
    txt('=== === ========= =========')
    txt('')
    txt('.. figure:: %s_waves.png' % symbol)
    txt('')
    txt('Back to the `periodic table`_.')
    txt('')
    txt('.. _periodic table: Setups_')
    txt('')

    dpi = 80
    fig = plt.figure(1, figsize=(9.2, 8.4), dpi=dpi)
    fig.subplots_adjust(left=.075, right=.99, top=.96, bottom=.06)

    plt.subplot(221)
    for phi_g, phit_g, id, color in zip(gen.vu_j, gen.vs_j,
                                        id_j, colors):
        plt.plot(r_g, r_g * phi_g, color + '-', label=id)
        plt.plot(r_g, r_g * phit_g, color + '--', label='_nolegend_')
    plt.legend()
    plt.axis('tight')
    lim = plt.axis(xmin=0, xmax=rmax*1.2)
    plt.plot([rmax, rmax], lim[2:], 'k--', label='_nolegend_')
    plt.text(rmax, lim[2], r'$r_c$', ha='left', va='bottom', size=17)
    plt.title('Partial Waves')
    plt.xlabel('r [Bohr]')
    plt.ylabel(r'$r^{l+1}\phi,\ r^{l+1}\tilde{\phi},\ \rm{[Bohr}^{-3/2}\rm{]}$')

    plt.subplot(222)
    for pt_g, id, color in zip(gen.vq_j, id_j, colors):
        plt.plot(r_g, r_g * pt_g, color + '-', label=id)
    plt.axis('tight')
    lim = plt.axis(xmin=0, xmax=rmax*1.2)
    plt.plot([rmax, rmax], lim[2:], 'k--', label='_nolegend_')
    plt.text(rmax, lim[2], r'$r_c$', ha='left', va='bottom', size=17)
    plt.legend()
    plt.title('Projectors')
    plt.xlabel('r [Bohr]')
    plt.ylabel(r'$r^{l+1}\tilde{p},\ \rm{[Bohr}^{-3/2}\rm{]}$')

    plt.subplot(223)
    if len(gen.logd) > 0:
        plt.plot(ref[:, 0], ref[:, 1], 'ko', label='_nolegend_')
        for l, color in enumerate(colors[:3]):
            id = 'spd'[l]
            plt.plot(loge, logd[l][0], linestyle='-', color=color, label=id)
            plt.plot(loge, logd[l][1], linestyle='--', color=color,
                     label='_nolegend_')
        plt.ylabel('log. deriv. at r=%s Bohr' % rlog)
    lim = plt.axis('tight')
    plt.legend()
    plt.title('Logarithmic Derivatives')
    plt.xlabel('Energy [Hartree]')

    plt.subplot(224)
    plt.plot(r_g, gen.nc, colors[0], label=r'$n_c$')
    plt.plot(r_g, gen.nct, colors[1], label=r'$\tilde{n}_c$')
    #plt.plot(r_g, gen.vbar_g, colors[2], label=r'$\bar{v}$')
    plt.axis('tight')
    lim = plt.axis(xmin=0, xmax=rmax * 1.2, ymax=max(gen.nct))
    plt.plot([rmax, rmax], lim[2:], 'k--', label='_nolegend_')
    plt.text(rmax, lim[2], r'$r_c$', ha='left', va='bottom', size=17)
    plt.legend()
    plt.title('Other stuff')
    plt.xlabel('r [Bohr]')

    plt.savefig('%s_waves.png' % symbol, dpi=dpi)
    
    if show:
        plt.show()


if __name__ == '__main__':
    from sys import argv
    input = argv[1].split('.')
    symbol = input[0]
    xcname = input[-1]
    setupname = 'paw'
    if len(input) > 2:
        setupname = '.'.join(input[1:-1])

    make_page(symbol, xcname, setupname, show=True)
