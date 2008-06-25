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
        colors.append('krgbymc'[j])

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
            i = elog.searchsorted(e)
            ref.append((elog[i], logd[l][1][i]))
        ref = np.array(ref)

    table1 = ['=========  =========',
              'total      %d' % gen.Z,
              'valence    %d' % gen.Nv,
              'core       %d' % gen.Nc,
              '=========  =========']
    table2 = ['====================  ==============',
              'compensation charges  %4.2f Bohr' % rcutcomp,
              'filtering             %4.2f Bohr' % rcutfilter,
              'core density          %4.2f Bohr' % rcore,
              '====================  ==============']
    table3 = ['========= ===============',
              'Kinetic   %12.4f Ha' % gen.Ekin,
              'Potential %12.4f Ha' % gen.Epot,
              'XC        %12.4f Ha' % gen.Exc,
              '--------- ---------------',
              'Total     %12.4f Ha' % gen.ETotal,
              '========= ===============']
    table4 = ['=== === ========= =========',
              'id  occ eigenvals cutoff',
              '--- --- --------- ---------']
    for id, f, eps, l in zip(id_j, gen.vf_j, gen.ve_j, gen.vl_j):
        table4.append('%3s  %d  %6.3f Ha %4.2f Bohr' % (id.replace('*','\*'), f,
                                                        eps, gen.rcut_l[l]))
    table4.append('=== === ========= =========')

    dpi = 80
    fig = plt.figure(figsize=(8.0, 8.0), dpi=dpi)
    fig.subplots_adjust(left=.075, right=.99, top=.96, bottom=.06)

    plt.subplot(221)
    for phi_g, phit_g, id, color in zip(gen.vu_j, gen.vs_j,
                                        id_j, colors):
        plt.plot(r_g, phi_g, color + '-', label=id)
        plt.plot(r_g, phit_g, color + '--', label='_nolegend_')
    plt.legend(loc='best')
    plt.axis('tight')
    lim = plt.axis(xmin=0, xmax=rmax * 2)
    plt.plot([rmax, rmax], lim[2:], 'k--', label='_nolegend_')
    plt.text(rmax, lim[2], r'$r_c$', ha='left', va='bottom', size=17)
    plt.title('Partial Waves')
    plt.xlabel('r [Bohr]')
    plt.ylabel(r'$r\phi,\ r\tilde{\phi},\ \rm{[Bohr}^{-1/2}\rm{]}$')

    plt.subplot(222)
    for pt_g, id, color in zip(gen.vq_j, id_j, colors):
        plt.plot(r_g, pt_g, color + '-', label=id)
    plt.axis('tight')
    lim = plt.axis(xmin=0, xmax=rmax*1.2)
    plt.plot([rmax, rmax], lim[2:], 'k--', label='_nolegend_')
    plt.text(rmax, lim[2], r'$r_c$', ha='left', va='bottom', size=17)
    plt.legend(loc='best')
    plt.title('Projectors')
    plt.xlabel('r [Bohr]')
    plt.ylabel(r'$r\tilde{p},\ \rm{[Bohr}^{-1/2}\rm{]}$')

    plt.subplot(223)
    if len(gen.logd) > 0:
        plt.plot(ref[:, 0], ref[:, 1], 'ko', label='_nolegend_')
        for l, color in enumerate(colors[:3]):
            id = 'spd'[l]
            plt.plot(elog, logd[l][0], linestyle='-', color=color, label=id)
            plt.plot(elog, logd[l][1], linestyle='--', color=color,
                     label='_nolegend_')
        plt.ylabel('log. deriv. at r=%.2f Bohr' % rlog)
    lim = plt.axis('tight')
    plt.legend(loc='best')
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
    plt.legend(loc='best')
    plt.title('Other stuff')
    plt.xlabel('r [Bohr]')

    plt.savefig('%s-setup.png' % symbol, dpi=dpi)
    
    if show:
        plt.show()

    return (table1, table2, table3, table4)


if __name__ == '__main__':
    from sys import argv
    input = argv[1].split('.')
    symbol = input[0]
    xcname = input[-1]
    setupname = 'paw'
    if len(input) > 2:
        setupname = '.'.join(input[1:-1])

    make_page(symbol, xcname, setupname, show=True)
