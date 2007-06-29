# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.
from math import log, pi

import Numeric as num

import gpaw.mpi as mpi


def xas(paw):
    assert not mpi.parallel
    nocc = paw.nvalence / 2 # restricted - for now
    for nucleus in paw.nuclei:
        if nucleus.setup.fcorehole != 0.0:
            P_ni = nucleus.P_uni[0, nocc:] 
            A_ci = nucleus.setup.A_ci
            ach = nucleus.a
            break

    #print 'core hole atom', ach
    eps_n = paw.kpt_u[0].eps_n[nocc:] * paw.Ha
    w_cn = num.dot(A_ci, num.transpose(P_ni))**2
    return eps_n, w_cn


def plot_xas(eps_n, w_cn, fwhm=0.5, linbroad=None, N=1000):
    # returns stick spectrum, e_stick and a_stick
    # and broadened spectrum, e, a
    # linbroad = [0.5, 540, 550]
    eps_n_tmp = eps_n.copy()
    emin = min(eps_n_tmp) - 2 * fwhm
    emax = max(eps_n_tmp) + 2 * fwhm

    e = emin + num.arange(N + 1) * ((emax - emin) / N)
    a = num.zeros(N + 1, num.Float)

    e_stick = eps_n_tmp.copy()
    a_stick = num.zeros(len(eps_n_tmp), num.Float)


    if linbroad == None:
        #constant broadening fwhm
        alpha = 4*log(2) / fwhm**2
        for n, eps in enumerate(eps_n_tmp):
            x = -alpha * (e - eps)**2
            x = num.clip(x, -100.0, 100.0)
            w = sum(w_cn[:, n])
            a += w * (alpha / pi)**0.5 * num.exp(x)
            a_stick[n] = sum(w_cn[:, n])
    else:
        # constant broadening fwhm until linbroad[1] and a constant broadening
        # over linbroad[2] with fwhm2= linbroad[0]
        fwhm2 = linbroad[0]
        lin_e1 = linbroad[1]
        lin_e2 = linbroad[2]
        for n, eps in enumerate(eps_n_tmp):
            if eps < lin_e1:
                alpha = 4*log(2) / fwhm**2
            elif eps <=  lin_e2:
                fwhm_lin = fwhm + (eps - lin_e1) * (fwhm2 - fwhm) / (lin_e2 - lin_e1)
                alpha = 4*log(2) / fwhm_lin**2
            elif eps >= lin_e2:
                alpha =  4*log(2) / fwhm2**2

            x = -alpha * (e - eps)**2
            x = num.clip(x, -100.0, 100.0)
            w = sum(w_cn[:, n])
            a += w * (alpha / pi)**0.5 * num.exp(x)
            a_stick[n] = sum(w_cn[:, n])
        
    return e_stick, a_stick, e, a
