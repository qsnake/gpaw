from gpaw.xc_functional import XCFunctional
from gpaw import setup_paths
from math import pi
import numpy as np

nspins = 2
for name in ['LDA', 'PBE', 'revPBE', 'RPBE',
             'LDAx', 'revPBEx', 'RPBEx',
             'None-C_PW', 'TPSS', 'M06L',
             'HCTH407',
             ]:
    libxc = XCFunctional(name, nspins)
    lxc_xc = libxc.calculate_xcenergy
    calc_sp = libxc.calculate_spinpolarized
    na = 2.0
    nb = 1.0
    print na, nb
    if (nb > 0.0): assert (nspins == 2)
    sigma0 = 2.0 # (0.0, 1.0, 1.0)
    sigma1 = 2.0
    sigma2 = 5.0 # (1.0, 2.0, 0.0)
    taua=(3.*pi**2)**(2./3.)*na**(5./3.)/2.*sigma0
    taub=(3.*pi**2)**(2./3.)*nb**(5./3.)/2.*sigma2
    if ((sigma1 > 0.0) or (sigma2 > 0.0)): assert (nspins == 2)

    d = 0.000001

    na_g = np.array([na])
    nb_g = np.array([nb])
    sigma0_g = np.array([sigma0])
    sigma1_g = np.array([sigma1])
    sigma2_g = np.array([sigma2])
    a2_g = np.array([2*sigma1+sigma0+sigma2])

    dExcdsigma0_g = np.zeros((1))
    dExcdsigma1_g = np.zeros((1))
    dExcdsigma2_g = np.zeros((1))

    exc_g = np.zeros_like(na_g)
    dExcdna_g = np.zeros_like(na_g)
    dExcdnb_g = np.zeros_like(na_g)

    taua_g = np.array([taua])
    taub_g = np.array([taub])
    dExcdtaua_g = np.zeros_like(taua_g)
    dExcdtaub_g = np.zeros_like(taua_g)

    calc_sp(exc_g, na_g, dExcdna_g, nb_g, dExcdnb_g,
            a2_g, sigma0_g, sigma2_g,
            dExcdsigma1_g, dExcdsigma0_g, dExcdsigma2_g,
            taua_g, taub_g, dExcdtaua_g, dExcdtaub_g)

    (exc, ex, ec, d_exc, d_ex, d_ec) = lxc_xc(na, nb, sigma0, sigma1, sigma2, taua, taub)
    # for definitions see c/libxc.c
    dExcdna = d_exc[0]
    dExcdnb = d_exc[1]
    dExcdsigma0 = d_exc[2]
    dExcdsigma1 = d_exc[3]
    dExcdsigma2 = d_exc[4]
    dExcdtaua = d_exc[5]
    dExcdtaub = d_exc[6]

    dExdna = d_ex[0]
    dExdnb = d_ex[1]
    dExdsigma0 = d_ex[2]
    dExdsigma1 = d_ex[3]
    dExdsigma2 = d_ex[4]
    dExdtaua = d_ex[5]
    dExdtaub = d_ex[6]

    dEcdna = d_ec[0]
    dEcdnb = d_ec[1]
    dEcdsigma0 = d_ec[2]
    dEcdsigma1 = d_ec[3]
    dEcdsigma2 = d_ec[4]
    dEcdtaua = d_ec[5]
    dEcdtaub = d_ec[6]

    dExcdna_N=((lxc_xc(na + d, nb, sigma0, sigma1, sigma2,taua,taub)[0] - lxc_xc(na - d, nb, sigma0, sigma1,sigma2,taua,taub)[0]) / d / 2)
    dExcdnb_N=((lxc_xc(na, nb + d, sigma0, sigma1, sigma2,taua,taub)[0] - lxc_xc(na, nb - d, sigma0, sigma1,sigma2,taua,taub)[0]) / d / 2)
    dExcdsigma0_N=((lxc_xc(na, nb, sigma0 + d, sigma1, sigma2,taua,taub)[0] - lxc_xc(na, nb, sigma0 - d, sigma1,sigma2,taua,taub)[0]) / d / 2)
    dExcdsigma1_N=((lxc_xc(na, nb, sigma0, sigma1 + d, sigma2,taua,taub)[0] - lxc_xc(na, nb, sigma0, sigma1 - d,sigma2,taua,taub)[0]) / d / 2)
    dExcdsigma2_N=((lxc_xc(na, nb, sigma0, sigma1, sigma2 + d,taua,taub)[0] - lxc_xc(na, nb, sigma0, sigma1, sigma2 -d,taua,taub)[0]) / d / 2)
    dExcdtaua_N=((lxc_xc(na, nb, sigma0, sigma1, sigma2, taua +d ,taub)[0] - lxc_xc(na, nb, sigma0, sigma1, sigma2,taua -d ,taub)[0]) / d / 2)
    dExcdtaub_N=((lxc_xc(na, nb, sigma0, sigma1, sigma2, taua ,taub + d)[0] - lxc_xc(na, nb, sigma0, sigma1, sigma2,taua ,taub -d )[0]) / d / 2)


    dExdna_N=((lxc_xc(na + d, nb, sigma0, sigma1, sigma2,taua,taub)[1] - lxc_xc(na - d, nb, sigma0, sigma1,sigma2,taua,taub)[1]) / d / 2)
    dExdnb_N=((lxc_xc(na, nb + d, sigma0, sigma1, sigma2,taua,taub)[1] - lxc_xc(na, nb - d, sigma0, sigma1,sigma2,taua,taub)[1]) / d / 2)
    dExdsigma0_N=((lxc_xc(na, nb, sigma0 + d, sigma1, sigma2,taua,taub)[1] - lxc_xc(na, nb, sigma0 - d, sigma1,sigma2,taua,taub)[1]) / d / 2)
    dExdsigma1_N=((lxc_xc(na, nb, sigma0, sigma1 + d, sigma2,taua,taub)[1] - lxc_xc(na, nb, sigma0, sigma1 - d,sigma2,taua,taub)[1]) / d / 2)
    dExdsigma2_N=((lxc_xc(na, nb, sigma0, sigma1, sigma2 + d,taua,taub)[1] - lxc_xc(na, nb, sigma0, sigma1, sigma2 -d,taua,taub)[1]) / d / 2)
    dExdtaua_N=((lxc_xc(na, nb, sigma0, sigma1, sigma2, taua +d ,taub)[1] - lxc_xc(na, nb, sigma0, sigma1, sigma2,taua -d ,taub)[1]) / d / 2)
    dExdtaub_N=((lxc_xc(na, nb, sigma0, sigma1, sigma2, taua ,taub + d)[1] - lxc_xc(na, nb, sigma0, sigma1, sigma2,taua ,taub -d )[1]) / d / 2)

    dEcdna_N=((lxc_xc(na + d, nb, sigma0, sigma1, sigma2,taua,taub)[2] - lxc_xc(na - d, nb, sigma0, sigma1,sigma2,taua,taub)[2]) / d / 2)
    dEcdnb_N=((lxc_xc(na, nb + d, sigma0, sigma1, sigma2,taua,taub)[2] - lxc_xc(na, nb - d, sigma0, sigma1,sigma2,taua,taub)[2]) / d / 2)
    dEcdsigma0_N=((lxc_xc(na, nb, sigma0 + d, sigma1, sigma2,taua,taub)[2] - lxc_xc(na, nb, sigma0 - d, sigma1,sigma2,taua,taub)[2]) / d / 2)
    dEcdsigma1_N=((lxc_xc(na, nb, sigma0, sigma1 + d, sigma2,taua,taub)[2] - lxc_xc(na, nb, sigma0, sigma1 - d,sigma2,taua,taub)[2]) / d / 2)
    dEcdsigma2_N=((lxc_xc(na, nb, sigma0, sigma1, sigma2 + d,taua,taub)[2] - lxc_xc(na, nb, sigma0, sigma1, sigma2 -d,taua,taub)[2]) / d / 2)
    dEcdtaua_N=((lxc_xc(na, nb, sigma0, sigma1, sigma2, taua +d ,taub)[2] - lxc_xc(na, nb, sigma0, sigma1, sigma2,taua -d ,taub)[2]) / d / 2)
    dEcdtaub_N=((lxc_xc(na, nb, sigma0, sigma1, sigma2, taua ,taub + d)[2] - lxc_xc(na, nb, sigma0, sigma1, sigma2,taua ,taub -d )[2]) / d / 2)

    error1 = [0.0, 'exact']
    error2 = [0.0, 'exact']
    for E in [
        ('dExcdna', dExcdna, dExcdna_g[0], dExcdna_N),
        ('dExcdnb', dExcdnb, dExcdnb_g[0], dExcdnb_N),
        ('dExcdsigma0', dExcdsigma0, dExcdsigma0_g[0], dExcdsigma0_N),
        ('dExcdsigma1', dExcdsigma1, dExcdsigma1_g[0], dExcdsigma1_N),
        ('dExcdsigma2', dExcdsigma2, dExcdsigma2_g[0], dExcdsigma2_N),
        ('dExcdtaua', dExcdtaua, dExcdtaua_g[0], dExcdtaua_N),
        ('dExcdtaub', dExcdtaub, dExcdtaub_g[0], dExcdtaub_N),
        ('dExdna', dExdna, dExdna, dExdna_N), # N/A in calculate_spinpolarized
        ('dExdnb', dExdnb, dExdnb, dExdnb_N), # N/A in calculate_spinpolarized
        ('dExdsigma0', dExdsigma0, dExdsigma0, dExdsigma0_N), # N/A in calculate_spinpolarized
        ('dExdsigma1', dExdsigma1, dExdsigma1, dExdsigma1_N), # N/A in calculate_spinpolarized
        ('dExdsigma2', dExdsigma2, dExdsigma2, dExdsigma2_N), # N/A in calculate_spinpolarized
        ('dExdtaua', dExdtaua, dExdtaua, dExdtaua_N), # N/A in calculate_spinpolarized
        ('dExdtaub', dExdtaub, dExdtaub, dExdtaub_N), # N/A in calculate_spinpolarized
        ('dEcdna', dEcdna, dEcdna, dEcdna_N), # N/A in calculate_spinpolarized
        ('dEcdnb', dEcdnb, dEcdnb, dEcdnb_N), # N/A in calculate_spinpolarized
        ('dEcdsigma0', dEcdsigma0, dEcdsigma0, dEcdsigma0_N), # N/A in calculate_spinpolarized
        ('dEcdsigma1', dEcdsigma1, dEcdsigma1, dEcdsigma1_N), # N/A in calculate_spinpolarized
        ('dEcdsigma2', dEcdsigma2, dEcdsigma2, dEcdsigma2_N), # N/A in calculate_spinpolarized
        ('dEcdtaua', dEcdtaua, dEcdtaua, dEcdtaua_N), # N/A in calculate_spinpolarized
        ('dEcdtaub', dEcdtaub, dEcdtaub, dEcdtaub_N), # N/A in calculate_spinpolarized
        ]:
        for e in E[3:]:
            de1 = abs(e - E[1])
            de2 = abs(e - E[2])
            if de1 > error1[0]:
                error1[0] = de1
                error1[1] = E[0]
            if de2 > error2[0]:
                error2[0] = de2
                error2[1] = E[0]
    print name, error1[0], error1[1], error2[0], error2[1]
    assert error1[0] < 1.0e-9
    assert error2[0] < 1.0e-9
