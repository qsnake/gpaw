from gpaw.xc_functional import XCFunctional
from math import pi

nspins = 2
for name in ['LDA', 'PBE', 'revPBE', 'RPBE',
             'LDAx', 'revPBEx', 'RPBEx',
             'None-C_PW','TPSS','M06L']:
    libxc = XCFunctional(name, nspins)
    lxc_xc = libxc.calculate_xcenergy
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

    error = [0.0, 'exact']
    for E in [
        ('dExcdna', dExcdna, dExcdna_N),
        ('dExcdnb', dExcdnb, dExcdnb_N),
        ('dExcdsigma0', dExcdsigma0, dExcdsigma0_N),
        ('dExcdsigma1', dExcdsigma1, dExcdsigma1_N),
        ('dExcdsigma2', dExcdsigma2, dExcdsigma2_N),
        ('dExcdtaua', dExcdtaua, dExcdtaua_N),
        ('dExcdtaub', dExcdtaub, dExcdtaub_N),
        ('dExdna', dExdna, dExdna_N),
        ('dExdnb', dExdnb, dExdnb_N),
        ('dExdsigma0', dExdsigma0, dExdsigma0_N),
        ('dExdsigma1', dExdsigma1, dExdsigma1_N),
        ('dExdsigma2', dExdsigma2, dExdsigma2_N),
        ('dExdtaua', dExdtaua, dExdtaua_N),
        ('dExdtaub', dExdtaub, dExdtaub_N),
        ('dEcdna', dEcdna, dEcdna_N),
        ('dEcdnb', dEcdnb, dEcdnb_N),
        ('dEcdsigma0', dEcdsigma0, dEcdsigma0_N),
        ('dEcdsigma1', dEcdsigma1, dEcdsigma1_N),
        ('dEcdsigma2', dEcdsigma2, dEcdsigma2_N),
        ('dEcdtaua', dEcdtaua, dEcdtaua_N),
        ('dEcdtaub', dEcdtaub, dEcdtaub_N),
        ]:
        for e in E[2:]:
            de = abs(e - E[1])
            if de > error[0]:
                error[0] = de
                error[1] = E[0]
    print name, error[0], error[1]
    assert error[0] < 1.0e-9
