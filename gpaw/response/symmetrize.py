import numpy as np
from ase.dft.kpoints import get_monkhorst_shape

def find_kq(bzkpt_kG, q):
    """Find the index of k+q for all kpoints in BZ."""

    nkpt = bzkpt_kG.shape[0]
    kq = np.zeros(nkpt, dtype=int)
    nkptxyz = get_monkhorst_shape(bzkpt_kG)
        
    dk = 1. / nkptxyz 
    kmax = (nkptxyz - 1) * dk / 2.
    N = np.zeros(3, dtype=int)

    for k in range(nkpt):
        kplusq = bzkpt_kG[k] + q
        for dim in range(3):
            if kplusq[dim] > 0.5: # 
                kplusq[dim] -= 1.
            elif kplusq[dim] < -0.5:
                kplusq[dim] += 1.

            N[dim] = int(np.round((kplusq[dim] + kmax[dim])/dk[dim]))

        kq[k] = N[2] + N[1] * nkptxyz[2] + N[0] * nkptxyz[2]* nkptxyz[1]

        tmp = bzkpt_kG[kq[k]]
        if (abs(kplusq - tmp)).sum() > 1e-8:
            print k, kplusq, tmp
            raise ValueError('k+q index not correct!')

    return kq


def find_ibzkpt(symrel, kpt_IBZkG, kptBZ):

    find = False
    ibzkpt = 0
    iop = 0
    timerev = False

    for ioptmp in range(len(symrel)):
        for i in range(kpt_IBZkG.shape[0]):
            tmp = np.inner(symrel[ioptmp], kpt_IBZkG[i])
            if (np.abs(tmp - kptBZ) < 1e-8).all():
                ibzkpt = i
                iop = ioptmp
                find = True
                break
        if find == True:
            break
    
    if find == False:
        for ioptmp in range(len(symrel)):
            for i in range(kpt_IBZkG.shape[0]):
                tmp = np.inner(symrel[ioptmp], kpt_IBZkG[i])
                if (np.abs(tmp + kptBZ) < 1e-8).all():
                    ibzkpt = i
                    iop = ioptmp
                    find = True
                    timerev = True
                    break
            if find == True:
                break
            
    if find == False:        
        print kptBZ
        print kpt_IBZkG
        raise ValueError('Cant find corresponding IBZ kpoint!')

    return ibzkpt, iop, timerev


def symmetrize_wavefunction(a_g, op_cc, kpt0, kpt1, timerev):

    if (np.abs(op_cc - np.eye(3,dtype=int)) < 1e-10).all():
        if timerev:
            return a_g.conj()
        else:
            return a_g
    elif (np.abs(op_cc + np.eye(3,dtype=int)) < 1e-10).all():
        return a_g.conj()
    else:
        import _gpaw
        b_g = np.zeros_like(a_g)
        if timerev:
            _gpaw.symmetrize_wavefunction(a_g, b_g, op_cc.T.copy(), kpt0, -kpt1)
            return b_g.conj()
        else:
            _gpaw.symmetrize_wavefunction(a_g, b_g, op_cc.T.copy(), kpt0, kpt1)
            return b_g
