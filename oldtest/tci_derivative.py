# This test calculates derivatives of lcao overlap matrices such as
#
#   a          ~a
#  P      =  < p  | Phi   >
#   i mu        i      mu
#
# and compares to finite difference results.

from ase.data.molecules import molecule
from ase.units import Bohr
from gpaw import GPAW
from gpaw.atom.basis import BasisMaker

obasis = BasisMaker('O').generate(2, 1)
hbasis = BasisMaker('H').generate(2, 1)
basis = {'O' : obasis, 'H' : hbasis}

system1 = molecule('H2O')
system1.center(vacuum=2.0)
system1.positions[1] -= 0.2

system2 = system1.copy()
system2.set_cell((3., 3., 3.))
system2.set_pbc(1)

def runcheck(system, dR, kpts=None):
    calc = GPAW(mode='lcao', basis=basis, txt=None)
    system.set_calculator(calc)

    calc.initialize(system)
    calc.set_positions(system)

    wfs = calc.wfs
    tci = wfs.tci

    tci.lcao_forces = True

    calc.initialize(system)
    calc.set_positions(system)

    a = 0
    c = 2
    na = len(system)
    rna = range(na)

    T1 = wfs.T_qMM.copy()
    P1_a = [wfs.P_aqMi[b].copy() for b in rna]
    S1 = wfs.S_qMM.copy()
    Theta1 = tci.Theta_qMM.copy()
    dTdR_tci = tci.dTdR_kcmm[0, c].copy()
    dPdR_tci_a = [tci.dPdR_akcmi[b][0, c].copy() for b in rna]
    dSdR_tci = tci.dSdR_kcmm[0, c].copy()
    dThetadR_tci = tci.dThetadR_kcmm[0, c].copy()

    system.positions[a,c] += dR

    calc.initialize(system)
    calc.set_positions(system)

    T2 = wfs.T_qMM.copy()
    P2_a = [wfs.P_aqMi[b].copy() for b in rna]
    S2 = wfs.S_qMM.copy()
    Theta2 = tci.Theta_qMM.copy()

    dTdR_fd = (T2 - T1) / dR * Bohr
    dPdR_fd_a = [(p2 - p1) / dR * Bohr for p2, p1 in zip(P2_a, P1_a)]
    dSdR_fd = (S2 - S1) / dR * Bohr
    dThetadR_fd = (Theta2 - Theta1) / dR * Bohr

    dPdRa_ami = wfs.get_projector_derivatives(tci, a, c, 0)
    dSdR_real = wfs.get_overlap_derivatives(tci, a, c, dPdRa_ami, 0)

    errs = [abs(dTdR_tci * tci.mask_amm[a] - dTdR_fd).max(),
            abs(dSdR_real - dSdR_fd).max(),
            max([abs(dPdRa_ami[b] - dPdR_fd_a[b]).max() for b in rna])]

    print 'err dTdR', errs[0]
    print 'err dSdR', errs[1]
    print 'err dPdR', errs[2]

    for err in errs:
        assert err < 2 * dR

for dR in [1e-5]: #[1e-3, 1e-5, 1e-7]:
    # Other values of dR should work fine, but we like short tests
    print 'dR =', dR
    print '---------------'
    print 'Gamma point'
    runcheck(system1, dR)
    print
    print 'Arbitrary k point'
    runcheck(system2, dR, kpts=[(0.2, 0.3, 0.1)])
    print
