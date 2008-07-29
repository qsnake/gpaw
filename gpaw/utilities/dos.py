from math import pi, sqrt
import numpy as npy
from ase.units import Hartree
from ase.parallel import paropen
from gpaw.utilities import pack, wignerseitz
from gpaw.setup_data import SetupData
from gpaw.gauss import Gauss

import gpaw.mpi as mpi

def print_projectors(nucleus):
    """Print information on the projectors of input nucleus object.

    If nucleus is a string, treat this as an element name.
    """
    if type(nucleus) is str:
        setup = SetupData(nucleus, 'LDA', 'paw')
        n_j = setup.n_j
        l_j = setup.l_j
    else:
        n_j = nucleus.setup.n_j
        l_j = nucleus.setup.l_j
    
    angular = [['1'],
               ['y', 'z', 'x'],
               ['xy', 'yz', '3z^2-r^2', 'xz', 'x^2-y^2'],
               ['3x^2y-y^3', 'xyz', '5yz^2-yr^2', '5z^3-3zr^2',
                '5xz^2-xr^2', 'x^2z-y^2z', 'x^3-3xy^2'],
               ]
    print ' i n l m'
    print '--------'
    i = 0
    for n, l in zip(n_j, l_j):
        for m in range(2*l+1):
            if n == -1:
                n = '*'
            print '%2s %s %s_%s' % (i, n, 'spdf'[l], angular[l][m])
            i += 1

def get_angular_projectors(nucleus, angular, type='bound'):
    """Determine the projector indices which have specified angula
    quantum number.

    angular can be s, p, d, f, or a list of these.
    If type is 'bound', only bound state projectors are considered, otherwise
    all projectors are included.
    """
    # Get the number of relevant j values
    if type == 'bound':
        nj = 0
        while nucleus.setup.n_j[nj] != -1: nj += 1
    else:
        nj = len(nucleus.setup.n_j)
            

    # Choose the relevant projectors
    projectors = []
    i = j = 0
    for j in range(nj):
        m = 2 * nucleus.setup.l_j[j] + 1
        if 'spdf'[nucleus.setup.l_j[j]] in angular:
            projectors.extend(range(i, i + m))
        j += 1
        i += m

    return projectors

def delta(x, x0, width):
    """Return a gaussian of given width centered at x0."""
    return npy.exp(npy.clip(-((x - x0) / width)**2,
                            -100.0, 100.0)) / (sqrt(pi) * width)

def fold(energies, weights, npts, width):
    """Take a list of energies and weights, and sum a delta function
    for each."""
    emin = min(energies) - 5 * width
    emax = max(energies) + 5 * width
    e = npy.linspace(emin, emax, npts)
    dos_e = npy.zeros(npts)
    for e0, w in zip(energies, weights):
        dos_e += w * delta(e, e0, width)
    return e, dos_e

def raw_orbital_LDOS(paw, a, spin, angular='spdf'):
    """Return a list of eigenvalues, and their weight on the specified atom.

    angular can be s, p, d, f, or a list of these.
    If angular is None, the raw weight for each projector is returned.

    An integer value for ``angular`` can also be used to specify a specific
    projector function.
    """
    w_k = paw.weight_k
    nk = len(w_k)
    nb = paw.nbands
    nucleus = paw.nuclei[a]

    energies = npy.empty(nb * nk)
    weights_xi = npy.empty((nb * nk, nucleus.setup.ni))
    x = 0
    for k, w in enumerate(w_k):
        energies[x:x + nb] = paw.collect_eigenvalues(k=k, s=spin)
        u = spin * nk + k
        weights_xi[x:x + nb, :] = w * npy.absolute(nucleus.P_uni[u])**2
        x += nb

    if angular is None:
        return energies, weights_xi
    elif type(angular) is int:
        return energies, weights_xi[angular]
    else:
        projectors = get_angular_projectors(nucleus, angular, type='bound')
        weights = npy.sum(npy.take(weights_xi,
                                   indices=projectors, axis=1), axis=1)
        return energies, weights

def molecular_LDOS(paw, mol, spin, lc=None, wf=None, P_aui=None):
    """Returns a list of eigenvalues, and their weights on a given molecule
    
       If wf is None, the weights are calculated as linear combinations of
       atomic orbitals using P_uni. lc should then be a list of weights
       for each atom. For example, the pure 2pi_x orbital of a
       molecule can be obtained with lc=[[0,0,0,1.0],[0,0,0,-1.0]]. mol
       should be a list of atom numbers contributing to the molecule.

       If wf is not none, it should be a list of wavefunctions
       corresponding to different kpoints and a specified band. It should
       be accompanied by a list of arrays: P_uai=nucleus[a].P_uni for the
       band n and a in mol. The weights are then calculated as the overlap
       of all-electron KS wavefunctions with wf"""

    w_k = paw.weight_k
    nk = len(w_k)
    nb = paw.nbands
    
    P_un = npy.zeros((nk, nb), npy.complex)

    if wf is None:
        P_auni = npy.array([paw.nuclei[a].P_uni for a in mol])
        if lc is None:
            lc = [[1,0,0,0] for a in mol]
        N = 0
        for atom, w_a in zip(range(len(mol)), lc):
            i=0
            for w_o in w_a:
                P_un += w_o * P_auni[atom,:,:,i]
                N += abs(w_o)**2
                i +=1
        P_un /= sqrt(N)

    else:
        if len(wf) == 1:  # Using the Gamma point only
            wf = [wf[0] for u in range(nk)]
            P_aui = [P_aui[0] for u in range(nk)]
        P_uai = npy.conjugate(P_aui)
        for kpt in paw.kpt_u:
            w = npy.reshape(npy.conjugate(wf)[kpt.u], -1)
            for n in range(nb):
                psit_nG = npy.reshape(kpt.psit_nG[n], -1)
                dV = paw.gd.h_c[0] * paw.gd.h_c[1] * paw.gd.h_c[2]
                P_un[kpt.u][n] = npy.dot(w, psit_nG) * dV * paw.a0**1.5
                for a, b in zip(mol, range(len(mol))):
                    atom = paw.nuclei[a]
                    p_i = atom.P_uni[kpt.u][n]
                    for i in range(len(p_i)):
                        for j in range(len(p_i)):
                            P_un[kpt.u][n] += (P_aui[b][kpt.u][i] *
                                               atom.setup.O_ii[i][j] * p_i[j])
                print n, abs(P_un)[kpt.u][n]**2
                
            print 'Kpoint', kpt.u, ' Sum: ',  sum(abs(P_un[kpt.u])**2)
            
    energies = npy.empty(nb * nk)
    weights = npy.empty(nb * nk)
    x = 0
    for k, w in enumerate(w_k):
        energies[x:x + nb] = paw.collect_eigenvalues(k=k, s=spin)
        u = spin * nk + k
        weights[x:x + nb] = w * npy.absolute(P_un[u])**2
        x += nb

    return energies, weights

                    
def raw_wignerseitz_LDOS(paw, a, spin):
    """Return a list of eigenvalues, and their weight on the specified atom"""
    gd = paw.gd
    atom_index = gd.empty(dtype=int)
    atom_ac = npy.array([n.spos_c * gd.N_c for n in paw.nuclei])
    wignerseitz(atom_index, atom_ac, gd.beg_c, gd.end_c)

    w_k = paw.weight_k
    nk = len(w_k)
    nb = paw.nbands
    nucleus = paw.nuclei[a]

    energies = npy.empty(nb * nk)
    weights = npy.empty(nb * nk)
    x = 0
    for k, w in enumerate(w_k):
        u = spin * nk + k
        energies[x:x + nb] = paw.collect_eigenvalues(k=k, s=spin)
        for n, psit_G in enumerate(paw.kpt_u[u].psit_nG):
            P_i = nucleus.P_uni[u, n]
            P_p = pack(npy.outer(P_i, P_i))
            Delta_p = sqrt(4 * pi) * nucleus.setup.Delta_pL[:, 0]
            weights[x + n] = w * (gd.integrate(npy.absolute(
                npy.where(atom_index == a, psit_G, 0.0))**2)
                                  + npy.dot(Delta_p, P_p))
        x += nb
    return energies, weights


class RawLDOS:
    """Class to get the unfolded LDOS"""
    def __init__(self, calc):
        self.paw = calc
        for nucleus in calc.nuclei:
            if not hasattr(nucleus.setup,'l_i'):
                # get the mapping
                l_i = []
                for l in nucleus.setup.l_j:
                    for i in range(2*l+1):
                        l_i.append(l)
                nucleus.setup.l_i = l_i

    def get(self, atom):
        """Return the s,p,d weights for each state"""
        spd = npy.zeros((self.paw.nspins * self.paw.nkpts, self.paw.nbands, 3))

        if hasattr(atom, '__iter__'):
            # atom is a list of atom indicies 
            for a in atom:
                spd += self.get(a)
            return spd

        k=0
        nucleus = self.paw.nuclei[atom]
        for k in range(self.paw.nkpts):
            for s in range(self.paw.nspins):
                myu = self.paw.get_myu(k, s)
                u = k * self.paw.nspins + s
                if myu is not None and nucleus.in_this_domain:
                    for n in range(self.paw.nbands):
                        for i, P in enumerate(nucleus.P_uni[myu, n]):
                            spd[u, n, nucleus.setup.l_i[i]] += abs(P)**2
                        
        self.paw.domain.comm.sum(spd)
        self.paw.kpt_comm.sum(spd)
        return spd

    def by_element(self):
        # get element indicees
        elemi = {}
        for i,a in enumerate(self.paw.atoms):
            symbol = a.symbol
            if elemi.has_key(symbol):
                elemi[symbol].append(i)
            else:
                elemi[symbol] = [i]
        for key in elemi.keys():
            elemi[key] = self.get(elemi[key])
        return elemi

    def by_element_to_file(self, 
                           filename='ldos_by_element.dat',
                           width=None):
        """Write the LDOS by element to a file"""
        ldbe = self.by_element()

        f = paropen(filename,'w')

        if width is None:
            # unfolded ldos
            eu = '[eV]'
            print >> f, '# e_i' + eu + '  spin  kpt     n   kptwght',
            for key in ldbe:
                if len(key) == 1: key=' '+key
                print  >> f, ' '+key+':s     p        d      ',
            print  >> f,' sum'
            for k in range(self.paw.nkpts):
                for s in range(self.paw.nspins):
                    u = k * self.paw.nspins + s
                    e_n = self.paw.collect_eigenvalues(k=k, s=s) * Hartree
                    myu = self.paw.get_myu(k, s)
                    if myu is None:
                        w = 0.
                    else:
                        w = self.paw.kpt_u[myu].weight
                    self.paw.kpt_comm.max(w)
                    for n in range(self.paw.nbands):
                        sum = 0.
                        print >> f, '%10.5f %2d %5d' % (e_n[n], s, k), 
                        print >> f, '%6d %8.4f' % (n, w),
                        for key in ldbe:
                            spd = ldbe[key][u,n]
                            for l in range(3):
                                sum += spd[l]
                                print >> f, '%8.4f' % spd[l],
                        print >> f, '%8.4f' % sum
        else:
            # folded ldos

            gauss = Gauss(width)

            # minimal and maximal energies
            emin = 1.e32
            emax = -1.e32
            for k in range(self.paw.nkpts):
                for s in range(self.paw.nspins):
                    e_n = self.paw.collect_eigenvalues(k=k, s=s).tolist()
                    e_n.append(emin)
                    emin = min(e_n)
                    e_n[-1] = emax
                    emax = max(e_n)
            emin *= Hartree
            emax *= Hartree
            emin -= 4*width
            emax += 4*width

            # set de to sample 4 points in the width
            de = width/4.
            
            for s in range(self.paw.nspins):
                print >> f, '# Gauss folded, width=%g [eV]' % width
                print >> f, '# e[eV]  spin ',
                for key in ldbe:
                    if len(key) == 1: key=' '+key
                    print  >> f, ' '+key+':s     p        d      ',
                print  >> f

                # loop over energies
                emax=emax+.5*de
                e=emin
                while e<emax:
                    val = {}
                    for key in ldbe:
                        val[key] = npy.zeros((3))
                    for k in range(self.paw.nkpts):
                        u = k * self.paw.nspins + s
                        myu = self.paw.get_myu(k, s)
                        if myu is None:
                            w = 0.
                        else:
                            w = self.paw.kpt_u[myu].weight
                        self.paw.kpt_comm.max(w)

                        e_n = self.paw.collect_eigenvalues(k=k, s=s) * Hartree
                        for n in range(self.paw.nbands):
                            w_i = w * gauss.get(e_n[n] - e)
                            for key in ldbe:
                                val[key] += w_i * ldbe[key][u, n]

                    print >> f, '%10.5f %2d' % (e, s), 
                    for key in val:
                        spd = val[key]
                        for l in range(3):
                            print >> f, '%8.4f' % spd[l],
                    print >> f
                    e += de
                            

        f.close()
