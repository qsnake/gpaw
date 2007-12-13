from math import pi, sqrt
import Numeric as num
from gpaw.utilities import pack, wignerseitz
from ASE.Units import units

def print_projectors(nucleus):
    """Print information on the projectors of input nucleus object"""
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
    If type is 'bound', only bound state projectors are considered, otherwize
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
    return num.exp(num.clip(-((x - x0) / width)**2,
                            -100.0, 100.0)) / (sqrt(pi) * width)

def fold_ldos(energies, weights, npts, width):
    """Take a list of energies and weights, and sum a delta function
    for each."""
    emin = min(energies) - 5 * width
    emax = max(energies) + 5 * width
    step = (emax - emin) / (npts - 1)
    e = num.arange(emin, emax + 1e-7, step, typecode=num.Float)
    ldos_e = num.zeros(npts, typecode=num.Float)
    for e0, w in zip(energies, weights):
        ldos_e += w * delta(e, e0, width)
    return e, ldos_e

def raw_orbital_LDOS(calc, a, spin, angular='spdf'):
    """Return a list of eigenvalues, and their weight on the specified atom.

    angular can be s, p, d, f, or a list of these.
    If angular is None, the raw weight for each projector is returned"""
    w_k = calc.GetIBZKPointWeights()
    nk = len(w_k)
    nb = calc.GetNumberOfBands()
    nucleus = calc.nuclei[a]

    energies = num.empty(nb * nk, num.Float)
    weights_xi = num.empty((nb * nk, nucleus.setup.ni), num.Float)
    x = 0
    for k, w in enumerate(w_k):
        energies[x:x + nb] = calc.GetEigenvalues(kpt=k, spin=spin)
        u = spin * nk + k
        weights_xi[x:x + nb, :] = w * num.absolute(nucleus.P_uni[u])**2
        x += nb

    if angular is None:
        return energies, weights_xi
    else:
        projectors = get_angular_projectors(nucleus, angular)
        weights = num.sum(num.take(weights_xi,
                                   indices=projectors, axis=1), axis=1)
        return energies, weights

def raw_wignerseitz_LDOS(calc, a, spin):
    """Return a list of eigenvalues, and their weight on the specified atom"""
    atom_index = calc.gd.empty(typecode=num.Int)
    atom_c = num.array([n.spos_c * calc.gd.N_c for n in calc.nuclei])
    wignerseitz(atom_index, atom_c, calc.gd.beg_c, calc.gd.end_c)

    w_k = calc.GetIBZKPointWeights()
    nk = len(w_k)
    nb = calc.GetNumberOfBands()
    nucleus = calc.nuclei[a]

    energies = num.empty(nb * nk, num.Float)
    weights = num.empty(nb * nk, num.Float)
    x = 0
    for k, w in enumerate(w_k):
        u = spin * nk + k
        energies[x:x + nb] = calc.GetEigenvalues(kpt=k, spin=spin)
        for n, psit_G in enumerate(calc.kpt_u[u].psit_nG):
            P_i = nucleus.P_uni[u, n]
            P_p = pack(num.outerproduct(P_i, P_i))
            Delta_p = sqrt(4 * pi) * nucleus.setup.Delta_pL[:, 0]
            weights[x + n] = w * (calc.gd.integrate(num.absolute(
                num.where(atom_index == a, psit_G, 0.0))**2)
                                  + num.dot(Delta_p, P_p))
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

    def get(self,atom):
        """Return the s,p,d weights for each state"""
        spd = num.zeros((self.paw.nspins,self.paw.nbands,3),num.Float)

        if hasattr(atom, '__iter__'):
            # atom is a list of atom indicies 
            for a in atom:
                spd += self.get(a)
            return spd
        
        nucleus = self.paw.nuclei[atom]
        for s in range(self.paw.nspins):
            for n in range(self.paw.nbands):
                for i,P in enumerate(nucleus.P_uni[s,n]):
                     spd[s,n,nucleus.setup.l_i[i]] += abs(P)**2
        return spd

    def by_element(self):
        # get element indicees
        elemi = {}
        for i,a in enumerate(self.paw.atoms):
            symbol = a.GetChemicalSymbol()
            if elemi.has_key(symbol):
                elemi[symbol].append(i)
            else:
                elemi[symbol] = [i]
        for key in elemi.keys():
            elemi[key] = self.get(elemi[key])
        return elemi

    def by_element_to_file(self,filename='ldos_by_element.dat'):
        """Write the LDOS by element to a file"""
        ldbe = self.by_element()
        f = open(filename,'w')
        eu = '['+units.GetEnergyUnit()+']'
        print >> f, '# e_i'+eu+'  spin   n ',
        for key in ldbe:
            if len(key) == 1: key=' '+key
            print  >> f, ' '+key+':s     p        d      ',
        print  >> f,' sum'
        for s in range(self.paw.nspins):
            e_n = self.paw.GetEigenvalues(spin=s)
            for n in range(self.paw.nbands):
                sum = 0.
                print >> f, '%10.5f' % e_n[n], s, '%6d' % n,
                for key in ldbe:
                    spd = ldbe[key][s,n]
                    for l in range(3):
                        sum += spd[l]
                        print >> f, '%8.4f' % spd[l],
                print >> f, '%8.4f' % sum
        f.close()
