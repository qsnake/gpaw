from math import pi, sqrt
from ASE.Utilities.DOS import DOS
from ASE.Units import units, Convert

import Numeric as num

def print_projectors(nucleus):
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


class LDOS(DOS):
    def __init__(self, calc, width=None, window=None, npts=201):
        """Electronic Local Density Of States object.
        
        'calc' is a gpaw calculator instance.
        'width' is the width of the "delta-functions". Defaults to the electronic temperature of the calculation.
        'window' is the energy window. Default is from the lowest to the highest eigenvalue.
        'npts' is the number of energy points.
        """
        DOS.__init__(self, calc, width, window, npts)
        self.P_auni = [nucleus.P_uni for nucleus in calc.nuclei]

        self.occ_a = []
        for nucleus in calc.nuclei:
            ni = j = 0
            while nucleus.setup.n_j[j] != -1:
                ni += 2 * nucleus.setup.l_j[j] + 1
                j += 1
            self.occ_a.append(ni)

    def Delta(self, energy):
        """Return a delta-function centered at 'energy'."""
        x = -((self.energies - energy) / self.width)**2
        x = num.clip(x, -100.0, 100.0)
        return num.exp(x) / (sqrt(pi) * self.width)

    def GetLDOS(self, a, spin=None):
        """Get the DOS projected onto the projector functions of atom a."""

        if spin is None:
            if self.nspins == 2:
                # Spin-polarized calculation, but no spin specified -
                # return the total DOS:
                return self.GetLDOS(a, spin=0) + self.GetLDOS(a, spin=1)
            else:
                spin = 0
        
        if hasattr(a, '__iter__'):
            # a is a list of atom indicies -
            # sum all bound-state angular chanels and add indicated atoms:
            dos_e = num.zeros(self.npts, num.Float)
            for atom in a:
                dos_ie = self.GetLDOS(atom, spin)
                for i in range(self.occ_a[atom]):
                    dos_e += dos_ie[i]
            return dos_e

        nk = len(self.w_k)
        P_kni = self.P_auni[a][spin * nk : (spin + 1) * nk]
        ni = len(P_kni[0, 0])
        dos_ie = num.zeros((ni, self.npts), num.Float)
        for w, P_ni, e_n in zip(self.w_k, P_kni, self.e_skn[spin]):
            for P_i, e in zip(P_ni, e_n):
                for i, P in enumerate(P_i):
                    dos_ie[i] += w * abs(P)**2 * self.Delta(e)
        return dos_ie
    
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
