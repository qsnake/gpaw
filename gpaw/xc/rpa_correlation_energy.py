import numpy as np
from time import ctime
from gpaw import GPAW
from gpaw.response.df import DF
from gpaw.utilities import devnull
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.mpi import rank
from ase.parallel import paropen
import sys

class RPACorrelation:

    def __init__(self, calc, txt=None):
        
        self.calc = calc
       
        if txt is None:
            if rank == 0:
                self.txt = sys.stdout
            else:
                sys.stdout = devnull
                self.txt = devnull
        else:
            assert type(txt) is str
            from ase.parallel import paropen
            self.txt = paropen(txt, 'w')

        self.nspins = calc.wfs.nspins
        self.bz_k_points = calc.wfs.bzk_kc
        self.atoms = calc.get_atoms()
        self.setups = calc.wfs.setups
        self.ibz_q_points, self.q_weights = self.get_ibz_q_points(self.bz_k_points) 
        self.print_initialization()
        self.initialized = 0
        

    def get_rpa_correlation_energy(self,
                                   kcommsize=1,
                                   ecut=10,
                                   nbands=None,
                                   restart=None,
                                   w=np.linspace(0, 200., 32)):

        self.initialize_calculation(w, ecut, nbands, kcommsize)
        
        E_q = []
        if restart is not None:
            assert type(restart) is str
            try:
                f = paropen(restart, 'r')
                lines = f.readlines()
                for line in lines:
                    E_q.append(eval(line))
                f.close()
                print >> self.txt, 'Correlation energy from %s Q-points obtained from restart file: ' % len(E_q), restart
                print >> self.txt
            except:
                IOError

        for index, q in enumerate(self.ibz_q_points[len(E_q):]):
            E_q.append(self.get_E_q(index=index, q=q, nbands=self.nbands,
                                    kcommsize=kcommsize, ecut=ecut, w=w))
            if restart is not None:
                f = paropen(restart, 'a')
                print >> f, E_q[-1]
                f.close()

        E = np.dot(np.array(self.q_weights), np.array(E_q).real)
        print >> self.txt, 'RPA correlation energy:'
        print >> self.txt, 'E_c = %s eV' % E
        print >> self.txt
        print >> self.txt, 'Calculation completed at:  ', ctime()
        print >> self.txt
        print >> self.txt, '------------------------------------------------------'
        print >> self.txt
        return E


    def get_E_q(self,
                index=None,
                q=[0., 0., 0.],
                integrated=True,
                kcommsize=1,
                ecut=10,
                nbands=None,
                w=np.linspace(0, 200., 32)):
        
        if index is None:
            self.initialize_calculation(w, ecut, nbands, kcommsize)

        if abs(q[0]) < 0.001 and abs(q[1]) < 0.001 and abs(q[2]) < 0.001:
            q = [1.e-5, 0., 0.]
            optical_limit = True
        else:
            optical_limit = False
            
        df = DF(calc=self.calc,
                nbands=self.nbands,
                eta=0.0,
                q=q,
                w=w * 1j,
                ecut=ecut,
                kcommsize=kcommsize,
                optical_limit=optical_limit,
                hilbert_trans=False)
        df.txt = devnull

        if index is None:
            print >> self.txt, 'Calculating RPA dielectric matrix at:'
        else:
            print >> self.txt, '#', index, '- Calculating RPA dielectric matrix at:'
        
        if optical_limit:
            print >> self.txt, 'Q = [0 0 0]'
        else:
            print >> self.txt, 'Q = %s' % q 
            
        e_wGG = df.get_RPA_dielectric_matrix()

        Nw_local = len(e_wGG)
        local_int = np.zeros(Nw_local, dtype=complex)

        integrand = np.empty(len(w), complex)
        for i in range(Nw_local):
            local_int[i] = (np.log(np.linalg.det(e_wGG[i]))
                            + len(e_wGG[0]) - np.trace(e_wGG[i]))
            #local_int[i] = (np.sum(np.log(np.linalg.eigvals(e_wGG[i])))
            #                + self.npw - np.trace(e_wGG[i]))
        df.wcomm.all_gather(local_int, integrand)
        del df
        del e_wGG
        dw = w[1] - w[0]
        E_q = dw * np.sum((integrand[:-1]+integrand[1:])/2.) / (2.*np.pi)
        print >> self.txt, 'E_c(Q) = %s eV' % E_q.real
        print >> self.txt
        if index is None:
            print >> self.txt, 'Calculation completed at:  ', ctime()
            print >> self.txt
            print >> self.txt, '------------------------------------------------------'
        if integrated:
            return E_q
        else:
            return integrand
       

    def get_ibz_q_points(self, bz_k_points):

        # Get all q-points
        all_qs = []
        for k1 in bz_k_points:
            for k2 in bz_k_points:
                all_qs.append(k1-k2)
        all_qs = np.array(all_qs)

        # Fold q-points into Brillouin zone
        all_qs[np.where(all_qs > 0.501)] -= 1.
        all_qs[np.where(all_qs < -0.499)] += 1.

        # Make list of non-identical q-points in full BZ
        bz_qs = [all_qs[0]]
        for q_a in all_qs:
            q_in_list = False
            for q_b in bz_qs:
                if (abs(q_a[0]-q_b[0]) < 0.01 and
                    abs(q_a[1]-q_b[1]) < 0.01 and
                    abs(q_a[2]-q_b[2]) < 0.01):
                    q_in_list = True
                    break
            if q_in_list == False:
                bz_qs.append(q_a)
        self.bz_q_points = bz_qs
                
        # Obtain q-points and weights in the irreducible part of the BZ
        kpt_descriptor = KPointDescriptor(bz_qs, self.nspins)
        kpt_descriptor.set_symmetry(self.atoms, self.setups, usesymm=True)
        ibz_q_points = kpt_descriptor.ibzk_kc
        q_weights = kpt_descriptor.weight_k
        return ibz_q_points, q_weights


    def print_initialization(self):
        
        print >> self.txt, '------------------------------------------------------'
        print >> self.txt, 'Calculating non-self consistent RPA correlation energy'
        print >> self.txt, '------------------------------------------------------'
        print >> self.txt, 'Started at:  ', ctime()
        print >> self.txt
        print >> self.txt, 'Atoms                          :   %s' % self.atoms.get_name()
        print >> self.txt, 'Ground state XC functional     :   %s' % self.calc.hamiltonian.xc.name
        print >> self.txt, 'Valence electrons              :   %s' % self.setups.nvalence
        print >> self.txt, 'Number of Bands                :   %s' % self.calc.wfs.nbands
        print >> self.txt, 'Number of Converged Bands      :   %s' % self.calc.input_parameters['convergence']['bands']
        print >> self.txt, 'Number of Spins                :   %s' % self.nspins
        print >> self.txt, 'Number of K-points             :   %s' % len(self.calc.wfs.bzk_kc)
        print >> self.txt, 'Number of Q-points             :   %s' % len(self.bz_q_points)
        print >> self.txt, 'Number of Irreducible K-points :   %s' % len(self.calc.wfs.ibzk_kc)
        print >> self.txt, 'Number of Irreducible Q-points :   %s' % len(self.ibz_q_points)
        print >> self.txt
        for q, weight in zip(self.ibz_q_points, self.q_weights):
            print >> self.txt, 'Q: [%1.3f %1.3f %1.3f] - weight: %1.3f' % (q[0],q[1],q[2],
                                                                           weight)
        print >> self.txt
        print >> self.txt, '------------------------------------------------------'
        print >> self.txt, '------------------------------------------------------'
        print >> self.txt
        

    def initialize_calculation(self, w, ecut, nbands, kcommsize):
        
        dummy = DF(calc=self.calc,
                   eta=0.0,
                   w=w * 1j,
                   q=[0.,0.,0.],
                   ecut=ecut,
                   hilbert_trans=False,
                   kcommsize=kcommsize)
        dummy.txt = devnull
        dummy.spin = 0
        dummy.initialize()

        if nbands is None:
            nbands = dummy.npw
        self.nbands = nbands
        
        print >> self.txt, 'Planewave cut off          : %s eV' % ecut
        print >> self.txt, 'Number of Planewaves       : %s' % dummy.npw
        print >> self.txt, 'Response function bands    : %s' % nbands
        print >> self.txt, 'Frequency range            : %s - %s eV' % (w[0], w[-1])
        print >> self.txt, 'Number of frequency points : %s' % len(w)
        print >> self.txt
        print >> self.txt, 'Parallelization scheme'
        print >> self.txt, '     Total cpus         : %d' % dummy.comm.size
        if dummy.nkpt == 1:
            print >> self.txt, '     Band parsize       : %d' % dummy.kcomm.size
        else:
            print >> self.txt, '     Kpoint parsize     : %d' % dummy.kcomm.size
        print >> self.txt, '     Frequency parsize  : %d' % dummy.wScomm.size
        print >> self.txt, 'Memory usage estimate'
        print >> self.txt, '     chi0_wGG(Q)        : %f M / cpu' % (dummy.Nw_local *
                                                                     dummy.npw**2 * 16.
                                                                    / 1024**2)
        print >> self.txt
        del dummy
