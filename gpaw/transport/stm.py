import numpy as np
from ase.units import Bohr
from gpaw.operators import Laplace
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lfc import NewLocalizedFunctionsCollection as LFC
import gpaw.mpi as mpi

class LocalizedFunctions:
    """ 
       A typical transverse plane of some grid...
       (pbc's only in transverse directions)

        --------------------------------------------------(3)
       |    Extended region                                |
       |    .........                         .........    |
       |    .    ---.-------------------------.--(2)  .    |
       |    .   |   .                         .   |   .    |
       |    o2..|....                         o3..|....    |    
       |        |                                 |        |
       |        |     Fixed region                |        |
       |        |                                 |        |
       |        |                                 |        |
       |        |                                 |        |
       |        |                                 |        |
       |        |                                 |        |
       |        |                                 |        |
       |    ....|..oo                         ....|....    |
       |    .   |   .                         .   |   .    |
       |    .  (1)--.-------------------------.---    .    |
       |    o........                         o1.......    |
       |                                                   |
      (0)--------------------------------------------------
        
       Extended region = region which is used to extend the potential in order to
                         get rid of pbc's
       
       o1, o2, o3 = corners of LocalizedFunctions objects which are periodic
                    translations of LF object with corner at o.
        
       Some vectors:
       (1)-(0) = (3)-(2) = pbc_cutoff (if pbc_cutoff = 0 <=> (0)=(1) /\ (2)=(3))
        o  - (1) = v1_c
        oo - (2) = v2_c   
        
       more doc to come.......
    """

    def __init__(self, gd, f_iG, corner_c, index=None, vt_G=None):
        self.gd = gd
        assert not gd.is_non_orthogonal()
        self.size_c = np.array(f_iG.shape[1:4])
        self.f_iG = f_iG
        self.corner_c = corner_c
        self.index = index
        self.vt_G = vt_G
   
    def periodic(self, extension = 0):
        """extension - Extension of the surface unit cell in terms of gridpoints"""
        
        self.extension = extension
        
        extension_c = np.array([extension,extension,0],dtype=int)

        v1_c = np.sign(self.corner_c[:2] - extension_c[:2])
        v2_c = np.sign(self.corner_c[:2] + self.size_c[:2]-\
                     (self.gd.end_c[:2] - extension_c[:2] - 1))
        
        # Translation vectors along the axes of the transverse unit-cell.
        trans_c = []
        for i in np.where(v1_c == -1)[0]:
            v = np.zeros(3,dtype=int)    
            v[i] = 1
            trans_c.append(v)
            
        for i in np.where(v2_c == 1)[0]:
            v = np.zeros(3,dtype=int)
            v[i] = -1
            trans_c.append(v)
        
        # Translation vectors along the diagonal of the transverse unit-cell.
        trans_diag_c = []
        for i in range(len(trans_c)):
            for j in range(i,len(trans_c)):
                v = trans_c[i]+trans_c[j]
                if not len(np.where(v == 0)[0]) >= 2:
                    trans_diag_c.append(v)
        
        trans_c = trans_c+trans_diag_c
        trans_c.append(np.zeros(3)) # The original LF object
        
        trans_c[:] *= (self.gd.N_c-1) 
        self.periodic_list = trans_c+self.corner_c
 
        list = []
        for corner in self.periodic_list:
            list.append(LocalizedFunctions(self.gd,self.f_iG,
                                        corner_c=corner,
                                        index=self.index,
                                        vt_G=self.vt_G))
        return list


    def __len__(self):
        return len(self.f_iG)

    def apply_t(self):
        """Apply kinetic energy operator and return new object."""
        p = 2  # padding
        newsize_c = self.size_c + 2 * p
        gd = GridDescriptor(N_c=newsize_c + 1,
                            cell_cv=self.gd.h_c * (newsize_c + 1),
                            pbc_c=False,
                            comm=mpi.serial_comm)
        T = Laplace(gd, scale =1/2., n=p)
        f_ig = np.zeros((len(self.f_iG),) + tuple(newsize_c))
        f_ig[:, p:-p, p:-p, p:-p] = self.f_iG
        Tf_iG = np.empty_like(f_ig)
        T.apply(f_ig, Tf_iG)
        return LocalizedFunctions(self.gd, Tf_iG, self.corner_c - p,
                                  self.index)
        
    def overlap(self, other):
        start_c = np.maximum(self.corner_c, other.corner_c)
        stop_c = np.minimum(self.corner_c + self.size_c,
                            other.corner_c + other.size_c)
        if (start_c < stop_c).all():
            astart_c = start_c - self.corner_c
            astop_c = stop_c - self.corner_c
            a_iG = self.f_iG[:,
                astart_c[0]:astop_c[0],
                astart_c[1]:astop_c[1],
                astart_c[2]:astop_c[2]].reshape((len(self.f_iG), -1))
            bstart_c = start_c - other.corner_c
            bstop_c = stop_c - other.corner_c
            b_iG = other.f_iG[:,
                bstart_c[0]:bstop_c[0],
                bstart_c[1]:bstop_c[1],
                bstart_c[2]:bstop_c[2]].reshape((len(other.f_iG), -1))
            if self.vt_G is not None:
                a_iG *= self.vt_G[start_c[0]:stop_c[0],
                                  start_c[1]:stop_c[1],
                                  start_c[2]:stop_c[2]].reshape((-1,))
            return self.gd.dv * np.inner(a_iG, b_iG)
        else:
            return None
    
    def restrict(self):
        """Restricts the box of the objet to the current grid"""
        #XXX We probably do not need this function.
        start_c = np.maximum(self.corner_c, np.zeros(3))
        stop_c = np.minimum(self.corner_c + self.size_c, self.gd.N_c)
        
        if (start_c < stop_c).all():
            astart_c = start_c - self.corner_c
            astop_c = stop_c -self.corner_c
            a_iG = self.f_iG[:,
                astart_c[0]:astop_c[0],
                astart_c[1]:astop_c[1],
                astart_c[2]:astop_c[2]]
            new_corner_c = self.corner_c
            for i in np.where(self.corner_c<0):
                new_corner_c[i] = 0
            return LocalizedFunctions(self.gd, a_iG,
                                      new_corner_c, self.index,
                                      self.vt_G, self.extension_c) 
        else:
            return None

    def __or__(self, other):
        if isinstance(other, LocalizedFunctions):
            return self.overlap(other)

        # other is a potential:
        vt_G = other
        return LocalizedFunctions(self.gd, self.f_iG, self.corner_c,
                                  self.index, vt_G)

class WannierFunction(LocalizedFunctions):
    def __init__(self, gd, wanf_G, corner_c, index=None):
        LocalizedFunctions.__init__(self, gd, wanf_G[np.newaxis, :, :, :],
                                    corner_c, index)

class AtomCenteredFunctions(LocalizedFunctions):
    def __init__(self, gd, spline_j, spos_c, index=None):
        rcut = max([spline.get_cutoff() for spline in spline_j])
        corner_c = np.ceil(spos_c * gd.N_c - rcut / gd.h_c).astype(int)
        size_c = np.ceil(spos_c * gd.N_c + rcut / gd.h_c).astype(int) - corner_c
        smallgd = GridDescriptor(N_c=size_c + 1,
                                 cell_cv=gd.h_c * (size_c + 1),
                                 pbc_c=False,
                                 comm=mpi.serial_comm)
        lfc = LFC(smallgd, [spline_j])
        lfc.set_positions((spos_c[np.newaxis, :] * gd.N_c - corner_c + 1) /
                          smallgd.N_c)
        ni = lfc.Mmax
        f_iG = smallgd.zeros(ni)
        lfc.add(f_iG, {0: np.eye(ni)})
        LocalizedFunctions.__init__(self, gd, f_iG, corner_c, 
                                     index=index)

class STM:
    def __init__(self, tip, surface):
        self.tip = tip
        self.srf = surface

        tgd = tip.gd
        sgd = surface.gd

        #assert tgd.h_cv == sgd.h_cv
        assert not (tgd.h_c - sgd.h_c).any()

    def initialize(self, tip_atom_index, dmin=2.0):
        #XXX We might want to look at the order of stuff that happens in this function.
        self.dmin = dmin/Bohr
        self.tip_atom_index = tip_atom_index     
        # Extension of the surface unit cell
        tgd = self.tip.gd
        sgd = self.srf.gd
        tipcell_c = tgd.cell_c    
        srfcell_c = sgd.cell_c
        
        extension = np.ceil(0.5 * max([tipcell_c[0]/sgd.h_c[0],\
                                  tipcell_c[1]/sgd.h_c[1]])).astype(int)
        self.extension = extension
        extension_c = np.array([extension,extension,0])
        self.extension_c = extension_c
        print extension
        dmin /= Bohr # XXX is done twice
        tip_pos_av = self.tip.atoms.get_positions() / Bohr
        srf_pos_av = self.srf.atoms.get_positions() / Bohr
        tip_zmin = tip_pos_av[tip_atom_index, 2]
        srf_zmax = srf_pos_av[:, 2].max()

        #offset_c = (tip_pos_av[tip_atom_index] / self.tip.gd.h_c).astype(int)
        tip_zmin_a = np.empty(len(tip_pos_av))
        
        for a, setup in enumerate(self.tip.wfs.setups):
            rcutmax = max([phit.get_cutoff() for phit in setup.phit_j])
            tip_zmin_a[a] = tip_pos_av[a, 2] - rcutmax - tip_zmin
    
        srf_zmax_a = np.empty(len(srf_pos_av))
        for a, setup in enumerate(self.srf.wfs.setups):
            rcutmax = max([phit.get_cutoff() for phit in setup.phit_j])
            srf_zmax_a[a] = srf_pos_av[a, 2] + rcutmax - srf_zmax
        
        tip_indices = np.where(tip_zmin_a < srf_zmax_a.max() - dmin)[0]  
        srf_indices = np.where(srf_zmax_a > tip_zmin_a.min() + dmin)[0]  
        print 'Tip atoms:', tip_indices
        print 'Surface atoms:', srf_indices

        self.tip_functions = []
        i=0
        print i
        for a in tip_indices:
            setup = self.tip.wfs.setups[a]
            spos_c = tip_pos_av[a] / self.tip.gd.cell_c
            if a == self.tip_atom_index:
                print 'tip_box0', i
                self.tip_atom_box_0 = i
            for phit in setup.phit_j:
                f = AtomCenteredFunctions(self.tip.gd, [phit], spos_c, i)
                self.tip_functions.append(f)
                i += len(f.f_iG)
        self.ni = i
        print extension
        # Apply kinetic energy:
        self.tip_functions_kin = []
        for f in self.tip_functions:
            self.tip_functions_kin.append(f.apply_t())

        self.srf_functions = []
        j = 0
        for a in srf_indices:
            setup = self.srf.wfs.setups[a]
            spos_c = srf_pos_av[a] / self.srf.gd.cell_c
            for phit in setup.phit_j:
                print 'srf', j
                f = AtomCenteredFunctions(self.srf.gd, [phit], spos_c, j)
                self.srf_functions += f.periodic(extension)

                j += len(f.f_iG)
        self.nj = j
        
        # Apply kinetic energy:
        self.srf_functions_kin = []
        for f in self.srf_functions:
            self.srf_functions_kin.append(f.apply_t())

        # Extend the surface grid
        svt_G = self.srf.get_effective_potential()
        
        newsize_c = 2 * self.extension_c + sgd.N_c
        
        extvt_G = np.zeros(newsize_c)
        extvt_G[extension:-extension,extension:-extension,:] = svt_G       
        extvt_G[:extension,extension:-extension]  = svt_G[-extension:]
        extvt_G[-extension:,extension:-extension] = svt_G[extension:]
        extvt_G[:,:extension] = extvt_G[:,-2*extension:-extension]        
        extvt_G[:,-extension:] = extvt_G[:,extension:2*extension]        
        
        # XXX N_c = newsize_c + 1 ?
        extsgd = GridDescriptor(N_c=newsize_c,
                                  cell_cv=sgd.h_c*(newsize_c),
                                  pbc_c=False,
                                  comm=mpi.serial_comm)   
        
        # Set positions of the surface orbitals in the extended grid
        for f, f_kin in zip(self.srf_functions, self.srf_functions_kin):
            f.corner_c += extension_c
            f_kin.corner_c += extension_c
    
        self.extsgd = extsgd
        self.extvt_G = extvt_G

        #XXX Get isolated tip and surface hamiltonians and
        # and initialize all the transport stuff
    
    def position_tip_cell(self,grdpt_c, return_v = False):   
        """Positions origin of the tip unit cell at grdpt_c
            in the extended surface unit cell."""         
        
        tvt_G = self.tip.get_effective_potential() 
        tip_pos_av = self.tip.atoms.get_positions() / Bohr
        srf_pos_av = self.srf.atoms.get_positions() / Bohr
        tip_zmin = tip_pos_av[self.tip_atom_index, 2]
        srf_zmax = srf_pos_av[:, 2].max()

        #corner of the tip unit cell in the extended grid        
        cell_corner_c = grdpt_c + self.extension_c      
        cell_corner_c[2] = np.ceil(\
           (srf_zmax + self.dmin - tip_zmin) / self.extsgd.h_c[2]).astype(int) 
        
        for f, f_kin in zip(self.tip_functions,self.tip_functions_kin):
            f.corner_c += cell_corner_c
            f_kin.corner_c += cell_corner_c
        
        # Add the tip potential at the respective place in the extended cell
        # of the surface
        size_c = self.tip.gd.N_c
        current_Vt = self.extvt_G.copy()
        current_Vt[cell_corner_c[0]:cell_corner_c[0] + size_c[0],
                   cell_corner_c[1]:cell_corner_c[1] + size_c[1],
                   cell_corner_c[2]:cell_corner_c[2] + size_c[2]] += tvt_G
        
        self.current=current_Vt
        self.tip_cell_corner_c = cell_corner_c
        if return_v:
            return_v        

    def position_tip(self, tip_pos):
        """Positions the tip atom at the position tip_pos in the
           original surface unit cell"""

    def get_V(self, tip_pos):
        """Returns the overlap hamiltonian for a position of the tip_atom """
        #set corners with resepct to large grid
        #add pots
        #actually calculate overlap

    def linescan(self,stargdp, endgdp):
        print 'liiiiiiinescan'
        # make list of gridpoints
        #for i in gridpints
        #    v = self.get_V(i)
        #    current = stm_calc(v)
    def get_s(self):
        S_ij = np.zeros((self.ni, self.nj))
        for s in self.srf_functions:
            j1 = s.index
            j2 = j1 + len(s)
            for t in self.tip_functions:
                i1 = t.index
                i2 = i1 + len(t)
                overlap = (t | vt_G | s) # + kinetic energy XXX
                if overlap is not None:
                    S_ij[i1:i2, j1:j2] += overlap
  
        return S_ij
