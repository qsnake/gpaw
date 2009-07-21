
import numpy as np

from ase.units import Bohr
from gpaw import debug
from gpaw.mpi import world
from gpaw.utilities.dscftools import mpi_debug
from gpaw.overlap import Overlap
from gpaw.utilities import unpack
from gpaw.lfc import NewLocalizedFunctionsCollection as NewLFC

#from arbitrary_tci import projector_overlap_matrix2, generate_atomic_overlaps2

# -------------------------------------------------------------------

class PairOverlap:
    def __init__(self, gd, setups):
        self.gd = gd
        self.setups = setups
        self.ni_a = np.cumsum([0]+[setup.ni for setup in self.setups])

    def __len__(self):
        return self.ni_a[-1]

    def assign_atomic_pair_matrix(self, X_aa, a1, a2, dX_ii):
        X_aa[self.ni_a[a1]:self.ni_a[a1+1],\
             self.ni_a[a2]:self.ni_a[a2+1]] = dX_ii

    def extract_atomic_pair_matrix(self, X_aa, a1, a2):
        return X_aa[self.ni_a[a1]:self.ni_a[a1+1],\
                    self.ni_a[a2]:self.ni_a[a2+1]]

    def calculate_overlaps(self, spos_ac, lfc1, lfc2=None):
        raise RuntimeError('This is a virtual member function.')

    def calculate_atomic_pair_overlaps(self, lfs1, lfs2): #XXX Move some code here from above...
        raise RuntimeError('This is a virtual member function.')

class GridPairOverlap(PairOverlap):


    def calculate_overlaps(self, spos_ac, lfc1, lfc2=None):
        # CONDITION: The two sets of splines must belong to the same kpoint!

        if lfc2 is None:
            lfc2 = lfc1

        if isinstance(lfc1, NewLFC) and isinstance(lfc2, NewLFC):
            return self.calculate_overlaps2(spos_ac, lfc1, lfc2)

        assert not isinstance(lfc1, NewLFC) and not isinstance(lfc2, NewLFC)

        nproj = len(self)
        X_aa = np.zeros((nproj,nproj), dtype=float) # XXX always float?

        if debug:
            if world.rank == 0:
                print 'DEBUG INFO'

            mpi_debug('lfc1.lfs_a.keys(): %s' % lfc1.lfs_a.keys())
            mpi_debug('lfc2.lfs_a.keys(): %s' % lfc2.lfs_a.keys())
            mpi_debug('N_c=%s, beg_c=%s, end_c=%s' % (self.gd.N_c,self.gd.beg_c,self.gd.end_c))

        if debug:
            assert len(lfc1.spline_aj) == len(lfc1.spos_ac) # not distributed
            assert len(lfc2.spline_aj) == len(lfc2.spos_ac) # not distributed
            #assert lfc1.lfs_a.keys() == lfc2.lfs_a.keys() # XXX must they be equal?!?

        # Both loops are over all atoms in all domains
        for a1, spline1_j in enumerate(lfc1.spline_aj):
            # We assume that all functions have the same cut-off:
            rcut1 = spline1_j[0].get_cutoff()
            if debug: mpi_debug('a1=%d, spos1_c=%s, rcut1=%g, ni1=%d' % (a1,spos_ac[a1],rcut1,self.setups[a1].ni))

            for a2, spline2_j in enumerate(lfc2.spline_aj):
                # We assume that all functions have the same cut-off:
                rcut2 = spline2_j[0].get_cutoff()
                if debug: mpi_debug('  a2=%d, spos2_c=%s, rcut2=%g, ni2=%d' % (a2,spos_ac[a2],rcut2,self.setups[a2].ni))

                X_ii = self.extract_atomic_pair_matrix(X_aa, a1, a2)

                b1 = 0
                for beg1_c, end1_c, sdisp1_c in self.gd.get_boxes(spos_ac[a1], rcut1, cut=False): # loop over lfs1.box_b instead?
                    if debug: mpi_debug('    b1=%d, beg1_c=%s, end1_c=%s, sdisp1_c=%s' % (b1,beg1_c,end1_c,sdisp1_c), ordered=False)

                    # Atom a1 has at least one piece so the LFC has LocFuncs
                    lfs1 = lfc1.lfs_a[a1]

                    # Similarly, the LocFuncs must have the piece at hand
                    box1 = lfs1.box_b[b1]
                    bra_iB1 = box1.get_functions()

                    if debug:
                        assert lfs1.dtype == lfc1.dtype
                        assert bra_iB1.shape[0] == lfs1.ni
                        assert self.setups[a1].ni == lfs1.ni, 'setups[%d].ni=%d, lfc1.lfs_a[%d].ni=%d' % (a1,self.setups[a1].ni,a1,lfs1.i)

                    b2 = 0
                    for beg2_c, end2_c, sdisp2_c in self.gd.get_boxes(spos_ac[a2], rcut2, cut=False): # loop over lfs2.box_b instead?
                        if debug: mpi_debug('      b2=%d, beg2_c=%s, end2_c=%s, sdisp2_c=%s' % (b2,beg2_c,end2_c,sdisp2_c), ordered=False)

                        # Atom a2 has at least one piece so the LFC has LocFuncs
                        lfs2 = lfc2.lfs_a[a2]

                        # Similarly, the LocFuncs must have the piece at hand
                        box2 = lfs2.box_b[b2]
                        ket_iB2 = box2.get_functions()

                        if debug:
                            assert lfs2.dtype == lfc2.dtype
                            assert ket_iB2.shape[0] == lfs2.ni
                            assert self.setups[a2].ni == lfs2.ni, 'setups[%d].ni=%d, lfc2.lfs_a[%d].ni=%d' % (a2,self.setups[a2].ni,a2,lfs2.ni)

                        # Find the intersection of the two boxes
                        beg_c = np.array([map(max, zip(beg1_c, beg2_c))]).ravel()
                        end_c = np.array([map(min, zip(end1_c, end2_c))]).ravel()

                        if debug: mpi_debug('        beg_c=%s, end_c=%s, size_c=%s' % (beg_c, end_c, tuple(end_c-beg_c)), ordered=False)

                        # Intersection is non-empty, add overlap contribution
                        if (beg_c <= end_c).all():
                            w1slice = [slice(None)]+[slice(b,e) for b,e in \
                                zip(beg_c-beg1_c, end_c-beg1_c)]
                            w2slice = [slice(None)]+[slice(b,e) for b,e in \
                                zip(beg_c-beg2_c, end_c-beg2_c)]

                            X_ii += self.gd.dv * np.inner( \
                                bra_iB1[w1slice].reshape((lfs1.ni,-1)), \
                                ket_iB2[w2slice].reshape((lfs2.ni,-1))) #XXX phase factors for kpoints

                        del ket_iB2
                        b2 += 1

                    del bra_iB1
                    b1 += 1

        self.gd.comm.sum(X_aa) # better to sum over X_ii?
        return X_aa


    def calculate_overlaps2(self, spos_ac, lfc1, lfc2=None):
        # CONDITION: The two sets of splines must belong to the same kpoint!

        if lfc2 is None:
            lfc2 = lfc1

        assert isinstance(lfc1, NewLFC) and isinstance(lfc2, NewLFC)

        nproj = len(self)
        X_aa = np.zeros((nproj,nproj), dtype=float) # XXX always float?

        if debug:
            if world.rank == 0:
                print 'DEBUG INFO'

            mpi_debug('len(lfc1.sphere_a): %d, lfc1.my_atom_indices: %s' % (len(lfc1.sphere_a),lfc1.my_atom_indices))
            mpi_debug('len(lfc2.sphere_a): %d, lfc2.my_atom_indices: %s' % (len(lfc2.sphere_a),lfc2.my_atom_indices))
            mpi_debug('N_c=%s, beg_c=%s, end_c=%s' % (self.gd.N_c,self.gd.beg_c,self.gd.end_c))

        if debug:
            assert len(lfc1.sphere_a) == len(lfc2.sphere_a) # XXX must they be equal?!?

        # Both loops are over all atoms in all domains
        for a1 in lfc1.my_atom_indices:
            sphere1 = lfc1.sphere_a[a1]

            # We assume that all functions have the same cut-off:
            spline1_j = sphere1.spline_j
            rcut1 = spline1_j[0].get_cutoff()
            if debug: mpi_debug('a1=%d, spos1_c=%s, rcut1=%g, ni1=%d' % (a1,spos_ac[a1],rcut1,self.setups[a1].ni), ordered=False)

            for a2 in lfc2.my_atom_indices:
                sphere2 = lfc2.sphere_a[a2]

                # We assume that all functions have the same cut-off:
                spline2_j = sphere2.spline_j
                rcut2 = spline2_j[0].get_cutoff()
                if debug: mpi_debug('  a2=%d, spos2_c=%s, rcut2=%g, ni2=%d' % (a2,spos_ac[a2],rcut2,self.setups[a2].ni), ordered=False)

                X_ii = self.extract_atomic_pair_matrix(X_aa, a1, a2)

                b1 = 0
                for beg1_c, end1_c, sdisp1_c in self.gd.get_boxes(spos_ac[a1], rcut1, cut=False): # loop over lfs1.box_b instead?
                    if debug: mpi_debug('    b1=%d, beg1_c=%s, end1_c=%s, sdisp1_c=%s' % (b1,beg1_c,end1_c,sdisp1_c), ordered=False)

                    b2 = 0
                    for beg2_c, end2_c, sdisp2_c in self.gd.get_boxes(spos_ac[a2], rcut2, cut=False): # loop over lfs2.box_b instead?
                        if debug: mpi_debug('      b2=%d, beg2_c=%s, end2_c=%s, sdisp2_c=%s' % (b2,beg2_c,end2_c,sdisp2_c), ordered=False)

                        # Find the intersection of the two boxes
                        beg_c = np.array([map(max, zip(beg1_c, beg2_c))]).ravel()
                        end_c = np.array([map(min, zip(end1_c, end2_c))]).ravel()

                        if debug: mpi_debug('        beg_c=%s, end_c=%s, size_c=%s' % (beg_c, end_c, tuple(end_c-beg_c)), ordered=False)

                        # Intersection is non-empty, add overlap contribution
                        if (beg_c < end_c).all():
                            #shape = tuple(end_c-beg_c)

                            # A_gm are function values, G_b boundary grid points

                            #"""
                            # THIS IS FAST BUT ONLY WORKS EXACTLY FOR PERIODIC!!!
                            i1 = 0
                            for j1, spline1 in enumerate(spline1_j):
                                #A1_gm, G1_b = sphere1.spline_to_grid(spline1, \
                                #    self.gd, beg_c, end_c, spos_ac[a1]-sdisp1_c)
                                #if debug: mpi_debug('         j1=%d, ng1max=%d, A1_gm: %s, G1_b: %s' % (j1,np.prod(np.array(end1_c)-np.array(beg1_c)),A1_gm.shape,G1_b.shape), ordered=False)
                                #nm1 = 2*spline1.get_angular_momentum_number()+1
                                #assert A1_gm.shape == (np.sum(G1_b[1::2]-G1_b[::2]), nm1)

                                #n = 0
                                #for a,b in zip(G1_b[::2], G1_b[1::2]):
                                #    for m in range(A1_gm.shape[-1]): #XXX ugly
                                #        np.put(bra1_mG[m], range(a,b), A1_gm[n:n+b-a,m])
                                #    n += b-a
                                bra1_mB = spline1.get_functions(self.gd, \
                                    beg_c, end_c, spos_ac[a1]-sdisp1_c)
                                nm1 = bra1_mB.shape[0]
                                if debug: mpi_debug('         j1=%d, nm1=%d, ng1max=%d, bra1_mB: %s' % (j1,nm1,np.prod(np.array(end_c)-np.array(beg_c)),bra1_mB.shape), ordered=False)

                                i2 = 0
                                for j2, spline2 in enumerate(spline2_j):
                                    #A2_gm, G2_b = sphere2.spline_to_grid(spline2, \
                                    #    self.gd, beg_c, end_c, spos_ac[a2]-sdisp2_c)
                                    #    #if debug: mpi_debug('    -> j=%d, ng2max=%d, A2_gm: %s, G2_b: %s, \n%s' % (j,np.prod(np.array(end2_c)-np.array(beg2_c)),A2_gm.shape,G2_b.shape,G2_b), ordered=False)
                                    #nm2 = 2*spline2.get_angular_momentum_number()+1
                                    #ket2_mG = self.gd.zeros(nm2, dtype=float) #XXX always float?

                                    #n = 0
                                    #for a,b in zip(G2_b[::2], G2_b[1::2]):
                                    #    for m in range(A2_gm.shape[-1]): #XXX ugly
                                    #        np.put(ket2_mG_mG[m], range(a,b), A2_gm[n:n+b-a,m])
                                    #    n += b-a
                                    ket2_mB = spline2.get_functions(self.gd, \
                                        beg_c, end_c, spos_ac[a2]-sdisp2_c)
                                    nm2 = ket2_mB.shape[0]
                                    if debug: mpi_debug('           j2=%d, nm2=%d, ng2max=%d, ket2_mB: %s' % (j2,nm2,np.prod(np.array(end_c)-np.array(beg_c)),ket2_mB.shape), ordered=False)

                                    X_mm = X_ii[i1:i1+nm1,i2:i2+nm2]
                                    X_mm += self.gd.dv * np.inner(bra1_mB.reshape((nm1,-1)), ket2_mB.reshape((nm2,-1))) #XXX phase factors for kpoints

                                    i2 += nm2
                                i1 += nm1
                            """
                            import _gpaw
                            def spline_to_box(spline, gd, start_c, end_c, spos_c):
                                dom = gd
                                h_cv = dom.cell_cv / gd.N_c[:, np.newaxis]
                                #offset_c = start_c - gd.beg_c # translate gd.beg_c to start_c
                                #corner_c = np.array([0,0,0]) # start_c is the new array origin
                                #pos_v = np.dot(spos_c, gd.cell_cv) - np.dot(offset_c, h_cv)
                                #A_gm, G_b = _gpaw.spline_to_grid(spline.spline, corner_c, end_c-start_c,
                                #                                 pos_v, h_cv, end_c-start_c, gd.beg_c)
                                h_cv = dom.cell_cv / gd.N_c[:, np.newaxis]
                                offset_c = start_c - gd.beg_c # translate gd.beg_c to start_c
                                pos_v = np.dot(spos_c, dom.cell_cv) - np.dot(offset_c, h_cv)
                                A_gm, G_b = _gpaw.spline_to_grid(spline.spline, start_c-offset_c, end_c-offset_c, pos_v, h_cv,
                                                                 gd.n_c, gd.beg_c*0)

                                #pos_v = np.dot(spos_c, gd.cell_cv) - np.dot(offset_c, h_cv)
                                nm, ng = A_gm.shape[-1], np.sum(G_b[1::2]-G_b[::2])
                                work_mB = np.zeros((nm,)+tuple(end_c-start_c), dtype=A_gm.dtype)

                                n = 0
                                for a,b in zip(G_b[::2], G_b[1::2]):
                                    for m,work_B in enumerate(work_mB):
                                        np.put(work_B, range(a,b), A_gm[n:n+(b-a),m])
                                    n += b-a

                                return work_mB

                            # THIS IS IN DEVELOPMENT!!!
                            work1_G = self.gd.zeros(dtype=float) #XXX always float?
                            work2_G = self.gd.zeros(dtype=float) #XXX always float?

                            nproj1 = sum([2*spline.get_angular_momentum_number()+1 for spline in spline1_j])
                            nproj2 = sum([2*spline.get_angular_momentum_number()+1 for spline in spline2_j])
                            assert X_ii.shape == (nproj1, nproj2)

                            i1 = 0
                            for j1, spline1 in enumerate(spline1_j):
                                bra1_mB = spline_to_box(spline1, self.gd, beg_c, end_c, spos_ac[a1]-sdisp1_c)
                                nm1 = 2*spline1.get_angular_momentum_number()+1

                                for m1,work1_G in enumerate(bra1_mB):
                                    i2 = 0
                                    for j2, spline2 in enumerate(spline2_j):
                                        ket2_mB = spline_to_box(spline2, self.gd, beg_c, end_c, spos_ac[a2]-sdisp2_c)
                                        nm2 = 2*spline2.get_angular_momentum_number()+1

                                        for m2,work2_G in enumerate(ket2_mB):
                                            if debug: mpi_debug('               i1=%d, i2=%d' % (i1,i2))

                                            X_ii[i1,i2] += self.gd.dv * np.vdot(work1_G, work2_G)

                                            i2 += 1

                                    i1 += 1
                            "#"#"
                            # THIS WORKS BUT IS VERY SLOW!!!
                            work1_G = self.gd.zeros(dtype=float) #XXX always float?
                            work2_G = self.gd.zeros(dtype=float) #XXX always float?

                            nproj1 = sum([2*spline.get_angular_momentum_number()+1 for spline in spline1_j])
                            nproj2 = sum([2*spline.get_angular_momentum_number()+1 for spline in spline2_j])
                            assert X_ii.shape == (nproj1, nproj2)

                            i1 = 0
                            for j1, spline1 in enumerate(spline1_j):
                                A1_gm, G1_b = sphere1.spline_to_grid(spline1, \
                                    self.gd, beg_c, end_c, spos_ac[a1]-sdisp1_c)
                                if debug: mpi_debug('         j1=%d, ng1max=%d, A1_gm: %s, G1_b: %s' % (j1,np.prod(np.array(end1_c)-np.array(beg1_c)),A1_gm.shape,G1_b.shape), ordered=False)
                                nm1 = 2*spline1.get_angular_momentum_number()+1
                                assert A1_gm.shape == (np.sum(G1_b[1::2]-G1_b[::2]), nm1)

                                for m1 in range(nm1):
                                    work1_G.fill(0.0)
                                    n1 = 0
                                    for a,b in zip(G1_b[::2], G1_b[1::2]):
                                        np.put(work1_G, range(a,b), A1_gm[n1:n1+b-a,m1])
                                        n1 += b-a

                                    i2 = 0
                                    for j2, spline2 in enumerate(spline2_j):
                                        A2_gm, G2_b = sphere2.spline_to_grid(spline2, \
                                            self.gd, beg_c, end_c, spos_ac[a2]-sdisp2_c)
                                        if debug: mpi_debug('           j2=%d, ng2max=%d, A2_gm: %s, G2_b: %s' % (j2,np.prod(np.array(end2_c)-np.array(beg2_c)),A2_gm.shape,G2_b.shape), ordered=False)
                                        nm2 = 2*spline2.get_angular_momentum_number()+1
                                        assert A2_gm.shape == (np.sum(G2_b[1::2]-G2_b[::2]), nm2)

                                        for m2 in range(nm2):
                                            work2_G.fill(0.0)
                                            n2 = 0
                                            for a,b in zip(G2_b[::2], G2_b[1::2]):
                                                np.put(work2_G, range(a,b), A2_gm[n2:n2+b-a,m2])
                                                n2 += b-a

                                            if debug: mpi_debug('               i1=%d, i2=%d' % (i1,i2))

                                            X_ii[i1,i2] += self.gd.dv * np.vdot(work1_G, work2_G)

                                            i2 += 1

                                    i1 += 1
                            """

                        #del ket_iB2
                        b2 += 1

                    #del bra_iB1
                    b1 += 1

        self.gd.comm.sum(X_aa) # better to sum over X_ii?
        return X_aa


"""
def extract_projectors(gd, pt, a):
    print 'pt.lfs_a: %s' % pt.lfs_a.keys()
    lfs = pt.lfs_a[a]
    assert len(lfs.box_b) == 1
    box = lfs.box_b[0]

    pt_iB = box.get_functions()
    assert pt_iB.shape[0] == lfs.ni
    shape = pt_iB.shape[1:]

    work_iG = gd.zeros(lfs.ni, dtype=pt.dtype)
    print 'a=%d, ni=%d, pt_iB.shape' % (a,lfs.ni), pt_iB.shape

    # We assume that all functions have the same cut-off:
    rcut = pt.spline_aj[a][0].get_cutoff()
    gdbox_b = gd.get_boxes(pt.spos_ac[a], rcut, cut=False)
    assert len(gdbox_b) == 1, 'What?'
    beg_c, end_c, sdisp_c = gdbox_b[0]
    work_iG[:,beg_c[0]:end_c[0],beg_c[1]:end_c[1],beg_c[2]:end_c[2]] = pt_iB

    #for i,spline in enumerate(pt.spline_aj[a]): #XXX wrong, loop over pt_iB!!!
    #    rcut = spline.get_cutoff()
    #    gdbox_b = gd.get_boxes(pt.spos_ac[a], rcut, cut=False)
    #    assert len(gdbox_b) == 1, 'What?'
    #    beg_c, end_c, sdisp_c = gdbox_b[0]
    #    print 'a=%d, pt_iB[%d].shape' % (a,i), pt_iB[i].shape
    #    print 'a=%d, work_iG[%d].shape' % (a,i), work_iG[i,beg_c[0]:end_c[0],beg_c[1]:end_c[1],beg_c[2]:end_c[2]].shape
    #    work_iG[i,beg_c[0]:end_c[0],beg_c[1]:end_c[1],beg_c[2]:end_c[2]] = pt_iB[i]

    return work_iG

def overlap_projectors(gd, pt, setups):
    work_aiG = {}
    for a in range(len(setups)):
        work_aiG[a] = extract_projectors(gd,pt,a)

    ni_a = np.cumsum([0]+[setup.ni for setup in setups])
    nproj = ni_a[-1]
    dB_aa = np.zeros((nproj,nproj), dtype=float)
    for a1, work1_iG in work_aiG.items():
        for a2, work2_iG in work_aiG.items():
            B_ii = np.zeros((setups[a1].ni,setups[a2].ni), dtype=float)
            for i1, work1_G in enumerate(work1_iG):
                for i2, work2_G in enumerate(work2_iG):
                    B_ii[i1,i2] = np.dot(work1_G.flat, work2_G.flat)*gd.dv
            dB_aa[ni_a[a1]:ni_a[a1+1], ni_a[a2]:ni_a[a2+1]] = B_ii
    return dB_aa
"""

class ProjectorPairOverlap(Overlap, GridPairOverlap):
    """
    TODO
    """
    def __init__(self, wfs, atoms):
        """TODO

        Attributes:

        ============  ======================================================
        ``dB_aa``     < p_i^a | p_i'^a' >
        ``xO_aa``     TODO
        ``dC_aa``     TODO
        ``xC_aa``     TODO
        ============  ======================================================
        """

        Overlap.__init__(self, wfs)
        GridPairOverlap.__init__(self, wfs.gd, wfs.setups)
        self.natoms = len(atoms)
        if debug:
            assert len(self.setups) == self.natoms
        self.update(wfs, atoms)

    def update(self, wfs, atoms):
        self.timer.start('Update two-center overlap')

        nproj = len(self)

        """
        self.dB_aa = np.zeros((nproj, nproj), dtype=float) #always float?
        for a1,setup1 in enumerate(self.setups):
            for a2 in wfs.pt.my_atom_indices:
                setup2 = self.setups[a2]
                R = (atoms[a1].get_position() - atoms[a2].get_position()) / Bohr

                if a1 == a2:
                    dB_ii = setup1.B_ii
                else:
                    dB_ii = projector_overlap_matrix2(setup1, setup2, R)
                #if a1 < a2:
                #    dB_ii = projector_overlap_matrix2(setup1, setup2, R)
                #elif a1 == a2:
                #    dB_ii = setup1.B_ii
                #else:
                #    dB_ii = self.dB_aa[ni_a[a2]:ni_a[a2+1], ni_a[a1]:ni_a[a1+1]].T

                #self.dB_aa[self.ni_a[a1]:self.ni_a[a1+1], \
                #           self.ni_a[a2]:self.ni_a[a2+1]] = dB_ii
                self.assign_atomic_pair_matrix(self.dB_aa, a1, a2, dB_ii)
        self.domain_comm.sum(self.dB_aa) #TODO too heavy?
        """
        #self.dB_aa = overlap_projectors(wfs.gd, wfs.pt, wfs.setups)

        #spos_ac = wfs.pt.spos_ac # not in NewLFC
        spos_ac = atoms.get_scaled_positions() % 1.0
        self.dB_aa = self.calculate_overlaps(spos_ac, wfs.pt)

        # Create two-center (block-diagonal) coefficients for overlap operator
        dO_aa = np.zeros((nproj, nproj), dtype=float) #always float?
        for a,setup in enumerate(self.setups):
            self.assign_atomic_pair_matrix(dO_aa, a, a, setup.O_ii)

        # Calculate two-center rotation matrix for overlap projections
        self.xO_aa = self.get_rotated_coefficients(dO_aa)

        # Calculate two-center coefficients for inverse overlap operator
        lhs_aa = np.eye(nproj) + self.xO_aa
        rhs_aa = -dO_aa
        self.dC_aa = np.linalg.solve(lhs_aa.T, rhs_aa.T).T #TODO parallel

        # Calculate two-center rotation matrix for inverse overlap projections
        self.xC_aa = self.get_rotated_coefficients(self.dC_aa)

        self.timer.stop('Update two-center overlap')

    def get_rotated_coefficients(self, X_aa):
        """Rotate two-center projector expansion coefficients with
        the projector-projector overlap integrals as basis.

        Performs the following operation and returns the result::

                    ---
            a1,a3   \       a1    a2     a2,a3
           Y      =  )  <  p   | p   >  X
            i1,i3   /       i1    i2     i2,i3
                    ---
                   a2,i2
        """
        return np.dot(self.dB_aa, X_aa)

    def apply_to_atomic_matrices(self, dI_asp, P_axi, wfs, kpt, shape=()):

        self.timer.start('Update two-center projections')

        nproj = len(self)
        dI_aa = np.zeros((nproj, nproj), dtype=float) #always float?

        for a, dI_sp in dI_asp.items():
            dI_p = dI_sp[kpt.s]
            dI_ii = unpack(dI_p)
            self.assign_atomic_pair_matrix(dI_aa, a, a, dI_ii)
        self.domain_comm.sum(dI_aa) #TODO too heavy?

        dM_aa = self.get_rotated_coefficients(dI_aa)
        Q_axi = wfs.pt.dict(shape, zero=True)
        for a1 in range(self.natoms):
            if a1 in Q_axi.keys():
                Q_xi = Q_axi[a1]
            else:
                # Atom a1 is not in domain so allocate a temporary buffer
                Q_xi = np.zeros(shape+(self.setups[a1].ni,), dtype=wfs.pt.dtype) #TODO
            for a2, P_xi in P_axi.items():
                dM_ii = self.extract_atomic_pair_matrix(dM_aa, a1, a2)
                Q_xi += np.dot(P_xi, dM_ii.T) #sum over a2 and last i in dM_ii
            self.domain_comm.sum(Q_xi)

        self.timer.stop('Update two-center projections')

        return Q_axi

    def apply(self, a_xG, b_xG, wfs, kpt, calculate_P_ani=True,
              extrapolate_P_ani=False):
        """Apply the overlap operator to a set of vectors.

        Parameters
        ==========
        a_nG: ndarray
            Set of vectors to which the overlap operator is applied.
        b_nG: ndarray, output
            Resulting S times a_nG vectors.
        kpt: KPoint object
            k-point object defined in kpoint.py.
        calculate_P_ani: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | a_nG> are calculated.
            When False, existing P_ani are used
        extrapolate_P_ani: bool
            When True, the integrals of projector times vectors#XXX TODO
            P_ni = <p_i | a_nG> are calculated.
            When False, existing P_ani are used

        """

        self.timer.start('Apply overlap')
        b_xG[:] = a_xG
        shape = a_xG.shape[:-3]
        P_axi = wfs.pt.dict(shape)

        if calculate_P_ani:
            wfs.pt.integrate(a_xG, P_axi, kpt.q)
        else:
            for a,P_ni in kpt.P_ani.items():
                P_axi[a][:] = P_ni

        Q_axi = wfs.pt.dict(shape)
        for a, Q_xi in Q_axi.items():
            Q_xi[:] = np.dot(P_axi[a], self.setups[a].O_ii)

        wfs.pt.add(b_xG, Q_axi, kpt.q)
        self.timer.stop('Apply overlap')

        if extrapolate_P_ani:
            for a1 in range(self.natoms):
                if a1 in Q_axi.keys():
                    Q_xi = Q_axi[a1]
                    Q_xi[:] = P_axi[a1]
                else:
                    # Atom a1 is not in domain so allocate a temporary buffer
                    Q_xi = np.zeros(shape+(self.setups[a1].ni,), dtype=wfs.pt.dtype) #TODO
                for a2, P_xi in P_axi.items():
                    # xO_aa are the overlap extrapolators across atomic pairs
                    xO_ii = self.extract_atomic_pair_matrix(self.xO_aa, a1, a2)
                    Q_xi += np.dot(P_xi, xO_ii.T) #sum over a2 and last i in xO_ii
                self.domain_comm.sum(Q_xi)

            return Q_axi
        else:
            return P_axi

    def apply_inverse(self, a_xG, b_xG, wfs, kpt, calculate_P_ani=True,
                      extrapolate_P_ani=False):

        self.timer.start('Apply inverse overlap')
        b_xG[:] = a_xG
        shape = a_xG.shape[:-3]
        P_axi = wfs.pt.dict(shape)

        if calculate_P_ani:
            wfs.pt.integrate(a_xG, P_axi, kpt.q)
        else:
            for a,P_ni in kpt.P_ani.items():
                P_axi[a][:] = P_ni

        Q_axi = wfs.pt.dict(shape, zero=True)
        for a1 in range(self.natoms):
            if a1 in Q_axi.keys():
                Q_xi = Q_axi[a1]
            else:
                # Atom a1 is not in domain so allocate a temporary buffer
                Q_xi = np.zeros(shape+(self.setups[a1].ni,), dtype=wfs.pt.dtype) #TODO
            for a2, P_xi in P_axi.items():
                # dC_aa are the inverse coefficients across atomic pairs
                dC_ii = self.extract_atomic_pair_matrix(self.dC_aa, a1, a2)
                Q_xi += np.dot(P_xi, dC_ii.T) #sum over a2 and last i in dC_ii
            self.domain_comm.sum(Q_xi)

        wfs.pt.add(b_xG, Q_axi, kpt.q)
        self.timer.stop('Apply inverse overlap')

        if extrapolate_P_ani:
            for a1 in range(self.natoms):
                if a1 in Q_axi.keys():
                    Q_xi = Q_axi[a1]
                    Q_xi[:] = P_axi[a1]
                else:
                    # Atom a1 is not in domain so allocate a temporary buffer
                    Q_xi = np.zeros(shape+(self.setups[a1].ni,), dtype=wfs.pt.dtype) #TODO
                for a2, P_xi in P_axi.items():
                    # xC_aa are the inverse extrapolators across atomic pairs
                    xC_ii = self.extract_atomic_pair_matrix(self.xC_aa, a1, a2)
                    Q_xi += np.dot(P_xi, xC_ii.T) #sum over a2 and last i in xC_ii
                self.domain_comm.sum(Q_xi)

            return Q_axi
        else:
            return P_axi





