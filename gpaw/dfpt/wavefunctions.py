"""This module implements a wave-function class."""

from gpaw.lfc import LocalizedFunctionsCollection as LFC

from gpaw.dfpt.kpointcontainer import KPointContainer


class WaveFunctions:
    """Class for wave-function related stuff (e.g. projectors and symmetry)."""
    
    def __init__(self, nbands, kpt_u, setups, gamma, kd, gd, symmetry=None):
        """Store and initialize required attributes.

        Parameters
        ----------
        nbands: int
            Number of occupied bands.
        kpt_u: list of KPoints
            List of KPoint instances from a ground-state calculation (i.e. the
            attribute ``calc.wfs.kpt_u``).
        setups: ...
            LFC setups.
        gamma: bool
            Gamma-point calculation if True.
        kd: KPointDescriptor
            Contains scaled coordinates of the k-points in the BZ and
            irreducible BZ.
        gd: GridDescriptor
            Descriptor for the coarse grid.            
        symmetry: ...
            Symmetry object ...

        """

        # K-point related attributes
        self.gamma = gamma
        self.kd = kd
        # Number of occupied bands
        self.nbands = nbands
        # Projectors
        self.pt = LFC(gd, [setup.pt_j for setup in setups])

        self.kpt_u = []
        
        for kpt in kpt_u:
            # Strip off KPoint attributes and store in your own KPointContainer
            # Note, only the occupied GS wave-functions are retained here !!
            kpt_ = KPointContainer(weight=kpt.weight,
                                   k=kpt.k,
                                   q=kpt.q,
                                   s=kpt.s,
                                   phase_cd=kpt.phase_cd,
                                   f_n=kpt.f_n,
                                   eps_n=kpt.eps_n,
                                   psit_nG=kpt.psit_nG[:nbands],
                                   psit1_nG=None,
                                   P_ani=None,
                                   dP_aniv=None)
            
            self.kpt_u.append(kpt_)

    def initialize(self, spos_ac):
        """Initialize projectors according to ``gamma`` attribute."""

        # Set positions on LFC's
        self.pt.set_positions(spos_ac)

        if not self.gamma:
            # Set k-vectors and update
            self.pt.set_k_points(self.kd.ibzk_qc)
            self.pt._update(spos_ac)

        # Calculate projector coefficients for the GS wave-functions
        self.calculate_projector_coef()
        
    def calculate_projector_coef(self):
        """Coefficients for the derivative of the non-local part of the PP.

        Parameters
        ----------
        k: int
            Index of the k-point of the Bloch state on which the non-local
            potential operates on.

        The calculated coefficients are the following (except for an overall
        sign of -1; see ``derivative`` member function of class ``LFC``):

        1. Coefficients from the projector functions::

                        /      a          
               P_ani =  | dG  p (G) Psi (G)  ,
                        /      i       n
                          
        2. Coefficients from the derivative of the projector functions::

                          /      a           
               dP_aniv =  | dG dp  (G) Psi (G)  ,
                          /      iv       n   

        where::
                       
                 a        d       a
               dp  (G) =  ---  Phi (G) .
                 iv         a     i
                          dR

        """

        n = self.nbands

        for kpt in self.kpt_u:

            # K-point index and wave-functions
            k = kpt.k
            psit_nG = kpt.psit_nG
            
            # Integration dicts
            P_ani   = self.pt.dict(shape=n)
            dP_aniv = self.pt.dict(shape=n, derivative=True)
    
            # 1) Integrate with projectors
            self.pt.integrate(psit_nG, P_ani, q=k)
            kpt.P_ani = P_ani
            
            # 2) Integrate with derivative of projectors
            self.pt.derivative(psit_nG, dP_aniv, q=k)
            kpt.dP_aniv = dP_aniv
