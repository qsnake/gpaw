import numpy as npy
from gpaw.xc_functional import XCFunctional
#these are used for calculating the gradient
from gpaw.grid_descriptor import GridDescriptor
from gpaw.domain import Domain
from gpaw.xc_functional import XC3DGrid
from math import pi
from gpaw.transformers import Transformer
from gpaw.utilities import  check_unit_cell
from ASE.Units import Convert
import os

class VanDerWaals:
    # 
    def __init__(self,density,gd=None,unitcell=None,xcname='revPBE',pbc=(True,True,True)):
        #self.Nunit=periodic_number_unitcells
        self.periodic=pbc 
        self.xcname=xcname
        #the density must be given with the shape spin,gridpoint_x,gridpoint_y,gridpoint_z,
        #This class only works for non spin polarized calculations. In case of spinpolarized 
        #calcultions, one should ad the spin up and spin down  densities and use that as input.  
        #

        self.density = density
        if gd is None:
            unitcell = npy.array(unitcell)
            check_unit_cell(unitcell)
            unitcell=unitcell.ravel()[::4]
            gd = GridDescriptor(Domain(unitcell,periodic=self.periodic), self.density.shape)
        self.gd = gd
        self.h_c = gd.h_c
        #GGA Exchange og Correlation
        v = self.gd.zeros()
        xc = XC3DGrid(self.xcname, gd)
        self.GGA_xc_energy = xc.get_energy_and_potential(self.density, v)
        #Set self.a2_g
        self.a2_g = xc.a2_g
        #LDA correlation energy
        c = XC3DGrid('None-C_PW', gd)
        self.LDA_c_energy = c.get_energy_and_potential(self.density, v)
        #GGA exchangee energy
        if self.xcname == 'revPBE':
            exchange = 'X_PBE_R-None'
        if self.xcname == 'RPBE':
            exchange = 'X_RPBE-None'
        #x = XC3DGrid(self.xcname+'x', gd)
        x = XC3DGrid(exchange, gd)
        self.GGA_x_energy = x.get_energy_and_potential(self.density, v)
        #self.density=npy.choose(npy.less_equal(density,ncut_dens),(density,ncut_dens))
        #hardcoded the density min is set to 1*10^-7
        ncut_dens=0.0000001
        self.density=npy.choose(npy.less_equal(density,ncut_dens),(density,ncut_dens))        
        self.k_f=(3.0*pi**2*self.density)**(1.0/3.0)
        self.q0=self.getqzero()
        self.phimat = self.get_phitab_from_1darrays()

    def get_e_xc_LDA(self):
        e_xc_LDA=self.get_e_xc_LDA_c()+self.get_e_x_LDA()
        return e_xc_LDA
    
    def get_e_xc_LDA_c(self):
        #this is for one spin only
        #PW91 LDA correlation
        c=1.7099210
        n_up=self.density/2.0
        n_down=self.density/2.0
        #nt=abs(n_up+n_down)
        n=self.density
        #npy.choose(npy.less_equal(nt,0.00001),(nt,0.00001))
        r=(3./(4.*npy.pi*n))**(1./3.)
        zeta=(n_up-n_down)/n
        wz=((1.+zeta)**(4./3.)+(1.-zeta)**(4./3.)-2.)/(2.**(4./3.)-2.)
        res=self.e_PW92_LDA(r,0.031091,0.21370,7.5957,3.5876,1.6382,0.49294,1.)*(1.-wz*zeta**4.)
        res=res+self.e_PW92_LDA(r,0.015545,0.20548,14.1189,6.1977,3.3662,0.62517,1.)*wz*zeta**4.
        res=res-self.e_PW92_LDA(r,0.016887,0.11125,10.357,3.6231,0.88026,0.49671,1.)*wz*(1.-zeta**4.)/c
        return(res)
#function used by def eps_c_PW92_LDA (n_up,n_down):
    def e_PW92_LDA (self,r,t,u,v,w,x,y,p):
        return(-2.*t*(1.+u*r)*npy.log(1.+1./(2.*t*(v*npy.sqrt(r)+w*r+x*r**(3./2.)+y*r**(p+1.)))))
    
    def get_e_x_LDA(self):
        result = (-3./(4.*npy.pi)*(3.*npy.pi*npy.pi*self.density)**(1./3.))
        return result
    
    def getqzero(self):
        #implementet as in PRL92(2004)246401-1
        e_xc_0 = self.get_e_xc_LDA()-self.get_e_x_LDA()*(-0.8491/9.0*self.a2_g/(2.0*self.k_f*self.density)**2.0)
        q_0 = e_xc_0/self.get_e_x_LDA()*self.k_f
        return q_0
    
    def get_phitab_from_1darrays(self, filename='phi_delta'):
        path = os.environ['VDW']
        #function that constucts phimat from files containing phi_delta(D)
        #The filename must be given as something+delta
        #
        file = open(path + '/grid.dat')
        phigrid = {}
        line = file.readline()
        while line:
            a = line.split()
            phigrid[a[0]] = float(a[1])
            line = file.readline()
        file.close()
        self.deltadelta = phigrid['deltadelta']
        self.deltaD=phigrid['deltaD']
        self.Dmax=phigrid['Dmax']
        self.deltamax=phigrid['deltamax']
        x = {}
        #filename='eta_2_phi_delta'
        faktor = 2.0*4.0*pi/pi**2.0
        for n in npy.arange(0.0,1.0,self.deltadelta):
            f = path+filename+str(n)+'.dat'
            data = self.read_array1d_from_txt_file(f)
            x[n] = npy.array(data[:])*faktor 
        #h=0.05 for D og delta 
        phimat = npy.zeros((len(x),len(x[0.0])))
        for n in range(0,phimat.shape[0]):
            for m in range(phimat.shape[1]):
                phimat[n,m] = x[n*0.05][m]
        return phimat
    
    def getphi(self,D,delta,Dtab,deltatab,deltaD,deltadelta,phitab_N):
        #a linear interpolation scheme
        #Asymtotisk graense
        C = 12.0*(4.0*pi/9.0)**3.0
        ddot2 = (D*(1-delta))**2.0
        d2 = (D*(1+delta))**2.0
        phi_asym = -C/(d2*ddot2*(d2+ddot2))
        mask = npy.where(D>=Dtab[-1],1,0)
        #phi(D=0, delta=x)=0 per definition
        #D is set to max int phitab, to make the interpolation possible
        D = npy.choose(mask,(D,Dtab[-1]-deltaD/100.0))
        #dette er aendre her phi(D=0, delta=x)=0 per definition
        n_D = (D/deltaD).astype(int) #-1 because Dtab starts at h and not 0
        #delta above the upper limit of delta in phitab is set to just below the upper limit
        delta = npy.choose(npy.greater_equal(delta,deltatab[len(deltatab)-1]),(delta,deltatab[len(deltatab)-1]-deltadelta/100.00))
        n_delta = (delta/deltadelta).astype(int)
        #
        t = (D-(n_D)*deltaD)/deltaD  
        u = (delta-n_delta*deltadelta)/deltadelta
        hack1 = npy.take(phitab_N,(n_D+n_delta*len(Dtab)))
        hack2 = npy.take(phitab_N,(n_D+1+n_delta*len(Dtab)))
        hack3 = npy.take(phitab_N,(n_D+1+(n_delta+1)*len(Dtab)))
        hack4 = npy.take(phitab_N,(n_D+(n_delta+1)*len(Dtab)))
        Phi = (1-t)*(1-u)*hack1+t*(1-u)*hack2+t*u*hack3+(1-t)*u*hack4
        npy.putmask(Phi,mask,phi_asym)
        return Phi
    def coarsen(self,oldgrid,n):
        ##################################################
        #function for coarsening the grid
        ##################################################
        if n == 1: return oldgrid
        coarsegd = self.gd.coarsen()
        t = Transformer(self.gd,coarsegd)
        a=coarsegd.empty()
        t.apply(oldgrid,a)
        print '1'
        if n == 2: return a
        verycoarsegd = coarsegd.coarsen()
        t = Transformer(coarsegd,verycoarsegd)
        b=verycoarsegd.empty()
        t.apply(a,b)
        print '2'
        if n==4: return b
        veryverycoarsegd = verycoarsegd.coarsen()
        t = Transformer(verycoarsegd,veryverycoarsegd)
        c=veryverycoarsegd.empty()
        t.apply(b,c)
        print '3'
        if n==8: return c    

    def int_6D_n_D2_cut_periodic_mic(self):
        ###################################################
        #introduces periodic boundary conditions using
        #the minimum image convention
        ###################################################
        ###imports arrays used by GetPhi
        Dtab = npy.arange(0,self.Dmax+self.deltaD,self.deltaD)
        deltatab = npy.arange(0,self.deltamax+self.deltadelta,self.deltadelta)
        deltaD = self.deltaD
        deltadelta = self.deltadelta
        phitab_N = self.phimat.copy()
        phitab_N.shape = [self.phimat.shape[0]*self.phimat.shape[1]]
        #import parameters used local
        n = self.n
        ncut = self.ncut
        h_c = self.h_c
        self.test=denstab =self.coarsen(self.density,n)
        denstab = self.density
        nx, ny, nz = self.density[::n,::n,::n].shape
        R = npy.zeros((nx, ny, nz, 3))
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    R[x, y, z] = [x, y, z]*h_c*n


        N = nx * ny * nz
        R.shape = (N,3)
        qtab_N = self.q0[::n,::n,::n].copy()
        #print qtab_N.shape
        qtab_N.shape = [N]
        denstab_N = denstab[::n,::n,::n].copy()
        denstab_N.shape = [N]
        qtab_N = npy.compress(npy.greater_equal(denstab_N,ncut),qtab_N)
        R=npy.compress(npy.greater_equal(denstab_N,ncut),R,axis=0)
        denstab_N=npy.compress(npy.greater_equal(denstab_N,ncut),denstab_N)
        #for analysis
        self.denstab_N=denstab_N
        print 'denstab_N.shape', denstab_N.shape[0]
        E_cl = 0.0
        #self.trackEnl = []
        uc = self.h_c*self.gd.N_c
        #for nn in range(len(self.periodic)):
        #    if self.periodic == False:
        #         uc[nn] = uc[nn]*4.
        for m in range(denstab_N.shape[0]):
            Rm = R[m]
            t = R - Rm
            for mm in range(len(self.periodic)):
                if self.periodic[mm]:
                    t[:,mm]=(t[:,mm]+(3./2.)*uc[mm])%uc[mm]-uc[mm]/2.0
            #tmic=(t+(3./2.)*uc)%uc-uc/2.0
            #r = npy.sqrt(npy.sum(tmic**2.0,axis=1))
            #print r
            r = npy.sqrt(npy.sum(t**2.0,axis=1))
            #self.r=r
            D = (qtab_N[m]+qtab_N)*r/2.0
            #The next line is a work around singularities for D=0
            Dmult = npy.choose(npy.less(D,1e-4),(D,10.0**8))
            #I have set delta to be positive, is this a definition?
            delta = npy.absolute(qtab_N[m]-qtab_N)/(qtab_N[m]+qtab_N)
            E_tmp = npy.sum(denstab_N[m]*denstab_N[:]*self.getphi(D,delta,Dtab,deltatab,deltaD,deltadelta,phitab_N)/(npy.pi*4.0*Dmult**2))
            #self.trackEnl.append(E_tmp)
            E_cl = E_cl+E_tmp
            #print E_tmp
            #print D
        E_cl = 0.5*E_cl*n**6*h_c[0]**2.0*h_c[1]**2.0*h_c[2]**2.0
        #print denstab.shape
        self.E_cl = E_cl
        return E_cl

    def int_6D_n_D2_cut_periodic_mic_coarsen(self):
        ###################################################
        #introduces periodic boundary conditions using
        #the minimum image convention
        ###################################################
        ###imports arrays used by GetPhi
        Dtab = npy.arange(0,self.Dmax+self.deltaD,self.deltaD)
        deltatab = npy.arange(0,self.deltamax+self.deltadelta,self.deltadelta)
        deltaD = self.deltaD
        deltadelta = self.deltadelta
        phitab_N = self.phimat.copy()
        phitab_N.shape = [self.phimat.shape[0]*self.phimat.shape[1]]
        #import parameters used local
        n = self.n
        ncut = self.ncut
        h_c = self.h_c
        denstab =self.coarsen(self.density,n)
        
        #denstab = self.density
        print 'denstab.shape', denstab.shape
        nx, ny, nz = denstab.shape
        #self.density[::n,::n,::n].shape
        R = npy.zeros((nx, ny, nz, 3))
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    R[x, y, z] = [x, y, z]*h_c*n


        N = nx * ny * nz
        print N
        R.shape = (N,3)
        qtab_N = self.coarsen(self.q0.copy(),n)
        print 'qtab_N.shape',qtab_N.shape
        qtab_N.shape = [N]
        #denstab_N = denstab[::n,::n,::n].copy()
        denstab_N = denstab.copy()
        denstab_N.shape = [N]
        qtab_N = npy.compress(npy.greater_equal(denstab_N,ncut),qtab_N)
        R=npy.compress(npy.greater_equal(denstab_N,ncut),R,axis=0)
        denstab_N=npy.compress(npy.greater_equal(denstab_N,ncut),denstab_N)
        #for analysis
        self.denstab_N=denstab_N
        print 'denstab_N.shape', denstab_N.shape[0]
        E_cl = 0.0
        #self.trackEnl = []
        uc = self.h_c*self.gd.N_c
        #for nn in range(len(self.periodic)):
        #    if self.periodic == False:
        #         uc[nn] = uc[nn]*4.
        for m in range(denstab_N.shape[0]):
            Rm = R[m]
            t = R - Rm
            for mm in range(len(self.periodic)):
                if self.periodic[mm]:
                    t[:,mm]=(t[:,mm]+(3./2.)*uc[mm])%uc[mm]-uc[mm]/2.0
            #tmic=(t+(3./2.)*uc)%uc-uc/2.0
            #r = npy.sqrt(npy.sum(tmic**2.0,axis=1))
            #print r
            r = npy.sqrt(npy.sum(t**2.0,axis=1))
            #self.r=r
            D = (qtab_N[m]+qtab_N)*r/2.0
            #The next line is a work around singularities for D=0
            Dmult = npy.choose(npy.less(D,1e-4),(D,10.0**8))
            #I have set delta to be positive, is this a definition?
            delta = npy.absolute(qtab_N[m]-qtab_N)/(qtab_N[m]+qtab_N)
            E_tmp = npy.sum(denstab_N[m]*denstab_N[:]*self.getphi(D,delta,Dtab,deltatab,deltaD,deltadelta,phitab_N)/(npy.pi*4.0*Dmult**2))
            #self.trackEnl.append(E_tmp)
            E_cl = E_cl+E_tmp
            #print E_tmp
            #print D
        E_cl = 0.5*E_cl*n**6*h_c[0]**2.0*h_c[1]**2.0*h_c[2]**2.0
        #print denstab.shape
        self.E_cl = E_cl
        return E_cl

    def GetEnergy(self,n=1,ncut=0.0005):
        self.n = n
        self.ncut = ncut
        print self.periodic
        #E_nl = -self.GGA_xc_energy+self.int_6D_n_D2_cut_periodic_mic()+self.LDA_c_energy+self.GGA_x_energy
        #test coarsen
        E_nl_c = -self.GGA_xc_energy+self.int_6D_n_D2_cut_periodic_mic_coarsen()+self.LDA_c_energy+self.GGA_x_energy
        #return Convert(E_nl,'Hartree','eV'),Convert(E_nl_c,'Hartree','eV')
        return Convert(E_nl_c,'Hartree','eV')
    def plotphi(self,ymax=8):
        import pylab as pylab
        Dtab = npy.arange(0,self.Dmax+self.deltaD,self.deltaD)
        pylab.ion()
        pylab.figure(55)
        ymax=int(ymax/self.deltaD)
        for n in [0,10,18]:
            pylab.plot(Dtab[:ymax],self.phimat[n,:ymax],label='delta='+str(n*0.05))
        pylab.legend(loc='upper right')
        pylab.ylabel(r'$\phi *4\pi D^2$')
        pylab.xlabel(r'$D$')
        pylab.plot(Dtab[:ymax],npy.zeros(len(self.phimat[0,:ymax])))
        pylab.show()
    def GetC6(self,n=1,ncut=0.0005):
        #Returns C6 in units of Hartree
        ncut=ncut
        h_c = self.h_c
        denstab=self.density
        nx, ny, nz = denstab[::n,::n,::n].shape
        print 'denstab.shape' ,denstab.shape
        N = nx * ny * nz
        print 'N' , N
        qtab_N = self.q0[::n,::n,::n].copy()
        print 'qtab_N.shape',qtab_N.shape
        qtab_N.shape = [N]

        denstab_N = denstab[::n,::n,::n].copy()
        denstab_N.shape = [N]
        print 'denstab_N.shape', denstab_N.shape
        qtab_N = npy.compress(npy.greater_equal(denstab_N,ncut),qtab_N)
        denstab_N = npy.compress(npy.greater_equal(denstab_N,ncut),denstab_N)
        #print denstab_N
        #print 'B:h_c[0]',h_c[0]
        #print denstab.shape
        print 'denstab_N.shape[0]', denstab_N.shape[0]
        C6 = 0.0
        C=(-12.*(4.*npy.pi/9.)**3)
        for m in range(denstab_N.shape[0]):
            #print C6
            C6 = C6+npy.sum(denstab_N[m]*denstab_N[:]*C/(qtab_N[m]**2*qtab_N[:]**2*(qtab_N[m]**2+qtab_N[:]**2)))
        #print 'C:h_c[0]',h_c[:], 'n=', n
        #print 'udenfor loop C6=',C6
        #print 'norm', n**6*h_c[0]**2.0*h_c[1]**2.0*h_c[2]**2.0
        C6 = -C6*n**6*h_c[0]**2.0*h_c[1]**2.0*h_c[2]**2.0
        #print denstab.shape
        Ry = 13.6058
        self.mik = qtab_N
        self.mikd = denstab_N
        return C6 ,'Ha*a0**6'

    def GetC6_coarse(self,n=1,ncut=0.0005):
        #Returns C6 in units of Hartree
        ncut=ncut
        h_c = self.h_c
        denstab =self.coarsen(self.density,n)
        print 'denstab.shape' ,denstab.shape
        nx, ny, nz = denstab.shape
        N = nx * ny * nz
        print 'N',N
        qtab_N = self.coarsen(self.q0.copy(),n)
        qtab_N.shape = [N]
        denstab_N = denstab.copy()
        denstab_N.shape = [N]
        print 'denstab_N.shape', denstab_N.shape
        qtab_N = npy.compress(npy.greater_equal(denstab_N,ncut),qtab_N)
        denstab_N = npy.compress(npy.greater_equal(denstab_N,ncut),denstab_N)
        C6 = 0.0
        C=(-12.*(4.*npy.pi/9.)**3)
        for m in range(denstab_N.shape[0]):
            C6 = C6+npy.sum(denstab_N[m]*denstab_N[:]*C/(qtab_N[m]**2*qtab_N[:]**2*(qtab_N[m]**2+qtab_N[:]**2)))
        C6 = -C6*n**6*h_c[0]**2.0*h_c[1]**2.0*h_c[2]**2.0
        Ry = 13.6058
        return C6 ,'Ha*a0**6'

    def read_array1d_from_txt_file(self,filename='Phi5_D0_10_1delta_0_09_01.dat'):
        file = open(filename)
        line = file.readline()
        filearray1D = []
        while line:
            filearray1D.append(float(line))
            line=file.readline()
        file.close()
        return filearray1D
        



