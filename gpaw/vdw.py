import Numeric as num
from gpaw.xc_functional import XCFunctional
#these are used for calculating the gradient
from gpaw.grid_descriptor import GridDescriptor
from gpaw.domain import Domain
from gpaw.xc_functional import XC3DGrid
from math import pi
from ReadArray1DFromTxtFile import ReadArray1DFromTxtFile
from gpaw.utilities import  check_unit_cell
from ASE.Units import Convert
import os

class VanDerWaals:
    # 
    def __init__(self,density,gd=None,unitcell=None,xcname='revPBE',pbc=None):
        #self.Nunit=periodic_number_unitcells
        self.periodic=pbc 
        self.xcname=xcname
        #hardcoded the density min is set to 1*10^-7
        #the density must be given with the shape spin,gridpoint_x,gridpoint_y,gridpoint_z,
        #this class only works for non spin polarized calculations
        ncut_dens=0.0000001
        self.density=num.choose(num.less_equal(density,ncut_dens),(density,ncut_dens))
        if gd is None:
            unitcell = num.array(unitcell)
            check_unit_cell(unitcell)
            unitcell=unitcell.flat[::4]
            gd = GridDescriptor(Domain(unitcell), self.density.shape)
        self.gd = gd
        self.h_c = gd.h_c
        #GGA Exchange og Correlation
        v = self.gd.new_array()
        xc = XC3DGrid(self.xcname, gd)
        self.GGA_xc_energy = xc.get_energy_and_potential(self.density, v)
        #Set self.a2_g
        self.a2_g = xc.a2_g
        #LDA correlation energy
        c = XC3DGrid('LDAc', gd)
        self.LDA_c_energy = c.get_energy_and_potential(self.density, v)
        #GGA exchangee energy
        x = XC3DGrid(self.xcname+'x', gd)
        self.GGA_x_energy = x.get_energy_and_potential(self.density, v)
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
        #num.choose(num.less_equal(nt,0.00001),(nt,0.00001))
        r=(3./(4.*num.pi*n))**(1./3.)
        zeta=(n_up-n_down)/n
        wz=((1.+zeta)**(4./3.)+(1.-zeta)**(4./3.)-2.)/(2.**(4./3.)-2.)
        res=self.e_PW92_LDA(r,0.031091,0.21370,7.5957,3.5876,1.6382,0.49294,1.)*(1.-wz*zeta**4.)
        res=res+self.e_PW92_LDA(r,0.015545,0.20548,14.1189,6.1977,3.3662,0.62517,1.)*wz*zeta**4.
        res=res-self.e_PW92_LDA(r,0.016887,0.11125,10.357,3.6231,0.88026,0.49671,1.)*wz*(1.-zeta**4.)/c
        return(res)
#function used by def eps_c_PW92_LDA (n_up,n_down):
    def e_PW92_LDA (self,r,t,u,v,w,x,y,p):
        return(-2.*t*(1.+u*r)*num.log(1.+1./(2.*t*(v*num.sqrt(r)+w*r+x*r**(3./2.)+y*r**(p+1.)))))
    
    def get_e_x_LDA(self):
        result = (-3./(4.*num.pi)*(3.*num.pi*num.pi*self.density)**(1./3.))
        return result
    
    def getqzero(self):
        #implementet as in PRL92(2004)246401-1
        e_xc_0 = self.get_e_xc_LDA()-self.get_e_x_LDA()*(-0.8491/9.0*self.a2_g/(2.0*self.k_f*self.density)**2.0)
        q_0 = e_xc_0/self.get_e_x_LDA()*self.k_f
        return q_0
    
    def get_phitab_from_1darrays(self,filename='phi_delta'):
        path=os.environ['VDW']
        #function that constucts phimat from files containing phi_delta(D)
        #The filename must be given as something+delta
        #
        file = open(path+'grid.dat')
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
        for n in num.arange(0.0,1.0,self.deltadelta):
            f = path+filename+str(n)+'.dat'
            data = self.read_array1d_from_txt_file(f)
            x[n] = num.array(data[:])*faktor 
        #h=0.05 for D og delta 
        phimat = num.zeros((len(x),len(x[0.0])),num.Float)
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
        mask = num.where(D>=Dtab[-1],1,0)
        #phi(D=0, delta=x)=0 per definition
        #D is set to max int phitab, to make the interpolation possible
        D = num.choose(mask,(D,Dtab[-1]-deltaD/100.0))
        #dette er aendre her phi(D=0, delta=x)=0 per definition
        n_D = (D/deltaD).astype(num.Int) #-1 fordi Dtab starter på h og ikke 0
        #delta above the upper limit of delta in phitab is set to just below the upper limit
        delta = num.choose(num.greater_equal(delta,deltatab[len(deltatab)-1]),(delta,deltatab[len(deltatab)-1]-deltadelta/100.00))
        n_delta = (delta/deltadelta).astype(num.Int)
        #
        t = (D-(n_D)*deltaD)/deltaD  
        u = (delta-n_delta*deltadelta)/deltadelta
        hack1 = num.take(phitab_N,(n_D+n_delta*len(Dtab)))
        hack2 = num.take(phitab_N,(n_D+1+n_delta*len(Dtab)))
        hack3 = num.take(phitab_N,(n_D+1+(n_delta+1)*len(Dtab)))
        hack4 = num.take(phitab_N,(n_D+(n_delta+1)*len(Dtab)))
        Phi = (1-t)*(1-u)*hack1+t*(1-u)*hack2+t*u*hack3+(1-t)*u*hack4
        num.putmask(Phi,mask,phi_asym)
        return Phi
    
    def int_6D_n_D2_cut(self):
        ###imports arrays used by GetPhi
        Dtab = num.arange(0,self.Dmax+self.deltaD,self.deltaD)
        deltatab = num.arange(0,self.deltamax+self.deltadelta,self.deltadelta)
        deltaD = self.deltaD
        deltadelta = self.deltadelta
        phitab_N = self.phimat.copy()
        phitab_N.shape = [self.phimat.shape[0]*self.phimat.shape[1]]
        #import parameters used local
        n = self.n
        ncut = self.ncut
        h_c = self.h_c
        denstab = self.density
        nx, ny, nz = self.density[::n,::n,::n].shape
        R = num.zeros((nx, ny, nz, 3), num.Float)
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
        qtab_N = num.compress(num.greater_equal(denstab_N,ncut),qtab_N)
        R=num.compress(num.greater_equal(denstab_N,ncut),R,axis=0)
        denstab_N=num.compress(num.greater_equal(denstab_N,ncut),denstab_N)
        #for analysis
        self.denstab_N=denstab_N
        print 'denstab_N.shape', denstab_N.shape[0]
        E_cl = 0.0
        self.trackEnl = []
        for m in range(denstab_N.shape[0]):
            Rm = R[m]
            t = R - Rm
            r = num.sqrt(num.sum(t**2.0,axis=1))
            D = (qtab_N[m]+qtab_N)*r/2.0
            #The next line is a work around singularities for D=0
            Dmult = num.choose(num.equal(D,0),(D,10.0**8))
            #I have set delta to be positive, is this a definition?
            delta = num.absolute(qtab_N[m]-qtab_N)/(qtab_N[m]+qtab_N)
            E_tmp = num.sum(denstab_N[m]*denstab_N[:]*self.getphi(D,delta,Dtab,deltatab,deltaD,deltadelta,phitab_N)/(num.pi*4.0*Dmult**2))
            self.trackEnl.append(E_tmp)
            E_cl = E_cl+E_tmp
        E_cl = 0.5*E_cl*n**6*h_c[0]**2.0*h_c[1]**2.0*h_c[2]**2.0
        #print denstab.shape
        self.E_cl = E_cl
        return E_cl
    def GetEnergy(self,n=1,ncut=0.0005):
        self.n = n
        self.ncut = 0.0005
        if self.periodic is None:
            E_nl = -self.GGA_xc_energy+self.int_6D_n_D2_cut()+self.LDA_c_energy+self.GGA_x_energy
            return Convert(E_nl,'Hartree','eV')
##         else:
##             E_nl = -self.Get_xc_energy()+self.int_6Dper_n_D2_cut()+self.LDA_c_energy()+self.GGA_x_energy()
##             return Convert(E_nl,'Hartree','eV')
    def plotphi(self):
        import pylab as pylab
        Dtab = num.arange(0,self.Dmax+self.deltaD,self.deltaD)
        pylab.ion()
        for n in range(self.phimat.shape[0]):
            pylab.figure(n)
            pylab.plot(Dtab,phimat[n,:])
    
    def GetC6(self):
        #Returns C6 in units of Hartree
        h_c = self.h_c

        s, nx, ny, nz = denstab[:,::n,::n,::n].shape
        #print denstab.shape
        N = nx * ny * nz
        print N
        qtab_N = qtab[0,::n,::n,::n].copy()
        #print qtab_N.shape
        qtab_N.shape = [N]
        denstab_N = denstab[:,::n,::n,::n].copy()
        denstab_N.shape = [N]
        qtab_N = num.compress(num.greater_equal(denstab_N,ncut),qtab_N)
        denstab_N = num.compress(num.greater_equal(denstab_N,ncut),denstab_N)
        #print denstab_N
        #print 'B:h_c[0]',h_c[0]
        #print denstab.shape
        #print 'denstab_N[0].shape', denstab_N[0].shape
        C6 = 0.0
        for m in range(denstab_N.shape[0]):
            C6 = C6+num.sum(denstab_N[m]*denstab_N[:]*-(12.*(4.*num.pi/9.)**3)/(qtab_N[m]**2*qtab_N[:]**2*(qtab_N[m]**2+qtab_N[:]**2)))
        #print 'C:h_c[0]',h_c[:], 'n=', n
        #print 'udenfor loop C6=',C6
        #print 'norm', n**6*h_c[0]**2.0*h_c[1]**2.0*h_c[2]**2.0
        C6 = -0.5*C6*n**6*h_c[0]**2.0*h_c[1]**2.0*h_c[2]**2.0
        #print denstab.shape
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
        


######hertil













## from gpaw.grid_descriptor import GridDescriptor
## from gpaw.domain import Domain
## from gpaw.xc_functional import XC3DGrid

## N = 24
## a = 5.0
## domain  = Domain((a, a, a))
## gd = GridDescriptor(domain, (N, N, N))
## xc = XC3DGrid('PBE', gd)
## n = gd.new_array()
## v = gd.new_array()
## n[:] = 0.1
## print gd.integrate(n)
## E = xc.get_energy_and_potential(n, v)
## a2 = xc.a2_g

