from math import sqrt, exp, pi
from gpaw.vdw import VanDerWaals
import Numeric as num
from gpaw.grid_descriptor import GridDescriptor
from gpaw.domain import Domain
import pylab
nfig=0

tolerance =0.000001

##################################################
#NON periodic test of H atom:
#q0
#C6 coefficients
#Energy'
#RPBE vs revPBE exchange
##################################################


n = 48
d = num.ones((2 * n, n, n))
a = 16.0
c = a / 2
h = a / n
            
uc = num.array([(2 * a, 0, 0),
                (0,     a, 0),
                (0,     0, a)])

dom=Domain(uc.flat[::4],periodic=(False,False,False))
gd=GridDescriptor(dom,d.shape)
dnon = gd.zeros()
nx,ny,nz=dnon.shape
for x in range(nx):
    for z in range(ny):
        for y in range(nz):
            r = sqrt((x * h - c*2.)**2 + (y * h - c)**2 + (z * h - c)**2)
            dnon[x, y, z] = exp(-2 * r) / pi

vdw_revPBE=VanDerWaals(dnon, gd=gd,xcname='revPBE',pbc=(False,False,False))
E_revPBE = vdw_revPBE.GetEnergy(n=1)
vdw_RPBE=VanDerWaals(dnon, gd=gd,xcname='RPBE',pbc=(False,False,False))
E_RPBE = vdw_RPBE.GetEnergy(n=1)





##########
#untag to plot phi
##########
#vdw_revPBE.plotphi()

##################################################
#Test of phi interpolation scheme
#This does at the moment only include a visual inspection
##################################################

def interpolation(ymax=8):
    Dtab = num.arange(0,vdw_revPBE.Dmax+vdw_revPBE.deltaD,vdw_revPBE.deltaD)
    nfig=60
    pylab.figure(nfig)
    nfig +=1
    #ymax=int(ymax/vdw_revPBE.deltaD)
    for n in [0,10,18]:
        pylab.plot(Dtab[:int(ymax/vdw_revPBE.deltaD)],vdw_revPBE.phimat[n,:int(ymax/vdw_revPBE.deltaD)],label='delta='+str(n*0.05))
    ##########
    #plot interpolated 
    ##########
    deltatab = num.arange(0,vdw_revPBE.deltamax+vdw_revPBE.deltadelta,vdw_revPBE.deltadelta)
    deltaD = vdw_revPBE.deltaD
    deltadelta = vdw_revPBE.deltadelta
    phitab_N = vdw_revPBE.phimat.copy()
    phitab_N.shape = [vdw_revPBE.phimat.shape[0]*vdw_revPBE.phimat.shape[1]]
    for n in [0,10,18]:
        pylab.plot(num.arange(0,ymax,0.001),vdw_revPBE.getphi(num.arange(0,ymax,0.001),(num.ones(len(num.arange(0,ymax,0.001)),num.Float)*(n*0.05)),Dtab,deltatab,deltaD,deltadelta,phitab_N),label='interpolated'+str(n*0.05))
    pylab.legend(loc='upper right')
    pylab.ylabel(r'$\phi *4\pi D^2$')
    pylab.xlabel(r'$D$')
    pylab.plot(Dtab[:int(ymax/vdw_revPBE.deltaD)],num.zeros(len(vdw_revPBE.phimat[0,:int(ymax/vdw_revPBE.deltaD)]),num.Float))
    pylab.show()





##########
#q0 test
#reference min, max and average
# 
##########
ref_label = ['min', 'max', 'average']
reference = num.array([0.024754860873480139,7.512329131951498,0.87798411512677765])

q0=vdw_RPBE.q0
q0test = num.array([min(min(min(q0))),max(max(max(q0))),num.sum(num.sum(num.sum(q0)))/(q0.shape[0]*q0.shape[1]*q0.shape[2])])-reference

for n in range(len(q0test)):
    if q0test[n] < tolerance:
        if q0test[n] < -tolerance:
            print 'q0 test failed'+ref_label[n] 

##########
#plotting q0 and density at x axis going through H atoms
##########

pylab.figure(nfig)
nfig +=1
pylab.plot(q0[:,int((ny-1.)/2.+1),int((nz-1.)/2.+1)],label = 'q0_nonpbc')
pylab.plot(dnon[:,int((ny-1.)/2.+1),int((nz-1.)/2.+1)]*10.,label = 'density_nonpbc *10')
pylab.legend()
#pylab.legend('upper right')
#pylab.show()


##########
#comparing gradient from vdw and analytic gradient
##########


analytic = gd.zeros()
nx,ny,nz=dnon.shape
for x in range(nx):
    for z in range(ny):
        for y in range(nz):
            r = sqrt((x * h - c*2.)**2 + (y * h - c)**2 + (z * h - c)**2)
            analytic[x, y, z] = (-2.*exp(-2 * r) / pi)**2.

a2_g=vdw_RPBE.a2_g
pylab.figure(nfig)
nfig +=1
#pylab.plot(a2_g[:,int((ny-1.)/2.+1),int((nz-1.)/2.+1)],label = 'a2_g_nonpbc')
#pylab.plot(analytic[:,int((ny-1.)/2.+1),int((nz-1.)/2.+1)],label = 'density_nonpbc *10')
pylab.plot(analytic[:,int((ny-1.)/2.+1),int((nz-1.)/2.+1)]-a2_g[:,int((ny-1.)/2.+1),int((nz-1.)/2.+1)],label = 'analytic-a2_g')
pylab.legend()
#pylab.legend('upper right')
#pylab.show()

##########
#C6 i Hartree
##########



C6 = []
C6_c = []

for n in [1,2,4,8]:
    C6_c.append(vdw_revPBE.GetC6_coarse(n=n)[0])
    C6.append(vdw_revPBE.GetC6(n=n)[0])














##################################################
#Periodic test of H atom:
#q0
#C6 coefficients
#Energy'
#RPBE vs revPBE exchange
##################################################

n = 48
d = num.ones((2 * n, n, n))
a = 16.0
c = a / 2
h = a / n
        
uc = num.array([(2 * a, 0, 0),
                (0,     a, 0),
                (0,     0, a)])

dom=Domain(uc.flat[::4],periodic=(True,True,True))
gd=GridDescriptor(dom,d.shape)
d = gd.zeros()
nx,ny,nz=d.shape
for x in range(nx):
    for z in range(ny):
        for y in range(nz):
            r = sqrt((x * h - c*2.)**2 + (y * h - c)**2 + (z * h - c)**2)
            d[x, y, z] = exp(-2 * r) / pi

vdw_revPBE=VanDerWaals(d, gd=gd,xcname='revPBE',pbc=(True,True,True))
E_revPBE = vdw_revPBE.GetEnergy(n=1)
vdw_RPBE=VanDerWaals(d, gd=gd,xcname='RPBE',pbc=(True,True,True))
E_RPBE = vdw_RPBE.GetEnergy(n=1)

##########
#Energy Test
#
# 
##########
ref_label = ['E_revPBE', 'E_RPBE']
reference = num.array([-0.69792719721511098,-0.6979223118689889])


test=[E_revPBE,E_RPBE]
for n in range(len(reference)):
    if abs(reference[n]- test[n]) > tolerance:
            print 'Energy test failed'+ref_label[n] 


























##########
#plotting q0 and density at x axis going through H atoms
##########


pylab.figure(nfig)
nfig +=1

pylab.plot(q0[:,int((ny)/2.),int((nz)/2.)],label = 'q0')
pylab.plot(d[:,int((ny)/2.),int((nz)/2.)]*10.,label = 'density *10')
pylab.legend()
#pylab.legend('upper right')
pylab.show()


##################################################
#Non periodic test of:
#Binding curve of H-H model molecule
##################################################



##################################################
#Periodic test of chain of H atoms:
#Enegy
##################################################







## n = 48
## d = num.ones((2 * n, n, n), num.Float)
## a = 16.0
## c = a / 2
## h = a / n
## for x in range(2 * n):
##     for z in range(n):
##         for y in range(n):
##             r = sqrt((x * h - c)**2 + (y * h - c)**2 + (z * h - c)**2)
##             d[x, y, z] = exp(-2 * r) / pi

## print num.sum(d.flat) * h**3
## uc = num.array([(2 * a, 0, 0),
##                 (0,     a, 0),
##                 (0,     0, a)])

## dom=Domain(uc.flat[::4],periodic=(True,True,True))
## gd=GridDescriptor(dom,d.shape)
## dnon = gd.zeros()
## nx,ny,nz=dnon.shape
## for x in range(nx):
##     for z in range(ny):
##         for y in range(nz):
##             r = sqrt((x * h - c)**2 + (y * h - c)**2 + (z * h - c)**2)
##             dnon[x, y, z] = exp(-2 * r) / pi

## vdw=VanDerWaals(d, unitcell=uc,xcname='revPBE',pbc=(True,True,True))
## vdw.coarsen(d,8)
## #vdw.GetC6()
## ##################################################
## #test using GridDescriptor
## #non periodic
## ##################################################

## e1 = VanDerWaals(dnon, gd=gd,xcname='revPBE',pbc=(False,False,False)).GetEnergy(n=4)
## dnon += dnon[::-1].copy()
## e2 = VanDerWaals(dnon, gd=gd,xcname='revPBE',pbc=(False,False,False)).GetEnergy(n=4)
## print  'revPBE',e1, e2, 2 * e1 - e2
## #RPBE
## #e1 = VanDerWaals(d, unitcell=uc,xcname='RPBE',pbc=(False,False,False)).GetEnergy(n=4)
## #d += d[::-1].copy()
## #e2 = VanDerWaals(d, unitcell=uc,xcname='RPBE',pbc=(False,False,False)).GetEnergy(n=4)
## #print 'RPBE',e1, e2, 2 * e1 - e2

## #pbc using mic
## for x in range(2 * n):
##     for z in range(n):
##         for y in range(n):
##             r = sqrt((x * h - c)**2 + (y * h - c)**2 + (z * h - c)**2)
##             d[x, y, z] = exp(-2 * r) / pi
                                    
## e1 = VanDerWaals(d, unitcell=uc,xcname='revPBE',pbc=(True,True,True)).GetEnergy(n=4)
## d += d[::-1].copy()
## e2 = VanDerWaals(d, unitcell=uc,xcname='revPBE',pbc=(True,True,True)).GetEnergy(n=4)
## print  'revPBE mic',e1, e2, 2 * e1 - e2
