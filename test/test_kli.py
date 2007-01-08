from ASE import Atom, ListOfAtoms
from gpaw import Calculator

# First, lets test the groundstate of hydrogen atom.
# This should be the same as in Hartree-Fock.
# Compare the result to Hartree-Fock calculation.

"""
a = 4.0
for d in [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]:

    d = d * 0.5291

    H2 = ListOfAtoms([Atom('H',(a/2-d/2,a/2,a/2)),
                      Atom('H',(a/2+d/2,a/2,a/2))],                      
                     periodic=False,
                     cell=(a, a, a));

    H2.SetCalculator(Calculator(h=0.2, setups='ae', xc='KLI'))
    e2_kli = H2.GetPotentialEnergy()
    H2.SetCalculator(Calculator(h=0.2, setups='ae', xc='EXX'))
    e2_exx = H2.GetPotentialEnergy()
                                
    print "DATAROW", d/0.5291, e2_kli/27.211, e2_exx/27.211
"""

# This gives approximately correct behaviour

b = 0.5291
a = 12

c = a / 2.0
H6 = ListOfAtoms(
       [Atom('H', (-1*b+c,  -2*b+c, -1*b+c)),
	Atom('H', (1*b+c,    1*b+c,  2*b+c)),
	Atom('H', (-1.5*b+c,  1*b+c,  2*b+c)),
	Atom('H', (c+0.1*b, c+0.1*b, c+0.1*b))],
       periodic=False,
        cell=(a,a,a));

for h2 in [0.4, 0.3, 0.2, 0.1]:
	H6.SetCalculator(Calculator(h=h2,
				    setups='ae',
				    xc = 'KLI',
				    spinpol = False))
	e2 = H6.GetPotentialEnergy()
	print "DATAROW", h2, e2


"""
 ITERATION  ENERGY          1e-ENERGY        2e-ENERGY     NORM[dD(SAO)]  TOL
   8  -1.9739928266394    -6.4141112244     2.2503730772    0.207D-03 0.238D-09
                            Exc =    -1.058279281108     N = 3.9999976760
                 Dev. HF-Energy =     0.03286602 eV
          max. resid. norm for Fia-block=  6.149D-03 for orbital      3a
          max. resid. fock norm         =  2.353D-02 for orbital      5a

 convergence criteria satisfied after  8 iterations


                  ------------------------------------------
                 |  total energy      =     -1.97399282664  |
                  ------------------------------------------
                 :  kinetic energy    =      1.75399614560  :
                 :  potential energy  =     -3.72798897224  :
                 :  virial theorem    =      1.88855244149  :
                 :  wavefunction norm =      1.00000000000  :
                  ..........................................


 orbitals $scfmo  will be written to file mos

    irrep                  1a          2a          3a          4a          5a
 eigenvalues H         -0.57482    -0.30826    -0.26362    -0.17792    -0.05172
            eV         -15.6417     -8.3882     -7.1735     -4.8415     -1.4075
 occupation              2.0000      2.0000

"""
