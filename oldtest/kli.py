from gpaw.atom.all_electron import AllElectron


def check(name, atomE, atomeig, total, HOMO):

    #print name, "   ", atomE,"/", total, " | ", atomeig,"/",HOMO," |"
    print "|",name, "  |%12.4f/%12.4f | %12.4f/%12.4f |" % (atomE,total,atomeig,HOMO)
    assert abs(atomE-total)<8e-3
    assert abs(atomeig-HOMO)<1e-3

A = AllElectron('He', 'KLI', False)
A.run()
HeE = A.Ekin+A.Epot+A.Exc
Heeig = A.e_j[0]

A = AllElectron('Be', 'KLI', False)
A.run()

BeE = A.Ekin+A.Epot+A.Exc
Beeig = A.e_j[1]

A = AllElectron('Ne', 'KLI', False)
A.run()
NeE = A.Ekin+A.Epot+A.Exc
Neeig = A.e_j[2]

A = AllElectron('Mg', 'KLI', False)
A.run()
MgE = A.Ekin+A.Epot+A.Exc
Mgeig = A.e_j[3]

print "Checking calculated KLI-all-electron data..."
print "---------------------------------------------------------------"
print "| Atom |  Total(KLI)/Total(Ref)   | eig. homo.(KLI)/(Ref)     |"
print "---------------------------------------------------------------"

check('Be',BeE, Beeig,  (-29.1460+0.3e-3)/2, -0.6177/2)
check('Ne',NeE, Neeig, (-257.0942+1.1e-3)/2, -1.6988/2)
check('Mg',MgE, Mgeig, (-399.2292+1.8e-3)/2, -0.5048/2)
print "---------------------------------------------------------------"
print "(ref) p. 109 and 113 of Phys. Rev. A, 45 p. 101"
print
