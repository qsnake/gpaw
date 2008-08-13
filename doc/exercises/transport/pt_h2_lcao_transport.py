from ase.transport.calculators import TransportCalculator
import numpy as npy
import pickle


#Read in the hamiltonians
h, s = pickle.load(file('scat_hs.pickle'))
h1, s1 = pickle.load(file('lead1_hs.pickle'))
h2, s2 = pickle.load(file('lead2_hs.pickle'))
pl1 = len(h1) / 2 # left principal layer size
pl2 = len(h2) / 2 # right principal layer size

tcalc = TransportCalculator(h=h, h1=h1, h2=h2, #hamiltonian matrices
                            s=s, s1=s1, s2=s2, #overlap matrices
                            pl1=pl1, pl2=pl2,  #principal layer sizes
                            energies=[0.0],    #energies
                            align_bf=1,        #align the the Fermi levels
                            verbose=False)      #print extra information?

#Calculate the conductance (the zero (0.0) energy corresponds to the
#Fermi level)
G = tcalc.get_transmission()[0]
print "Conductance: %.2f 2e^2/h" % G
#Change the desired range of energies
energies = npy.arange(-8,4,0.05)
tcalc.set(energies=energies)

#The basis functions of the two Hydrogen atoms
Pt_N = 5 # 
Pt_nbf = 9 #number of bf per Pt atom (basis=sz)
H_nbf = 4  # number of bf per H atom (basis=sz)
bf_H1 = Pt_nbf * Pt_N
bfs = range(bf_H1, bf_H1 + 2 * H_nbf)
print bfs
#Calculate the transmission and the projected density of states (pdos)
#of the Hydrogen atoms basis functions.
print "Calculating the transmission (T) and the H atomic orbital pdos (pdos_je)"
tcalc.set(pdos=bfs)
T_e = tcalc.get_transmission()
pdos_je = tcalc.get_pdos()

#Diagonalize the subspace corresponding the the Hydrogen molecule
print "Diagonalizing the H2 subspace and recalculates the pdos"
h_rot, s_rot, eps_n, vec_jn = tcalc.subdiagonalize_bfs(bfs)
for n in range(len(eps_n)):
    print "bf %i correpsonds to the eigenvalue %.2f eV" % (bfs[n],eps_n[n])

#Set the rotated hamiltonian and overlap matrices corresponding
#to the new basis spanning the molecular subspace.
tcalc.set(h=h_rot,s=s_rot)
#calculate the pdos of the new basis functions
pdos_rot_je = tcalc.get_pdos()

#Cut the coupling to the anti-bonding orbital.
print "Cutting the coupling to the renormalized molecular state at %.2f eV" % eps_n[1]
h_rot_cut, s_rot_cut = tcalc.cutcoupling_bfs([bfs[1]])
tcalc.set(h=h_rot_cut,s=s_rot_cut)
T_cut_antibonding_e = tcalc.get_transmission()

#Cut the coupling to the boning-orbital.
print "Cutting the coupling to the renormalized molecular state at %.2f eV" % eps_n[0]
tcalc.set(h=h_rot,s=s_rot)
h_rot_cut, s_rot_cut = tcalc.cutcoupling_bfs([bfs[0]])
tcalc.set(h=h_rot_cut,s=s_rot_cut)
T_cut_bonding_e = tcalc.get_transmission()

#Dump the data to txt files.
for j in range(2):
    fd = file('pdos%.2i.dat' % j,'w') 
    for e, pdos in zip(energies,pdos_je[j]):
        print >> fd, e, pdos

for j in range(2):
    fd = file('pdos_rot%.2i.dat' % j,'w') 
    for e, pdos in zip(energies,pdos_rot_je[j]):
        print >> fd, e, pdos

fd = file('T.dat','w')
for e,t in zip(energies,T_e):
    print >>fd, e, t

fd = file('T_cut_bonding.dat','w')
for e,t in zip(energies,T_cut_bonding_e):
    print >>fd, e, t

fd = file('T_cut_antibonding.dat','w')
for e,t in zip(energies,T_cut_antibonding_e):
    print >>fd, e, t

