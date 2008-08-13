import numpy as npy
from ase.transport.calculators import TransportCalculator


H_lead = npy.zeros((4,4))
#onsite energies
for i in range(4):
    H_lead[i,i] = 0.0
#nearest neighbor hopping is -1.0
for i in range(3):
    H_lead[i,i+1] = -1.0
    H_lead[i+1,i] = -1.0
#second nearest neighbor hopping is 0.2
for i in range(2):
    H_lead[i,i+2] = 0.2
    H_lead[i+2,i] = 0.2


H_scat = npy.zeros((6,6))
#Principal layers on either side of S
H_scat[:2,:2] = H_lead[:2,:2]
H_scat[-2:,-2:] = H_lead[:2,:2]
#Scatering region (hydrogen molecule)
H_scat[2,3] = -0.8 
H_scat[3,2] = -0.8
#coupling to the leads
H_scat[1,2] = 0.2
H_scat[2,1] = 0.2
H_scat[3,4] = 0.2
H_scat[4,3] = 0.2


tcalc = TransportCalculator(h=H_scat,  #Scattering Hamiltonian
                            h1=H_lead, #Lead 1 (left)
                            h2=H_lead, #Lead 2  (right)
                            pl=2)      #principal layer size

tcalc.set(energies=npy.arange(-3,3,0.02))
T_e = tcalc.get_transmission()

tcalc.set(pdos=[0,1])
pdos_ne = tcalc.get_pdos()

h_rot, s_rot, eps_n, vec_nn = tcalc.subdiagonalize_bfs([0,1])
tcalc.set(h=h_rot,s=s_rot)#Set the rotated matrices
for n in range(2):
    print "eigenvalue, eigenvector:", eps_n[n],',', vec_nn[:,n]

pdos_rot_ne = tcalc.get_pdos()

h_cut, s_cut = tcalc.cutcoupling_bfs([0])
tcalc.set(h=h_cut,s=s_cut)
T_cut_bonding_e = tcalc.get_transmission()


#dump the date to txt files.
fd = file('T.dat','w')
for e,t in zip(tcalc.energies, T_e):
    print >> fd, e, t
fd.close()

fd = file('T_cut_bonding.dat','w')
for e,t in zip(tcalc.energies, T_cut_bonding_e):
    print >> fd, e, t
fd.close()

for n in range(2):
    fd = file('pdos%i.dat' % n,'w')
    for e,p in zip(tcalc.energies,pdos_ne[n]):
        print >> fd, e, p
fd.close()

for n in range(2):
    fd = file('pdos_rot%i.dat' % n,'w')
    for e,p in zip(tcalc.energies,pdos_rot_ne[n]):
        print >> fd, e, p
fd.close()


