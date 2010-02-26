name1 = 'h2o_gs'
name2 = 'h2o_exc'

calc1 = GPAW(h=0.2,
             txt='h2o_gs.txt',
             xc='PBE')
atoms.set_calculator(calc1)
e = atoms.get_potential_energy()

Eref1 = calc1.Eref * Hartree
Etot1 = calc1.Etot * Hartree
E1 = Eref1 + Etot1

calc2 = GPAW(h=0.2,
             txt='h2o_exc.txt',
             xc='PBE',
             charge=-1,
             spinpol=True,
             setups={0:'fch1s'})

atoms.set_calculator(calc2)

magmom = [1, 0, 0]
atoms.set_magnetic_moments(magmom)

e = atoms.get_potential_energy()

Eref2 = calc2.Eref * Hartree
Etot2 = calc2.Etot * Hartree
E2 = Eref2 + Etot2

print 'Energy difference' , E2 - E1
