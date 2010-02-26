name= 'diamond333_hch'

atoms = read_xyz(diamond.xyz)
atoms *= (3,3,3)

calc = Calculator( h=0.2, txt = name +'.txt',
                    xc='PBE', setups={0:'hch1s'})
atoms.set_calculator(calc)

e = atoms.get_potential_energy()

calc.write(name + '.gpw')
