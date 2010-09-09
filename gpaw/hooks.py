import os
import random


class NotConverged:
    def __init__(self, dir='.'):
        self.dir = dir

    def __call__(self, calc):
        if calc.wfs.world.rank > 0:
            return

        from ase.io import write
        name = os.path.join(self.dir, ''.join(random.sample('gpaw' * 3, 12)))

        write(name + '.traj', calc.atoms.copy())

        fd = open(name + '.gkw', 'w')
        fd.write('%r\n' % dict(calc.input_parameters))
        fd.close()

        fd = open(name + '.txt', 'w')
        txt = calc.txt
        calc.txt = fd
        calc.print_logo()
        calc.print_cell_and_parameters()
        calc.print_positions()
        fd.close()
        calc.txt = txt

        os.chmod(name + '.traj', 0666)
        os.chmod(name + '.gkw', 0666)
        os.chmod(name + '.txt', 0666)


hooks = {}  # dictionary for callback functions

command = os.environ.get('GPAWSTARTUP')
if command is not None:
    exec(command)
home = os.environ.get('HOME')
if home is not None:
    rc = os.path.join(home, '.gpaw', 'rc.py')
    if os.path.isfile(rc):
        # Read file in ~/.gpaw/rc.py
        execfile(rc)

# Fill in allowed hooks:
locs = locals()
for name in ['converged', 'not_converged']:
    if name in locs:
        hooks[name] = locs[name]

# Backwards compatiblity:
if 'crashed' in locs and 'not_converged' not in hooks:
    hooks['not_converged'] = locs['crashed']
