from gpaw import *
from gpaw.mpi import world
from ase import *
from ase.parallel import rank, size

initial = read('initial.traj')
final = read('final.traj')

constraint = FixAtoms(mask=[atom.tag > 1 for atom in initial])

images = [initial]
j = 1 + rank * 3 // size  # my image number
for i in range(1, 4):
    image = initial.copy()
    comm = world.new_communicator(np.array([rank]))
    if i == j:
        calc = GPAW(h=0.3, kpts=(2, 2, 1),
                    txt='neb%d.txt' % j,
                    communicator=comm)
        image.set_calculator(calc)
    image.set_constraint(constraint)
    images.append(image)
images.append(final)

neb = NEB(images, parallel=True)
neb.interpolate()
qn = QuasiNewton(neb, logfile='qn.log')
if rank % (size // 3) == 0:
    traj = PickleTrajectory('neb%d.traj' % j, 'w', images[j], master=True)
    qn.attach(traj)
qn.run(fmax=0.05)
