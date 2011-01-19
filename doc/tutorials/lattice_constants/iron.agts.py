def agts(queue):
    iron = queue.add('iron.py', ncpus=8, walltime=8 * 60)
    queue.add('iron.agts.py', deps=[iron],
              creates=['Fe_conv_k.png', 'Fe_conv_h.png'])

if __name__ == '__main__':
    import numpy as np
    import pylab as plt
    from ase.utils.eos import EquationOfState
    from ase.io import read
    def f(width, k, g):
        filename = 'Fe-FD-%.2f-%02d-%2d.traj' % (width, k, g)
        configs = read(filename + '@::2')
        # Extract volumes and energies:
        volumes = [a.get_volume() for a in configs]
        energies = [a.get_potential_energy() for a in configs]
        eos = EquationOfState(volumes, energies)
        v0, e0, B = eos.fit()
        return v0, e0, B

    kk = [2, 4, 6, 8, 10, 12]

    plt.figure(figsize=(6, 4))
    for width in [0.05, 0.1, 0.15, 0.2]:
        a = []
        for k in kk:
            v0, e0, B = f(width, k, 12)
            a.append((2 * v0)**(1.0 / 3.0))
        print ('%7.3f ' * 7) % ((width,) + tuple(a))
        plt.plot(kk, a, label='width = %.2f eV' % width)
    plt.legend(loc='upper right')
    #plt.axis(ymin=2.83, ymax=2.85)
    plt.xlabel('number of k-points')
    plt.ylabel('lattice constant [Ang]')
    plt.savefig('Fe_conv_k.png')

    plt.figure(figsize=(6, 4))
    gg = np.arange(8, 32, 4)
    a = []
    for g in gg:
        v0, e0, B = f(0.1, 8, g)
        a.append((2 * v0)**(1.0 / 3.0))
    plt.plot(2.84 / gg, a, 'o-')
    #plt.axis(ymin=2.83, ymax=2.85)
    plt.xlabel('grid-spacing [Ang]')
    plt.ylabel('lattice constant [Ang]')
    plt.savefig('Fe_conv_h.png')
