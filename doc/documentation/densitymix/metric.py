# creates: metric.png

import pylab as pl
from math import pi, cos

# Special points in the BZ of a simple cubic cell
G = pi * pl.array([0., 0., 0.])
R = pi * pl.array([1., 1., 1.])
X = pi * pl.array([1., 0., 0.])
M = pi * pl.array([1., 1., 0.])

# The path for the band plot
path = [X, G, R, X, M, G]
textpath = [r'$X$', r'$\Gamma$', r'$R$', r'$X$', r'$M$', r'$\Gamma$']

# Make band data
qvec = []
lines = [0]
previous = path[0]
for next in path[1:]:
    Npoints = int(round(20 * pl.linalg.norm(next - previous)))
    lines.append(lines[-1] + Npoints)
    for t in pl.linspace(0, 1, Npoints):
        qvec.append((1 - t) * previous + t * next)
    previous = next

vasp = [1 / max(pl.linalg.norm(q), 1e-6)**2 for q in qvec]
gpaw = [( 1 + cos(qx) + cos(qy) + cos(qz) +
        cos(qx) * cos(qy) + cos(qx) * cos(qz) + cos(qy) * cos(qz) +
        cos(qx) * cos(qy) * cos(qz)) / 8. for qx, qy, qz in qvec]

# Plot band data
fig = pl.figure(1, figsize=(5, 3), dpi=90)
fig.subplots_adjust(left=.1, right=.95)
lim = [0, lines[-1], 0, 1.25]
pl.plot(vasp, 'k:', label='VASP')
pl.plot(gpaw, 'k-', label='GPAW')
for q in lines:
    pl.plot([q, q], lim[2:], 'k-')
pl.xticks(lines, textpath)
pl.yticks([0, 1], [r'$1$', r'$w+1$'])
pl.axis(lim)
pl.legend(loc='upper right', pad=.1, axespad=.06)
pl.title('Special metric for density changes')
pl.savefig('metric.png', dpi=90)
#pl.show()
