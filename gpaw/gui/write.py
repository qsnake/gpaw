from math import sqrt

import numpy as npy
import Numeric as num
from ASE import ListOfAtoms, Atom
from ASE.ChemicalElements.symbol import symbols
from ASE.Units import units, Convert

from gpaw.gui.rotate import rotate


def write_to_file(filename, atoms, type, frames, gui=None,
                  rotations=None, show_unit_cell=None):
    Z = atoms.Z
    natoms = atoms.natoms
    RR = atoms.RR
    
    if type == 'xyz':
        f = open(filename, 'w')
        frame = 0
        for i in frames:
            f.write('%d\nFrame number %d\n' % (natoms, frame))
            R = RR[i]
            for a in range(natoms):
                f.write('%s %r %r %r\n' % (symbols[Z[a]],
                                           R[a, 0], R[a, 1], R[a, 2]))
            frame += 1

    elif type == 'traj':
        loa = ListOfAtoms([Atom(Z=z) for z in Z],
                          cell=atoms.cell.tolist(), periodic=True)
        from ASE.Trajectories.NetCDFTrajectory import NetCDFTrajectory
        traj = NetCDFTrajectory(filename, loa)
        if not npy.isnan(atoms.EE[0]):
            def energy():
                return float(atoms.EE[i])
            traj.Add('PotentialEnergy', energy)
        if not npy.isnan(atoms.FF[0, 0, 0]):
            def forces():
                return atoms.FF[i].tolist()
            traj.Add('CartesianForces', forces)
        for i in frames:
            loa.SetCartesianPositions(RR[i])
            traj.Update()

    elif type == 'py':
        A = atoms.cell
        lines = ['from ASE import ListOfAtoms, Atom', '']
        if len(frames) > 1:
            lines += ['loas = [', '  ListOfAtoms(']
        else:
            lines += ['loa = ListOfAtoms(']
        for i in frames:
            R = RR[i]
            for a in range(atoms.natoms):
                if a == 0:
                    format = "    [Atom('%s', [%r, %r, %r]),"
                elif a < atoms.natoms - 1: 
                    format = "     Atom('%s', [%r, %r, %r]),"
                else:
                    format = "     Atom('%s', [%r, %r, %r])],"
                lines += [format % (symbols[Z[a]], R[a, 0], R[a, 1], R[a, 2])]

            lines += ['    cell=[(%r, %r, %r),' % tuple(A[0]),
                      '          (%r, %r, %r),' % tuple(A[1]),
                      '          (%r, %r, %r)],' % tuple(A[2])]
            if len(frames) == 1:
                lines += ['    periodic=True)', '']
            else:
                if i < len(frames) - 1:
                    lines += ['    periodic=True),'
                              '  ListOfAtoms(']
                else:
                    lines += ['    periodic=True)]', '']

        open(filename, 'w').write('\n'.join(lines))
    else:
        from ASE.ChemicalElements.cpk_color import cpk_colors

        if gui is None:
            rotation = rotate(rotations)
        else:
            rotation = gui.rotation
            show_unit_cell = gui.ui.get_widget(
                "/MenuBar/ViewMenu/ShowUnitCell").get_active()
            
        N = 0
        D = npy.zeros((3, 3))

        if show_unit_cell:
            A = atoms.cell
            nn = []
            for c in range(3):
                d = sqrt((A[c]**2).sum())
                n = max(2, int(d / 0.3))
                nn.append(n)
                N += 4 * n

            X = npy.empty((N + natoms, 3))
            T = npy.empty(N, int)

            n1 = 0
            for c in range(3):
                n = nn[c]
                dd = A[c] / (4 * n - 2)
                D[c] = dd
                P = npy.arange(1, 4 * n + 1, 4)[:, None] * dd
                T[n1:] = c
                for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                    n2 = n1 + n
                    X[n1:n2] = P + i * A[(c + 1) % 3] + j * A[(c + 2) % 3]
                    n1 = n2
            assert n2 == N
        else:
            X = npy.empty((natoms, 3))


        X[N:] = RR[-1]
        r = atoms.r

        R = X[N:]
        r2 = r**2
        for n in range(N):
            d = D[T[n]]
            if ((((R - X[n] - d)**2).sum(1) < r2) &
                (((R - X[n] + d)**2).sum(1) < r2)).any():
                T[n] = -1

        X = npy.dot(X, rotation)

        if gui is None:
            if show_unit_cell == 2:
                P = X[:, :2].copy()
                M = N
            else:
                P = X[N:, :2].copy()
                M = 0
            P[M:] -= r[:, None]
            P1 = P.min(0) 
            P[M:] += 2 * r[:, None]
            P2 = P.max(0)
            C = (P1 + P2) / 2
            S = 1.05 * (P2 - P1)
            scale = 50.0
            w = scale * S[0]
            if w > 500:
                w = 500
                scale = w / S[0]
            h = scale * S[1]
            offset = npy.array([scale * C[0] - w / 2,
                                scale * C[1] - h / 2,
                                0.0])
        else:
            w, h = gui.width, gui.height
            scale = gui.scale
            offset = gui.offset

        if type == 'eps':
            import time
            from matplotlib.backends.backend_ps import RendererPS, \
                 GraphicsContextPS, psDefs

            f = open(filename, 'w')

            # write the PostScript headers
            print >> f, '%!PS-Adobe-3.0 EPSF-3.0'
            print >> f, '%%Creator: G2'
            print >> f, '%%CreationDate: ' + time.ctime(time.time())
            print >> f, '%%Orientation: portrait'
            bbox = (0, 0, w, h)
            print >> f, '%%%%BoundingBox: %d %d %d %d' % bbox
            print >> f, '%%EndComments'

            Ndict = len(psDefs)
            print >> f, '%%BeginProlog'
            print >> f, '/mpldict %d dict def' % Ndict
            print >> f, 'mpldict begin'
            for d in psDefs:
                d = d.strip()
                for l in d.split('\n'):
                    print >> f, l.strip()
            print >> f, '%%EndProlog'

            print >> f, 'mpldict begin'
            print >> f, '%d %d 0 0 clipbox' % (w, h)

            renderer = RendererPS(w, h, f)
            line = renderer.draw_line
            gc = GraphicsContextPS()
        else:
            from matplotlib.backends.backend_agg import RendererAgg
            from matplotlib.backend_bases import GraphicsContextBase
            from matplotlib.transforms import Value, identity_transform
            renderer = RendererAgg(w, h, Value(72))
            identity = identity_transform()
            def line(gc, x1, y1, x2, y2):
                renderer.draw_lines(gc,
                                    (round(x1), round(x2)),
                                    (round(y1), round(y2)), identity)
            gc = GraphicsContextBase()
            
        # Write the figure
        D = npy.dot(D, rotation)[:, :2] * scale
        X *= scale
        X -= offset
        X[:, 1] = h - X[:, 1]
        D[:, 1] = -D[:, 1]
        d = 2 * scale * r
        indices = X[:, 2].argsort()
        arc = renderer.draw_arc
        for a in indices:
            x, y = X[a, :2]
            if a < N:
                c = T[a]
                if c != -1:
                    hx, hy = D[c]
                    def ir(x):return int(round(x))
                    line(gc, x - hx, y - hy, x + hx, y + hy)
            else:
                a -= N
                da = d[a]
                arc(gc, cpk_colors[Z[a]], x, y, da, da, 0, 360, 0)

        if type == 'eps':
            # Write the trailer
            print >> f, 'end'
            print >> f, 'showpage'
            f.close()
        else:
            renderer._renderer.write_png(filename)

