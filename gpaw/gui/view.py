#!/usr/bin/env python

# Emacs: treat this as -*- python -*-

import os
import gtk
import tempfile
from math import cos, sin, sqrt

import numpy as npy
from ASE.ChemicalElements.cpk_color import cpk_colors

from gpaw.gui.repeat import Repeat
from gpaw.gui.rotate import rotate


class View:
    def __init__(self, vbox, rotations):
        self.colors = [None] * (len(cpk_colors) + 1)
        self.nselected = 0
        self.rotation = rotate(rotations)
        
        self.drawing_area = gtk.DrawingArea()
        self.drawing_area.set_size_request(350, 350)
        self.drawing_area.connect('button_press_event', self.press)
        self.drawing_area.connect('button_release_event', self.release)
        self.drawing_area.connect('motion-notify-event', self.move)
        # Signals used to handle backing pixmap:
        self.drawing_area.connect("expose_event", self.expose_event)
        self.drawing_area.connect("configure_event", self.configure_event)
        self.drawing_area.set_events(gtk.gdk.BUTTON_PRESS_MASK |
                                     gtk.gdk.BUTTON_RELEASE_MASK |
                                     gtk.gdk.BUTTON_MOTION_MASK |
                                     gtk.gdk.POINTER_MOTION_HINT_MASK)
        vbox.pack_start(self.drawing_area)
        self.drawing_area.show()
        self.configured = False
        
    def set_coordinates(self, frame=None, focus=None):
        if self.ui.get_widget("/MenuBar/ViewMenu/ShowUnitCell").get_active():
            R1, R2 = self.plot_cell()
        else:
            R1 = R2 = npy.empty((0, 3))
        n = self.atoms.natoms
        self.X = npy.empty((n + len(R1), 3))
        self.X[n:] = R1
        self.R1 = self.X[n:]
        self.R2 = R2
        self.set_frame(frame, focus=focus)

    def set_frame(self, frame=None, focus=False):
        if frame is None:
            frame = self.frame
        self.frame = frame
        n = self.atoms.natoms
        self.X[:n] = self.atoms.RR[frame]
        self.R = self.X[:n]
        if focus:
            self.focus()
        else:
            self.draw()
        
    def set_colors(self):
        new = self.drawing_area.window.new_gc
        alloc = self.colormap.alloc_color
        for z in self.atoms.Z:
            if self.colors[z] is None:
                c, p, k = cpk_colors[z]
                self.colors[z] = new(alloc(int(65535 * c),
                                           int(65535 * p),
                                           int(65535 * k)))
            
    def plot_cell(self):
        V = self.atoms.cell
        R1 = []
        R2 = []
        for c in range(3):
            v = V[c]
            d = sqrt(npy.dot(v, v))
            n = max(2, int(d / 0.3))
            h = v / (2 * n - 1)
            R = npy.arange(n)[:, None] * (2 * h)
            for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                R1.append(R + i * V[(c + 1) % 3] + j * V[(c + 2) % 3])
                R2.append(R1[-1] + h)
        return npy.concatenate(R1), npy.concatenate(R2)

    def toggle_show_unit_cell(self, action):
        self.set_coordinates()
        
    def toggle_show_axes(self, action):
        self.draw()

    def repeat_window(self, menuitem):
        Repeat(self)
        
    def focus(self, x=None):
        if (self.atoms.natoms == 0 and not
            self.ui.get_widget("/MenuBar/ViewMenu/ShowUnitCell").get_active()):
            self.scale = 1.0
            self.offset = npy.zeros(3)
            self.draw()
            return
        
        P = npy.dot(self.X, self.rotation)[:, :2]
        n = self.atoms.natoms
        P[:n] -= self.atoms.r[:, None]
        P1 = P.min(0) 
        P[:n] += 2 * self.atoms.r[:, None]
        P2 = P.max(0)
        C = (P1 + P2) / 2
        S = 1.3 * (P2 - P1)
        if S[0] * self.height < S[1] * self.width:
            self.scale = self.height / S[1]
        else:
            self.scale = self.width / S[0]
        self.offset = npy.array([ self.scale * C[0] - self.width / 2,
                                  self.scale * C[1] - self.height / 2,
                                  0.0])
        self.draw()

    def draw(self, status=True):
        self.pixmap.draw_rectangle(self.black_gc, True, 0, 0,
                                   self.width, self.height)
        X = npy.dot(self.X, self.scale * self.rotation) - self.offset
        n = self.atoms.natoms
        if n > 0:
            self.center = sum(X[:n]) / n
        else:
            self.center = npy.array([self.width / 2, self.height / 2, 0.0])
        self.indices = X[:, 2].argsort()
        P = self.P = X[:n, :2]
        X1 = X[n:, :2].round().astype(int)
        X2 = (npy.dot(self.R2, self.scale * self.rotation) -
              self.offset).round().astype(int)
        r = self.atoms.r * self.scale
        A = (P - r[:, None]).round().astype(int)
        d = (2 * r).round().astype(int)
        selected_gc = self.selected_gc
        colors = self.colors
        Z = self.atoms.Z
        arc = self.pixmap.draw_arc
        line = self.pixmap.draw_line
        black_gc = self.black_gc
        dynamic = self.atoms.dynamic
        selected = self.atoms.selected
        for a in self.indices:
            if a < n:
                ra = d[a]
                arc(colors[Z[a]], True, A[a, 0], A[a, 1], ra, ra, 0, 23040)
                if not dynamic[a]:
                    R1 = int(0.14644 * ra)
                    R2 = int(0.85355 * ra)
                    line(black_gc,
                         A[a, 0] + R1, A[a, 1] + R1,
                         A[a, 0] + R2, A[a, 1] + R2)
                    line(black_gc,
                         A[a, 0] + R2, A[a, 1] + R1,
                         A[a, 0] + R1, A[a, 1] + R2)
                if selected[a]:
                    arc(selected_gc, False, A[a, 0], A[a, 1], ra, ra, 0,23040)
                else:
                    arc(black_gc, False, A[a, 0], A[a, 1], ra, ra, 0,23040)
            else:
                a -= n
                line(self.white_gc, X1[a, 0], X1[a, 1], X2[a, 0], X2[a, 1])

        if self.ui.get_widget('/MenuBar/ViewMenu/ShowAxes').get_active():
            self.draw_axes()

        if self.atoms.nframes > 1:
            self.draw_frame_number()
            
        self.drawing_area.window.draw_drawable(self.black_gc, self.pixmap,
                                               0, 0, 0, 0,
                                               self.width, self.height)

        if status:
            self.status()

    def draw_axes(self):
        L = npy.zeros((10, 2, 3))
        L[:3, 1] = self.rotation * 15
        L[3:5] = self.rotation[0] * 20
        L[5:7] = self.rotation[1] * 20
        L[7:] = self.rotation[2] * 20
        L[3:] += (((-4, 5,0), (4,-5,0)), ((-4,-5,0), ( 4, 5,0)),
                  ((-4,-5,0), (0, 0,0)), ((-4, 5,0), ( 4,-5,0)),
                  ((-4,-5,0), (4,-5,0)), (( 4,-5,0), (-4, 5,0)),
                  ((-4, 5,0), (4, 5,0)))
        L = (L + (20, self.height - 20, 0)).round().astype(int)
        line = self.pixmap.draw_line
        colors = ([self.white_gc] * 3 +
                  [self.red] * 2 + [self.green] * 2 + [self.blue] * 3)
        for i in L[:,1,2].argsort():
            (a,b),(c,d) = L[i, :, :2]
            line(colors[i], a,b,c,d)

    digits = npy.array(((1,1,1,1,1,1,0),
                        (0,1,1,0,0,0,0),
                        (1,0,1,1,0,1,1),
                        (1,1,1,1,0,0,1),
                        (0,1,1,0,1,0,1),
                        (1,1,0,1,1,0,1),
                        (1,1,0,1,1,1,1),
                        (0,1,1,1,0,0,0),
                        (1,1,1,1,1,1,1),
                        (0,1,1,1,1,0,1)), bool)

    bars = npy.array(((0,2,1,2),
                      (1,2,1,1),
                      (1,1,1,0),
                      (1,0,0,0),
                      (0,0,0,1),
                      (0,1,0,2),
                      (0,1,1,1))) * 5
    
    def draw_frame_number(self):
        n = str(self.frame)
        x = self.width - 3 - 8 * len(n)
        y = self.height - 27
        color = self.white_gc
        line = self.pixmap.draw_line
        for c in n:
            bars = View.bars[View.digits[int(c)]]
            for a, b, c, d in bars:
                line(color, a + x, b + y, c + x, d + y)
            x += 8
        
    def release(self, drawing_area, event):
        if event.button != 1:
            return

        selected = self.atoms.selected
        
        if event.time < self.t0 + 200:  # 200 ms
            d = self.P - self.C
            hit = npy.less((d**2).sum(1), (self.scale * self.atoms.r)**2)
            for a in self.indices[::-1]:
                if a < self.atoms.natoms and hit[a]:
                    if event.state & gtk.gdk.CONTROL_MASK:
                        selected[a] = not selected[a]
                    else:
                        selected[:] = False
                        selected[a] = True
                    break
            else:
                selected[:] = False
            self.draw()
        else:
            A = (event.x, event.y)
            C1 = npy.minimum(A, self.C)
            C2 = npy.maximum(A, self.C)
            hit = npy.logical_and(self.P > C1, self.P < C2)
            indices = npy.compress(hit.prod(1), npy.arange(len(hit)))
            if not (event.state & gtk.gdk.CONTROL_MASK):
                selected[:] = False
            selected[indices] = True
            self.draw()

    def press(self, drawing_area, event):
        self.button = event.button
        self.C = npy.array((event.x, event.y))
        self.t0 = event.time
        self.rotation0 = self.rotation
        self.offset0 = self.offset
        self.center0 = self.center
        
    def move(self, drawing_area, event):
        x, y, state = event.window.get_pointer()
        C = self.C
        if self.button == 1:
            window = self.drawing_area.window
            window.draw_drawable(self.black_gc, self.pixmap,
                                 0, 0, 0, 0,
                                 self.width, self.height)
            x0, y0 = C.round().astype(int)
            window.draw_rectangle(self.selected_gc, False,
                                  min(x, x0), min(y, y0),
                                  abs(x - x0), abs(y - y0))
            return
        if self.button == 2:
            return
        if state & gtk.gdk.SHIFT_MASK:
            self.offset = self.offset0 - (x - C[0], y - C[1], 0)
        else:
            # Snap mode: the a-b angle and t should multipla of 15 degrees ???
            a = x - C[0]
            b = y - C[1]
            t = sqrt(a * a + b * b)
            if t > 0:
                a /= t
                b /= t
            else:
                a = 1.0
                b = 0.0
            c = cos(0.01 * t)
            s = -sin(0.01 * t)
            rotation = npy.array([(c * a * a + b * b, (c - 1) * b * a, s * a),
                                  ((c - 1) * a * b, c * b * b + a * a, s * b),
                                  (-s * a, -s * b, c)])
            self.rotation = npy.dot(self.rotation0, rotation)
            self.offset = npy.dot(self.center0 + self.offset0,
                                  rotation) - self.center0

        self.draw(status=False)
        
    # Create a new backing pixmap of the appropriate size
    def configure_event(self, drawing_area, event):
        if self.configured:
            w = self.width
            h = self.height
        else:
            self.colormap = self.drawing_area.get_colormap()
            self.black_gc = self.drawing_area.get_style().black_gc
            self.white_gc = self.drawing_area.get_style().white_gc
            self.red = self.drawing_area.window.new_gc(
                self.colormap.alloc_color(62345, 0, 0), line_width=2)
            self.green = self.drawing_area.window.new_gc(
                self.colormap.alloc_color(0, 54456, 0), line_width=2)
            self.blue = self.drawing_area.window.new_gc(
                self.colormap.alloc_color(0, 0, 54456), line_width=2)
            self.selected_gc = self.drawing_area.window.new_gc(
                self.colormap.alloc_color(62345, 54456, 2234),
                line_width=3)
            
        x, y, self.width, self.height = drawing_area.get_allocation()
        self.pixmap = gtk.gdk.Pixmap(drawing_area.window,
                                     self.width, self.height)
        if self.configured:
            self.scale *= sqrt(1.0 * self.width * self.height / (w * h))
            self.offset[0] += (w - self.width) / 2.0
            self.offset[1] += (h - self.height) / 2.0
            self.draw()
        self.configured = True
        
    # Redraw the screen from the backing pixmap
    def expose_event(self, drawing_area, event):
        x , y, width, height = event.area
        gc = self.black_gc
        drawing_area.window.draw_drawable(gc, self.pixmap,
                                          x, y, x, y, width, height)

    def external_viewer(self, action):
        name = action.get_name()
        if name == 'XMakeMol':
            from gpaw.gui.write import write_to_file
            fd, filename = tempfile.mkstemp('.xyz', 'g2-')
            os.close(fd)
            write_to_file(filename, self.atoms, 'xyz',
                          range(self.atoms.nframes))
            os.system('(xmakemol -f %s &); (sleep 5; rm %s) &' %
                      (filename, filename))

        elif name == 'RasMol':
            from gpaw.gui.write import write_to_file
            fd, filename = tempfile.mkstemp('.xyz', 'g2-')
            os.close(fd)
            write_to_file(filename, self.atoms, 'xyz', [self.frame])
            os.system('(rasmol -xyz %s &); (sleep 5; rm %s) &' %
                      (filename, filename))

        elif name == 'VMD':
            from gpaw.gui.write import write_to_file
            fd, filename = tempfile.mkstemp('.xyz', 'g2-')
            os.close(fd)
            write_to_file(filename, self.atoms, 'xyz', [self.frame])
            os.system('(vmd %s &); (sleep 5; rm %s) &' % (filename, filename))
