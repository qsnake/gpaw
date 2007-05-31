#!/usr/bin/env python
import __future__
import gtk

import numpy as npy

from gpaw.gui.languages import translate as _
from gpaw.gui.widgets import pack


class Execute(gtk.Window):
    def __init__(self, gui):
        gtk.Window.__init__(self)
        #self.window.set_position(gtk.WIN_POS_CENTER)
        #self.window.connect("destroy", lambda w: gtk.main_quit())
        #self.window.connect('delete_event', self.exit)
        self.set_title('Execute')
        vbox = gtk.VBox()
        self.cmd1, self.cmd2 = pack(vbox, [gtk.Entry(23),gtk.Entry(23)])
        self.cmd1.connect('activate', self.execute)
        self.cmd2.connect('activate', self.execute)
        self.selected = gtk.CheckButton('Only selected atoms')
        pack(vbox, self.selected)
        self.add(vbox)
        vbox.show()
        self.show()
        self.gui = gui

    def execute(self, widget=None):
        cmd1 = self.cmd1.get_text()
        cmd2 = self.cmd2.get_text()
        if cmd1:
            code1 = compile(cmd1 + '\n', 'execute.py', 'single',
                            __future__.CO_FUTURE_DIVISION)
        if cmd2:
            code2 = compile(cmd2 + '\n', 'execute.py', 'single',
                            __future__.CO_FUTURE_DIVISION)

        gui = self.gui

        N = gui.nframes
        n = gui.natoms
        A = gui.cell
        S = gui.selected
        if self.selected.get_active():
            indices = npy.where(S)[0]
        else:
            indices = range(n)

        for i in range(N):
            R = gui.RR[i]
            #F =
            #fmax = sqrt(max((F**2).sum(1))

            if cmd1:
                exec code1

            if not cmd2:
                continue
            
            for a in indices:
                x, y, z = R[a]
                s = S[a]
                Z = gui.Z[a]
                #f =
                exec code2
                S[a] = s
                R[a] = x, y, z

        gui.set_frame(gui.frame)
        
    python = execute
