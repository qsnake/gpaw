#!/usr/bin/env python
import gtk
from math import sqrt

import numpy as npy

from gpaw.gui.languages import translate as _
from gpaw.gui.widgets import pack, Help


class Repeat(gtk.Window):
    def __init__(self, gui):
        gtk.Window.__init__(self)
        self.set_title('Repeat')
        vbox = gtk.VBox()
        pack(vbox, gtk.Label(_('Repeat atoms:')))
        self.repeat = [gtk.Adjustment(r, 1, 9, 1) for r in gui.atoms.repeat]
        pack(vbox, [gtk.SpinButton(r, 0, 0) for r in self.repeat])
        for r in self.repeat:
            r.connect('value-changed', self.change)
        button = pack(vbox, gtk.Button('Set unit cell'))
        button.connect('clicked', self.set_unit_cell)
        self.add(vbox)
        vbox.show()
        self.show()
        self.gui = gui

    def change(self, adjustment):
        self.gui.atoms.repeat_atoms([int(r.value) for r in self.repeat])
        self.gui.set_coordinates()
        return True
        
    def set_unit_cell(self, button):
        self.gui.atoms.cell = npy.dot(npy.diag(self.gui.atoms.repeat),
                                      self.gui.atoms.cell)
        self.gui.atoms.EE *= self.gui.atoms.repeat.prod()
        self.gui.atoms.repeat = npy.ones(3, int)
        for r in self.repeat:
            r.value = 1
        self.gui.set_coordinates()
        
