#!/usr/bin/env python
import gtk
from math import sqrt

import numpy as npy

from gpaw.gui.languages import translate as _
from gpaw.gui.widgets import pack, Help


class Constraints(gtk.Window):
    def __init__(self, gui):
        gtk.Window.__init__(self)
        self.set_title(_('Constraints'))
        vbox = gtk.VBox()
        b = pack(vbox, [gtk.Button(_('Constrain')),
                        gtk.Label(_(' selected atoms'))])[0]
        b.connect('clicked', self.selected)
        b = pack(vbox, [gtk.Button(_('Constrain')),
                        gtk.Label(_(' immobile atoms:'))])[0]
        b.connect('clicked', self.immobile)
        b = pack(vbox, gtk.Button('Clear constraint'))
        b.connect('clicked', self.clear)
        close = pack(vbox, gtk.Button(_('Close')))
        close.connect('clicked', lambda widget: self.destroy())
        self.add(vbox)
        vbox.show()
        self.show()
        self.gui = gui

    def selected(self, button):
        self.gui.atoms.dynamic = ~self.gui.atoms.selected
        self.gui.draw()

    def immobile(self, button):
        self.gui.atoms.set_dynamic()
        self.gui.draw()

    def clear(self, button):
        self.gui.atoms.dynamic[:] = True
        self.gui.draw()

