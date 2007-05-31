# -*- coding: utf-8 -*-

from math import sqrt, pi, acos

import gtk
import numpy as npy
from ASE.ChemicalElements.symbol import symbols
from ASE.ChemicalElements.name import names

from gpaw.gui.widgets import pack
from gpaw.gui.languages import translate as _

def formula(Z):
    hist = {}
    for z in Z:
        if z in hist:
            hist[z] += 1
        else:
            hist[z] = 1
    text = ''
    Z = hist.keys()
    Z.sort()
    for z in Z:
        text += symbols[z]
        n = hist[z]
        if n > 1:
            text += '<sub>%d</sub>' % n
    return text

class Status:
    def __init__(self, vbox):
        self.eventbox = gtk.EventBox()
        self.label = gtk.Label()
        self.eventbox.add(self.label)
        self.label.show()
        self.set_tip(self.eventbox, _('Tip for status box ...'))
        pack(vbox, self.eventbox)

    def status(self):
        # use where here:  XXX
        indices = npy.arange(self.atoms.natoms)[self.atoms.selected]
        n = len(indices)
        self.nselected = n
        
        if n == 0:
            self.label.set_text('')
            return

        Z = self.atoms.Z[indices]
        R = self.R[indices]

        if n == 1:
            text = (u' #%d %s (%s): %.3f Å, %.3f Å, %.3f Å' %
                    ((indices[0], names[Z[0]], symbols[Z[0]]) + tuple(R[0])))
        elif n == 2:
            D = R[0] - R[1]
            d = sqrt(npy.dot(D, D))
            text = u' %s-%s: %.3f Å' % (symbols[Z[0]], symbols[Z[1]], d)
        elif n == 3:
            d = []
            for c in range(3):
                D = R[c] - R[(c + 1) % 3]
                d.append(npy.dot(D, D))
            a = []
            for c in range(3):
                t1 = 0.5 * (d[c] + d[(c + 1) % 3] - d[(c + 2) % 3])
                t2 = sqrt(d[c] * d[(c + 1) % 3])
                try:
                    t3 = acos(t1 / t2)
                except ValueError:
                    if t1 > 0:
                        t3 = 0
                    else:
                        t3 = pi
                a.append(t3 * 180 / pi)
            text = (u' %s-%s-%s: %.1f°, %.1f°, %.1f°' %
                    tuple([symbols[z] for z in Z] + a))
        else:
            text = ' ' + formula(Z)
            
        self.label.set_markup(text)
