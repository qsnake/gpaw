import os
import pylab
import Numeric as num
from math import sqrt
from LinearAlgebra import solve_linear_equations as solve
from gpaw.testing.atomization_data import atomization_vasp
from gpaw.utilities import fix, fix2

from moleculetest import dd
from data import Ea12

def main(molecules, moleculedata, results, Ea):
    for formula, molecule in molecules.items():
        reference = moleculedata[formula]
        result = results[formula]
        if result['Em0'] is None:
            continue
        E0 = 0.0
        ok = True
        for atom in molecule:
            symbol = atom.GetChemicalSymbol()
            if Ea[symbol] is None:
                ok = False
                break
            E0 += Ea[symbol]
        if ok:
            result['Ea'] = E0 - result['Em0']
        if len(molecule) == 2:
            d = result['d0'] + dd
            M = num.zeros((4, 5), num.Float)
            for n in range(4):
                M[n] = d**-n
            a = solve(num.innerproduct(M, M), num.dot(M, result['Em'] - E0))

            dmin = 1 / ((-2 * a[2] +
                         sqrt(4 * a[2]**2 - 12 * a[1] * a[3])) / (6 * a[3]))
            #B = xmin**2 / 9 / vmin * (2 * a[2] + 6 * a[3] * xmin)

            dfit = num.arange(d[0] * 0.95, d[4] * 1.05, d[2] * 0.005)

            emin = a[0]
            efit = a[0]
            for n in range(1, 4):
                efit += a[n] * dfit**-n
                emin += a[n] * dmin**-n

            result['d'] = dmin
            result['Eamin'] = -emin
            
            pylab.plot(dfit, efit, '-', color='0.7')
            
            if ok:
                pylab.plot(d, result['Em'] - E0, 'g.')
            else:
                pylab.plot(d, result['Em'] - E0, 'ro')

            pylab.text(dfit[0], efit[0], fix(formula))

    pylab.xlabel(u'Bond length [Å]')
    pylab.ylabel('Energy [eV]')
    pylab.savefig('molecules.png')

    o = open('molecules.txt', 'w')
    print >> o, """\
.. contents::

==============
Molecule tests
==============

Atomization energies (*E*\ `a`:sub:) and bond lengths (*d*) for 20
small molecules calculated with the PBE functional.  All calculations
are done in a box of size 12.6 x 12.0 x 11.4 Å with a grid spacing of
*h*\ =0.16 Å and zero-boundary conditions.  Compensation charges are
expanded with correct multipole moments up to *l*\ `max`:sub:\ =2.
Open-shell atoms are treated as non-spherical with integer occupation
numbers, and zero-point energy is not included in the atomization
energies. The numbers are compared to very accurate, state-of-the-art,
PBE calculations (*ref* subscripts).

.. figure:: molecules.png
   

Bond lengths and atomization energies at relaxed geometries
===========================================================

(*rlx* subscript)

.. list-table::
   :widths: 2 3 8 5 6 8

   * -
     - *d* [Å]
     - *d*-*d*\ `ref`:sub: [Å]
     - *E*\ `a,rlx`:sub: [eV]
     - *E*\ `a,rlx`:sub:-*E*\ `a`:sub: [eV]
     - *E*\ `a,rlx`:sub:-*E*\ `a,rlx,ref`:sub: [eV] [1]_"""
    for formula, Ea1, Ea2 in Ea12:
        reference = moleculedata[formula]
        result = results[formula]
        if 'Eamin' in result:
            print >> o, '   * -', fix2(formula)
            print >> o, '     - %5.3f' % result['d']
            if 'dref' in reference:
                print >> o, ('     - ' +
                             ', '.join(['%+5.3f [%d]_' % (result['d'] - dref,
                                                          ref)
                                        for dref, ref in reference['dref']]))
            else:
                print >> o, '     -'
            print >> o, '     - %6.3f' % result['Eamin']
            if result.get('Ea') is not None:
                print >> o, '     - %6.3f' % (result['Eamin'] -
                                              result['Ea'])
            else:
                print >> o, '     - Unknown'
            if formula in atomization_vasp:
                print >> o, '     - %6.3f' % (result['Eamin'] -
                                              atomization_vasp[formula][1] /
                                              23.0605)
            else:
                print >> o, '     -'

    print >> o, """\

Atomization energies at experimental geometries
===============================================

.. list-table::
   :widths: 6 6 12

   * -
     - *E*\ `a`:sub: [eV]
     - *E*\ `a`:sub:-*E*\ `a,ref`:sub: [eV]"""
    for formula, Ea1, Ea2 in Ea12:
        reference = moleculedata[formula]
        result = results[formula]
        print >> o, '   * -', fix2(formula)
        if 'Ea' in result:
            print >> o, '     - %6.3f' % result['Ea']
            if 'Earef' in reference:
                print >> o, ('     - ' +
                             ', '.join(['%+5.3f [%d]_' % (result['Ea'] -
                                                          Ecref, ref)
                                        for Ecref, ref in reference['Earef']]))
            else:
                print >> o, '     -'
        else:
            print >> o, '     -'
            print >> o, '     -'
        
    print >> o, """

References
==========

.. [1] "The Perdew-Burke-Ernzerhof exchange-correlation functional
       applied to the G2-1 test set using a plane-wave basis set",
       J. Paier, R. Hirschl, M. Marsman and G. Kresse,
       J. Chem. Phys. 122, 234102 (2005)

.. [2] "Molecular and Solid State Tests of Density Functional
       Approximations: LSD, GGAs, and Meta-GGAs", S. Kurth,
       J. P. Perdew and P. Blaha, Int. J. Quant. Chem. 75, 889-909
       (1999)

.. [3] "Comment on 'Generalized Gradient Approximation Made Simple'",
       Y. Zhang and W. Yang, Phys. Rev. Lett.

.. [4] Reply to [3]_, J. P. Perdew, K. Burke and M. Ernzerhof

"""

    o.close()
    
    os.system('rst2html.py ' +
              '--no-footnote-backlinks ' +
              '--trim-footnote-reference-space ' +
              '--footnote-references=superscript molecules.txt molecules.html')
