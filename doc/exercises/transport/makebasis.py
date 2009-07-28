"""Generate basis sets for LCAO calculations."""
import os
from gpaw.atom.basis import BasisMaker
for symbol in ['H', 'Pt']:
    bm = BasisMaker(symbol, name='szp', non_relativistic_guess=True)
    bm.generate(1, 1).write_xml()
    os.system('mv %s.szp.basis %s' % (symbol, os.environ['GPAW_SETUP_PATH']))

