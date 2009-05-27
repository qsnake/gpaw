"""Generate basis sets for LCAO calculations."""
import os
from gpaw.atom.basis import BasisMaker
for symbol in ['H', 'Pt']:
    BasisMaker(symbol, name='dzp').generate(2, 1).write_xml()
    os.system('mv %s.dzp.basis %s' % (symbol, os.environ['GPAW_SETUP_PATH']))

