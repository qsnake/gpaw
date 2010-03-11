# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

version_base = '0.7'

ase_required_version = '3.2.0'
ase_required_svnversion = '1158'

try:
    from gpaw.svnversion import svnversion
except (AttributeError, ImportError):
    version = version_base
else:
    version = version_base + '.' + svnversion
