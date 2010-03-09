# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

version = '0.7'

ase_required_version = '3.2.0'
ase_required_svnrevision = '1158'

def get_gpaw_svnversion_from_import():
    try:
        # try to import the last svn version number from gpaw/svnversion.py
        from gpaw.svnversion import svnversion
    except:
        svnversion = None
    ##
    return svnversion

svnversion = get_gpaw_svnversion_from_import()
if svnversion:
    version = version+'.'+svnversion
