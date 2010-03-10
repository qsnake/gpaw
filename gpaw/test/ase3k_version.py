from gpaw.version import ase_required_svnversion
try:
    from ase.svnversion import svnversion as ase_svnversion
except ImportError:
    ase_svnversion = 'unknown'
if ase_svnversion == 'unknown':
    pass
else:
    full_ase_svnversion = ase_svnversion
    if ase_svnversion[-1] == 'M':
        ase_svnversion = ase_svnversion[:-1]
    if ase_svnversion.rfind(':') != -1:
        ase_svnversion = ase_svnversion[:ase_svnversion.rfind(':')]
    print "Required ase svnversion: "+ase_required_svnversion,
    print "; Current ase svnversion: "+full_ase_svnversion
    assert int(ase_svnversion) >= int(ase_required_svnversion), int(ase_svnversion)
