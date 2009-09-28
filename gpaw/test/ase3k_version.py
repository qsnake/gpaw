from gpaw.version import ase_required_svnrevision
try:
    from ase.svnrevision import svnrevision as ase_svnrevision
except ImportError:
    ase_svnrevision = 'unknown'
if ase_svnrevision == 'unknown':
    pass
else:
    full_ase_svnrevision = ase_svnrevision
    if ase_svnrevision[-1] == 'M':
        ase_svnrevision = ase_svnrevision[:-1]
    if ase_svnrevision.rfind(':') != -1:
        ase_svnrevision = ase_svnrevision[:ase_svnrevision.rfind(':')]
    print "Required ase svnrevision: "+ase_required_svnrevision,
    print "; Current ase svnrevision: "+full_ase_svnrevision
    assert int(ase_svnrevision) >= int(ase_required_svnrevision), int(ase_svnrevision)
