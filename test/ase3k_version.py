try:
    from ase.svnrevision import svnrevision as ase_svnrevision
except ImportError:
    ase_svnrevision = 'unknown'
print "ase svnrevision: "+ase_svnrevision
if ase_svnrevision == 'unknown':
    pass
else:
    if ase_svnrevision[-1] == 'M':
        ase_svnrevision = ase_svnrevision[:-1]
    if ase_svnrevision.rfind(':') != -1:
        ase_svnrevision = ase_svnrevision[:ase_svnrevision.rfind(':')]
    ase_svnrevision = int(ase_svnrevision)
    assert int(ase_svnrevision) >= 845, ase_svnrevision
