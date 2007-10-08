from gpaw.atom.configurations import configurations

moleculedata = {}
"""This is a dictionary of formulas containing various reference data.
All the stuff initialized below is stored in this variable."""

def get_magnetic_moment(symbol):
    """Returns the magnetic moment of one atom of the specified element,
    using Hunds rules"""
    magmom = 0.
    for n, l, f, e in configurations[symbol][1]:
        magmom += min(f, 2 * (2 * l + 1) - f)
    return magmom

# The first column containing numbers seems to be identical to
# some pbe values from gpaw.testing.atomization_data, and the
# molecules are obviously taken from that hashtable.
# The final column, however, does not seem to appear anywhere else.
# These values are from "reference 1" and "reference 2", respectively
Ea12 = [('H2',   104.6, 104.5),
        ('LiH',   53.5,  53.5),
        ('CH4',  419.8, 419.2),
        ('NH3',  301.7, 301.0),
        ('OH',   109.8, 109.5),
        ('H2O',  234.2, 233.8),
        ('HF',   142.0, 141.7),
        ('Li2',   19.9,  19.7),
        ('LiF',  138.6, 139.5),
        ('Be2',    9.8,   9.5),
        ('C2H2', 414.9, 412.9),
        ('C2H4', 571.5, 570.2),
        ('HCN',  326.1, 324.5),
        ('CO',   268.8, 267.6),
        ('N2',   243.2, 241.2),
        ('NO',   171.9, 169.7),
        ('O2',   143.7, 141.7),
        ('F2',    53.4,  51.9),
        ('P2',   121.1, 117.2),
        ('Cl2',   65.1,  63.1)]

for formula, Ea1, Ea2 in Ea12:
    moleculedata[formula] = {'dref': []}
    moleculedata[formula]['Earef'] = [(Ea1 / 23.0605, 2), (Ea2 / 23.0605, 3)]

# These values are from "reference 1"
for formula, d in [('Cl2', 1.999),
                   ('CO',  1.136),
                   ('F2',  1.414),
                   ('Li2', 2.728),
                   ('LiF', 1.583),
                   ('LiH', 1.604),
                   ('N2',  1.103),
                   ('O2',  1.218)]:
    moleculedata[formula]['dref'].append((d, 1))

# These values are from "reference 4"
for formula, d in [('H2', 1.418),
                   ('N2', 2.084),
                   ('NO', 2.189),
                   ('O2', 2.306),
                   ('F2', 2.672)]:
    moleculedata[formula]['dref'].append((d * 0.529177, 4))


