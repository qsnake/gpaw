
class ChangeNotifier(dict):
    def __init__(self, name, keys=None):
        if keys is None:
            keys = [] # there is a reason for this...

        self.name = name

        hashes = []
        values = []
        for event in keys:
            if isinstance(event, ChangeNotifier):
                hashes.append(event.name)
                values.append(event)
            else:
                hashes.append(event)
                values.append(False)

        dict.__init__(self, zip(hashes,values))

    def __call__(self, event):
        for key, value in self.items():
            if isinstance(value, ChangeNotifier):
                assert key != event
                value(event)
            elif key == event:
                assert type(value) is bool
                self[key] = True

    def __getattr__(self, key):

        if key == 'name':
            return object.__getattr__(self, key)

        return self[key]

    def __setattr__(self, key, value):

        if key == 'name':
            return object.__setattr__(self, key, value)

        assert key in self

        if isinstance(self[key], ChangeNotifier):
            raise NotImplementedError
        else:
            assert type(value) is bool
            self[key] = value

    def __nonzero__(self):
        for key, value in self.items():
            if value:
                return True

        return False

    def __repr__(self):
        txt = ','.join(map(str, self))
        return '<%s:%s(%s)>' % (self.__class__.__name__, self.name, txt)

    def extend(self, keys): #TODO allow ChangeNotifier in keys?
        values = [False]*len(keys)
        dict.update(self, zip(keys, values))

class EnergyNotifier(ChangeNotifier):
    def __init__(self, keys=None):
        ChangeNotifier.__init__(self, 'energy', keys)
        self.extend(['lmax', 'width', 'stencils', 'external', 'xc'])

class GridNotifier(ChangeNotifier):
    def __init__(self, keys=None):
        ChangeNotifier.__init__(self, 'grid', keys)
        self.extend(['h', 'gpts', 'setups', 'spinpol', 'usesymm', 'parsize', \
                     'parsize_bands', 'communicator'])

def InputNotifier():
    n_energy = EnergyNotifier()
    n_grid = GridNotifier()
    n_ham = ChangeNotifier('hamiltonian', ['charge', n_energy, n_grid])
    n_wfs = ChangeNotifier('wfs', ['kpts', 'nbands', 'mode', n_grid])
    n_occ = ChangeNotifier('occupations', ['kpts', 'nbands', n_energy, n_grid])
    n_dens = ChangeNotifier('density', ['charge', n_grid])
    n_all = ChangeNotifier('paw', [n_grid, n_ham, n_wfs, n_occ, n_dens])
    return n_all



