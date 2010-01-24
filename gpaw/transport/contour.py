import numpy as np
from gpaw.mpi import world
class Path:
    ###
    poles_num = 5
    bias_step = 0.1
    step = 0.2
    bias_window_begin = -3
    bias_window_end = 3
    zone_sample = np.array([0, 0.55278640450004, 1.44721359549996, 2.0]) / 2.
    zone_length = np.array([0, 0.55278640450004, 0.89442719099992007,
                                                     0.55278640450004]) / 2.0
    sample = np.array([0, 0.18350341907227,   0.55278640450004,   1.0,
         1.44721359549996,   1.81649658092773, 2.0]) / 2.
    zone_weight = np.array([1.0, 1.0, 5.0, 5.0, 1.0]) / 6.0
    weights0 = np.array([1.0, 0.0, 5.0, 0.0, 5.0, 0.0, 1.0]) / 6.0
    weights1 = np.array([77.0, 432.0, 625.0, 672.0, 625.0,
                          432.0, 77.0]) / 1470.

    def __init__(self, begin, end, index, maxdepth=7, type='Gaussian'):
        self.begin = begin
        self.end = end
        self.type = type
        self.index = index
        self.maxdepth = maxdepth
        self.nids = []
        self.energies = []
        self.functions = []
        self.num = 10 ** self.maxdepth
        self.full_nodes = False
    
    def get_poles_index(self, real_energy):
        if self.type == 'poles':
            return int((real_energy - self.bias_window_begin) //
                                              self.bias_step)

    def get_flags(self, num, path_flag=False):
        flags = []
        if self.type == 'Gaussian':
            if path_flag:
                assert num < self.num * 10
            else:
                assert num < self.num                
            digits = self.maxdepth
            
        elif self.type == 'linear':
            nids_num = int(abs((self.end - self.begin)) / self.step) + 1
            digits = int(np.ceil(np.log10(nids_num)))            
            
        elif self.type == 'poles':
            if self.full_nodes:
                nids_num = int((self.bias_window_end -
                                self.bias_window_begin) //
                                  self.bias_step + 1)                
                digits = int(np.ceil(np.log10(nids_num))) 
            else:
                digits = 3

        if path_flag:
            digits += 1

        for i in range(digits):
            unit = 10 ** (digits - i)
            digit = (num - (num // unit) * unit) // (unit / 10)
            flags.append(digit)            
        return np.array(flags)
        
    def get_full_nids(self):
        self.full_nodes = True
        self.nids = []
        self.energies = []
        depth = self.maxdepth    
        if self.type == 'Gaussian':
            base = self.num * self.index
            assert depth < 7
            dimension = [3] * (depth - 1) + [6]
            for i in range(self.num):
                flags = self.get_flags(i)
                this_node = True
                for j in range(depth):
                    if flags[j] >= dimension[j]:
                        this_node = False
                    else:
                        flags[j] += 1
                if this_node:
                    new_i = np.sum(flags * 10 ** np.arange(len(flags) - 1,
                                                           -1, -1))
                    self.nids.append(new_i + base)
                    self.energies.append(self.get_energy(flags))
        
        elif self.type == 'linear':
            num = int((np.abs(self.end - self.begin)) // self.step) + 1
            vector = (self.end - self.begin) / (num - 1)
            digits = int(np.ceil(np.log10(num)))
            base = self.index * 10 ** digits
            for i in range(num):
                self.nids.append(i + base)
                self.energies.append(self.begin + i * self.step * vector)
        
        elif self.type == 'poles':
            num = int((self.bias_window_end - self.bias_window_begin) //
                                                      self.bias_step + 1)
            real_energies = np.linspace(self.bias_window_begin,
                                        self.bias_window_end, num)
            digits = int(np.ceil(np.log10(num * self.poles_num)))
            base = self.index * 10 ** digits
            for i in range(num):
                for k in range(self.poles_num):
                    self.nids.append(k + i * 10 + base)
                    self.energies.append(real_energies[i] +
                                                 (2 * k + 1) * np.pi * 1.j)
        else:
            raise RuntimeWarning('Wrong path type %s' % self.type)
            
    def get_new_nids(self, zone, depth=1):
        assert depth < 7
        nids = []
        if self.type == 'Gaussian':
            base = zone * 10 ** (self.maxdepth - depth)
            for i in range(2, 7):
                nids.append(base + i)
        elif self.type == 'linear':
            assert depth == 1
            nids_num = int((self.end - self.begin) / self.step) + 1
            digits = int(np.ceil(np.log10(nids_num)))
            base = self.index * 10 ** digits
            for i in range(nids_num):
                nids.append(i + base)
        elif self.type == 'poles':
            assert depth == 1
            base = zone * 100
            res_nids0 = np.arange(1, 9, 2) * np.pi * 1.j + self.begin
            res_nids1 = np.arange(1, 9, 2) * np.pi * 1.j + self.end
            nids0 = np.arange(len(res_nids0)) + base + 10 + 1
            nids1 = np.arange(len(res_nids1)) + base + 20 + 1
            nids = np.append(nids0, nids1).tolist()
        else:
            raise RuntimeError('Wrong Path Type % s' % self.type)
        self.nids += nids
        return nids
    
    def get_ends_nids(self):
        nids = []
        num = self.num / 10
        if self.index in [1, 2, 3, 6]:
            nids.append(self.index * num + 1)
        nids.append(3 * num + 7)
        nids.append(6 * num + 7)
        self.nids += nids
        return nids
       
    def get_energy(self, flags, level=0):
        if self.type == 'Gaussian':
            pps = np.append([1], self.zone_length[np.array(flags[:-1])])
            lls = []
            for i in range(self.maxdepth):
                lls.append(np.product(pps[:i+1]))
            ss = np.append(self.zone_sample[np.array(flags[:-1])- 1],
                           self.sample[flags[-1] - 1])
            energy = np.sum(ss * lls) * (self.end - self.begin) + self.begin
            
        elif self.type == 'linear':
            num = int(abs(self.end - self.begin)) / self.step 
            tens = np.arange(len(flags) - 1, -1, -1) ** 10
            energy =  np.sum(flags * tens) / num * (self.end - self.begin) + self.begin
       
        elif self.type == 'poles':
            if self.full_nodes:
                num = int((self.bias_window_end - self.bias_window_begin) //
                                                      self.bias_step + 1)
                real_energies = np.linspace(self.bias_window_begin,
                                        self.bias_window_end, num)                
                tens = np.arange(len(flags[:-1]) - 1, -1) ** 10
                line_index = np.sum(flags[:-1] * tens)
                energy = real_energies[line_index] + (2 * flags[-1] -
                                                       1) * np.pi * 1.j
            else:
                lines = [self.begin, self.end]
                energy = lines[flags[0] - 1] + (2 * flags[-1] - 1) * np.pi * 1.j
        return energy
    
    def get_weight(self, flags):
        if self.type == 'Gaussian':
            wei0 = 0
            for i in range(self.maxdepth - 2):
                wei0 += flags[i] * self.zone_weight[i]
            wei1 = wei0
            wei0 += flags[-1] * self.weights0[6]
            wei1 += flags[-1] * self.weights1[6]
            return wei0, wei1

class Contour:
    # see the file description of contour
    eq_err = 1e-3
    ne_err = 1e-4
    eta = 1e-4
    kt = 0.1
    nkt = 1.
    dkt = np.pi
    calcutype = ['eqInt', 'eqInt', 'eqInt', 'resInt', 'neInt', 'locInt']
    def __init__(self, kt, fermi, bias, maxdepth=7, comm=None, neint='linear',
                  tp=None):
        self.kt = kt
        self.fermis = fermi
        self.bias = bias
        self.tp = tp
        self.neint = neint
        self.min_bias = np.min(bias)
        self.max_bias = np.max(bias)
        self.comm = comm
        if self.comm == None:
            self.comm = world
        assert np.abs(self.kt - 0.1) < 1e-6
        self.maxdepth = maxdepth
        self.num = 10 ** (self.maxdepth - 1)
       
    def get_dense_contour(self):
        self.paths = []
        depth = self.maxdepth
        self.paths.append(Path(-700., -700. + 20. * 1.j, 1, depth))
        self.paths.append(Path(-700. + 20. * 1.j,  -4. + np.pi * 1.j, 2, depth))
        self.paths.append(Path(-4. + np.pi * 1.j, 4. + np.pi * 1.j, 3, depth,
                              type='linear'))
        self.paths.append(Path(-3., 3., 4, depth, type='poles'))
        self.paths.append(Path(-5. + self.eta * 1.j, 5. + self.eta * 1.j, 5,
                              depth, type='linear')) 
        for i in range(5):
            self.paths[i].get_full_nids()

    def get_optimized_contour(self):
        assert self.tp is not None
        self.paths = []
        depth = self.maxdepth
        self.paths.append(Path(-700., -700. + 20. * 1.j, 1, depth))
        self.paths.append(Path(-700. + 20. * 1.j,
                                self.min_bias - self.nkt + self.dkt * 1.j, 2, depth))
        self.paths.append(Path(self.min_bias - self.nkt + self.dkt * 1.j,
                              self.min_bias + self.nkt + self.dkt * 1.j, 3, depth))
        self.paths.append(Path(self.min_bias, self.max_bias, 4, depth, type='poles'))
        self.paths.append(Path(self.min_bias - self.nkt + self.eta * 1.j,
                              self.max_bias + self.nkt + self.eta * 1.j, 5,
                              depth, type=self.neint))
        self.paths.append(Path(self.min_bias - self.nkt + self.dkt * 1.j,
                              self.max_bias + self.nkt + self.dkt * 1.j, 6, depth))
        zones = np.arange(1, 7)
        depth = 1
        converge = False
        while not converge:
            nids = self.collect(zones, depth)
            loc_nids = self.distribute(nids)
            self.calculate(loc_nids)
            converge, zones = self.check_convergence(zones, depth)
            depth += 1
        
    def collect(self, zones, depth):
        nids = []
        for zone in zones:
            path_index = zone // (10 ** (depth - 1)) - 1
            nids += self.paths[path_index].get_new_nids(zone, depth)
        if depth == 1:
            nids += self.paths[path_index].get_ends_nids()
        return nids
    
    def distribute(self, nids):
        loc_nids = np.array_split(nids, self.comm.size)
        return loc_nids[self.comm.rank]
    
    def calculate(self, loc_nids):
        for nid in loc_nids:
            exp10 = int(np.floor(np.log10(nid)))
            path_index = nid // (10 ** exp10) - 1
            flags = self.paths[path_index].get_flags(nid)
            energy = self.paths[path_index].get_energy(flags[1:])
            if self.tp.recal_path or (not self.tp.recal_path and
                                      path_index in [0, 1, 2, 5]) :
                calcutype = self.calcutype[path_index]
                green_function = self.tp.calgfunc(energy, calcutype)
                self.paths[path_index].functions.append(green_function)
                self.paths[path_index].energies.append(energy)
    
    def get_begin(self, zone, depth):
        if depth == 1:
            return np.array([zone * self.num + 1])
        else:
            tens = zone // 10
            digits = zone % 10
            return np.array([tens * 10 ** (self.maxdepth + 1 - depth)
                              + digits * 2 - 1]) 
    
    def get_end(self, zone, depth):
        if depth == 1:
            if zone in [1, 2]:
                return (zone + 1) * self.num + 1
            elif zone in [3, 6]:
                return zone * self.num + 7
       
    def check_convergence(self, zones, depth):
        new_zones = []
        nbmol = self.tp.nbmol
        converged = True
        for zone in zones:
            base = zone * 10 ** (self.maxdepth - depth)
            begin = self.get_begin(zone, depth)
            nids = np.append(begin, np.arange(2, 7) + base)
            end = self.get_end(zone, depth)
            nids = np.append(nids, end)

            original_nids = np.arange(1, 8) + base
            path_index = zone // 10 ** (depth - 1) - 1
            gr_sum0 = np.zeros([nbmol, nbmol])
            gr_sum1 = np.zeros([nbmol, nbmol])
            path = self.paths[path_index]            
            for nid, orig_nid in zip(nids, original_nids):
                if nid in path.nids:
                    flags = path.get_flags(orig_nid)
                    weight0, weight1 = path.get_weight(flags)
                    fermi_factor = path.get_fermi_factor(nid)
                    index = path.nids.index(nid)
                    gr_sum0 += path.functions[index] * weight0 * fermi_factor 
                    gr_sum1 += path.functions[index] * weight1 * fermi_factor                     
            if path_index in [0, 1, 2, 5]:
                err = self.eq_err 
            else:
                err = self.ne_err
            if np.max(abs(gr_sum0 - gr_sum1)) > err:
                converged = False
                new_zones += range(zone * 10 + 1, zone * 10 + 4)
        return converged, new_zones        
        
        
    