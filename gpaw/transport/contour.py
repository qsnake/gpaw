import numpy as np
from gpaw.mpi import world
from gpaw.transport.tools import fermidistribution, gather_ndarray_dict

class Path:
    ###
    poles_num = 5
    bias_step = 0.1
    step = 0.2
    bias_window_begin = -3
    bias_window_end = 3
    zone_sample = np.array([0, 0.55278640450004, 1.44721359549996, 2.0, 0.0]) / 2.
    zone_length = np.array([2., 0.55278640450004, 0.89442719099992007,
                                                     0.55278640450004]) / 2.0
    sample = np.array([0, 0.18350341907227,   0.55278640450004,   1.0,
         1.44721359549996,   1.81649658092773, 2.0]) / 2.
    zone_weight = np.array([6.0, 1.0, 5.0, 5.0, 1.0]) / 6.0
    weights0 = np.array([6.0, 1.0, 0.0, 5.0, 0.0, 5.0, 0.0, 1.0]) / 6.0
    weights1 = np.array([1470., 77.0, 432.0, 625.0, 672.0, 625.0,
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
            elif path_flag:
                digits = 2
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
            
    def get_new_nids(self, zone, depth=0):
        assert depth < 7
        nids = []
        if self.type == 'Gaussian':
            base = zone * 10 ** (self.maxdepth - depth)
            for i in range(2, 7):
                nids.append(base + i)
        elif self.type == 'linear':
            assert depth == 0
            nids_num = int((self.end - self.begin) / self.step) + 1
            digits = int(np.ceil(np.log10(nids_num)))
            base = self.index * 10 ** digits
            for i in range(nids_num):
                nids.append(i + base)
        elif self.type == 'poles':
            assert depth == 0
            base = zone * 100
            res_nids0 = np.arange(1, 9, 2) * np.pi * 1.j + self.begin
            res_nids1 = np.arange(1, 9, 2) * np.pi * 1.j + self.end
            nids0 = np.arange(len(res_nids0)) + base + 10 + 1
            nids1 = np.arange(len(res_nids1)) + base + 20 + 1
            nids = np.append(nids0, nids1).tolist()
        else:
            raise RuntimeError('Wrong Path Type % s' % self.type)
        #self.nids += nids
        return nids
    
    def add_node(self, nid, energy, function):
        self.nids.append(nid)
        self.functions.append(function)
        self.energies.append(energy)
  
    def get_ends_nids(self):
        nids = []
        num = self.num
        if self.index in [1, 2, 3, 6]:
            nids.append(self.index * num + 1)
        if self.index in [3, 6]:
            nids.append(self.index * num + 7)
        #self.nids += nids
        return nids
       
    def get_energy(self, flags):
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
            wei0 = np.abs(self.end - self.begin) / 2.
            for i in range(self.maxdepth - 1):
                wei0 *= self.zone_weight[flags[i + 1]]
            wei1 = wei0
            wei0 *= self.weights0[flags[-1]]
            wei1 *= self.weights1[flags[-1]]
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
        self.fermi = fermi
        self.bias = bias
        self.tp = tp
        self.neint = neint
        self.min_bias = np.min(bias)
        self.max_bias = np.max(bias)
        self.dtype = complex
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
        depth = 0
        converge = False
        while not converge and depth < self.maxdepth:
            nids, path_indices = self.collect(zones, depth)
            loc_nids, loc_path_indices = self.distribute(nids, path_indices)
            self.calculate(loc_nids, loc_path_indices)
            if depth == 0:
                self.joint(zones)
            converge, zones = self.check_convergence(zones, depth)
            depth += 1
            self.transfer(zones, depth)
            print depth
        
    def collect(self, zones, depth):
        nids = []
        path_indices = []
        for zone in zones:
            path_index = zone // (10 ** depth) - 1
            new_nids = self.paths[path_index].get_new_nids(zone, depth)
            nids += new_nids
            path_indices += [path_index] * len(new_nids)
            
            if depth == 0:
                new_nids = self.paths[path_index].get_ends_nids()
                nids += new_nids
                path_indices += [path_index] * len(new_nids)
        return nids, path_indices
    
    def distribute(self, nids, path_indices):
        loc_nids = np.array_split(nids, self.comm.size)
        loc_path_indices = np.array_split(path_indices, self.comm.size)
        return loc_nids[self.comm.rank], loc_path_indices[self.comm.rank]
    
    def calculate(self, loc_nids, path_indices):
        for nid, path_index in zip(loc_nids, path_indices):
            exp10 = int(np.floor(np.log10(nid)))
            flags = self.paths[path_index].get_flags(nid, True)
            energy = self.paths[path_index].get_energy(flags[1:])
            if self.tp.recal_path or (not self.tp.recal_path and
                                      path_index in [0, 1, 2, 5]) :
                calcutype = self.calcutype[path_index]
                green_function = self.tp.calgfunc(energy, calcutype)
                self.paths[path_index].functions.append(green_function)
                self.paths[path_index].energies.append(energy)
                self.paths[path_index].nids.append(nid)
                
    def joint(self, zones):
        my_zones = np.array_split(zones, self.comm.size)[self.comm.rank]
        my_info_dict = {}
        num = 0
        for zone in my_zones:
            if zone in [3, 4, 5, 6]:
                pass
            else:
                path_index = zone - 1
                order = 10 ** self.maxdepth
                base = zone * order
                nid = base + 7
                link_nid = nid + order - 6
                link_path_index = link_nid // (10 ** self.maxdepth) - 1
                flag = str(self.comm.rank) + '_' + str(num)
                my_info_dict[flag] = np.array([nid, link_nid,
                                           path_index, link_path_index], int)
                num += 1
                
        info_dict = gather_ndarray_dict(my_info_dict, self.comm)
        
        for name in info_dict:
            nid, link_nid, path_index, link_path_index = info_dict[name]
            rank = self.get_rank(link_path_index, link_nid)
            if self.comm.rank == rank:
                link_path = self.paths[link_path_index]                
                index = link_path.nids.index(link_nid)
                function = link_path.functions[index]
                energy = link_path.energies[index]
                self.paths[path_index].add_node(nid, energy, function)

    def transfer(self, zones, depth):
        my_zones = np.array_split(zones, self.comm.size)[self.comm.rank]
        my_info_dict = {}
        num = 0        
        for zone in my_zones:
            path_index = zone // (10 ** depth) - 1
            order = 10 ** (self.maxdepth - depth)
            base = zone * order
            node_index = zone % 10
            nid = zone * order + 1

            link_nid = (zone - node_index) * order + 2 * node_index - 1
            flag = str(self.comm.rank) + '_' + str(num)                
            my_info_dict[flag] = np.array([nid, link_nid, path_index], int)
            num += 1
                
            nid = zone * order + 7
            link_nid = (zone - node_index) * order + 2 * node_index + 1
            flag = str(self.comm.rank) + '_' + str(num)
            my_info_dict[flag] = np.array([nid, link_nid, path_index], int)
            num += 1                
                
                
        info_dict = gather_ndarray_dict(my_info_dict, self.comm)        
        for name in info_dict:
            nid, link_nid, path_index = info_dict[name]
            rank = self.get_rank(path_index, link_nid)
            if self.comm.rank == rank:
                path = self.paths[path_index]                
                index = path.nids.index(link_nid)
                function = path.functions[index]
                energy = path.energies[index]
                self.paths[path_index].add_node(nid, energy, function)                
            
    def get_rank(self, path_index, nid):
        info_array = np.zeros([self.comm.size], int)
        if nid in self.paths[path_index].nids:
            info_array[self.comm.rank] = 1
        self.comm.sum(info_array)
        assert np.sum(info_array) == 1
        return np.argmax(info_array)
          
    #def get_begin(self, zone, depth):
    #    if depth == 0:
    #        return np.array([zone * self.num * 10 + 1])
    #    else:
    #        tens = zone // 10
    #        digits = zone % 10
    #        return np.array([tens * 10 ** (self.maxdepth - depth)
    #                          + digits * 2 - 1]) 
    
    #def get_end(self, zone, depth):
    #    if depth == 0:
    #        if zone in [1, 2]:
    #            return (zone + 1) * self.num * 10 + 1
    #        elif zone in [3, 6]:
    #            return zone * self.num * 10 + 7
       
    def check_convergence(self, zones, depth):
        new_zones = []
        nbmol = self.tp.nbmol
        converged = True
        errs = [0]
        for zone in zones:
            if zone in [4, 5]:
                pass
            else:
                order = 10 ** (self.maxdepth - depth)
                base = zone * order
                #begin = self.get_begin(zone, depth)
                #nids = np.append(begin, np.arange(2, 7) + base)
                #end = self.get_end(zone, depth)
                #nids = np.append(nids, end)

                original_nids = np.arange(1, 8) + base
                path_index = zone // 10 ** depth - 1
                gr_sum0 = np.zeros([nbmol, nbmol], dtype=self.dtype)
                gr_sum1 = np.zeros([nbmol, nbmol], dtype=self.dtype)
                path = self.paths[path_index]            
                for nid in original_nids:
                    if nid in path.nids:
                        flags = path.get_flags(nid, True)
                        weight0, weight1 = path.get_weight(flags)
                        index = path.nids.index(nid)
                        gr_sum0 += path.functions[index] * weight0
                        gr_sum1 += path.functions[index] * weight1
                self.comm.sum(gr_sum0)
                self.comm.sum(gr_sum1)
                err = np.max(abs(gr_sum0 - gr_sum1))
                if err > self.eq_err:
                    converged = False
                    new_zones += range(zone * 10 + 1, zone * 10 + 4)
                    errs.append(err)
        print 'err', np.max(np.abs(errs))
        return converged, new_zones        
        
        
    