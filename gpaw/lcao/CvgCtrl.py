import numpy as npy

class CvgCtrl:
    def __init__(self, master):
        self.master = master
        self.matname = 'f'
        self.cvgmethod = 'CVG_None'
        self.ndiis = 10
        self.step = 0
        self.asmethod = 'SCL_None'
        self.alpha = 0.1
        self.alphamax = 0.1
        self.alphascaling = 0
        self.allowedmatmax = 1e-3
        self.asbeginstep = 10
        self.bscalealpha = False
        self.tol = 1e-4
        self.alf = 1e-4
        self.fmin = 100
        self.tolx = 1e-7
        self.bcvg = 0 #bool
        self.alam = 1.0
        self.alamin = 1.0
        #steady check
        self.bstdchk = 0 #bool
        self.bcvglast = 0 #bool
        self.record_dmatmax = []
        self.record_alpha = []

    def __call__(self, inputinfo, matname, other):
        self.other = other
        self.matname = matname
        if self.matname == 'f' or self.matname =='d':
            self.asmethod = inputinfo[self.matname + 'asmethodname']
            self.cvgmethod = inputinfo[self.matname + 'methodname']
            self.alpha = inputinfo[self.matname + 'alpha']
            self.alphascaling = inputinfo[self.matname + 'alphascaling']
            self.tol = inputinfo[self.matname + 'tol']
            self.allowedmatmax = inputinfo[self.matname + 'allowedmatmax']
            self.ndiis = inputinfo[self.matname +'ndiis']
            self.tolx = inputinfo[self.matname +'tolx']
            self.bstdchk = inputinfo[self.matname +'steadycheck']
        else:
            print 'CvgCtrl: Error in Init, matname is incoorrect'
        if self.other == None:
            ctrlmode = 'SelfCtrl'
        else:
            ctrlmode = 'Co-Ctrl'
        #print self.matname + 'CvgCtrl: Init:Method=', self.cvgmethod,\
        #                'tol0', self.tol, 'ctrlmode=', ctrlmode, \
        #                        'steady_check0', self.bstdchk
        #print 'alpha=', self.alpha, 'ndiis0', self.ndiis, 'tolx0', self.tolx
        # print 'alpha_control_method=', self.asmethod, 'alphascaling=', \
        #              self.alphascaling, 'allowedmatmax=', self.allowedmatmax
    
    def cvgjudge(self, matin, txt):
        dmatmax = 0
        self.bcvglast = self.bcvg
        self.bcvg = 0
        if self.matname == 'f' or self.matname == 'd':       
            nbmol = matin.shape[-1]
            if self.step > 0:
                dmatmax = npy.max(npy.abs(self.matlast - matin))
                aa = npy.argmax(npy.abs(self.matlast - matin))
                arg_max1 = aa / nbmol ** 2
                arg_max2 = (aa - arg_max1 * nbmol ** 2) / nbmol
                arg_max3 = aa % nbmol
                if dmatmax < self.tol:
                    self.bcvg = 1
                if self.tol >= 0:
                    if self.matname == 'f' and self.master:
                        txt('Hamiltonian: dmatmax= [%d, %d, %d] %f tol=%f isCvg=%d'\
                             %(arg_max1, arg_max2, arg_max3, dmatmax, self.tol, self.bcvg))
                        print 'Hamiltonian: dmatmax= [%d, %d, %d] %f tol=%f isCvg=%d'\
                             %(arg_max1, arg_max2, arg_max3, dmatmax, self.tol, self.bcvg)
                    elif self.matname == 'd' and self.master:
                        txt('Density: dmatmax= [%d %d, %d] %f tol=%f isCvg=%d'\
                            %(arg_max1, arg_max2, arg_max3, dmatmax, self.tol, self.bcvg))
                        print 'Density: dmatmax= [%d %d, %d] %f tol=%f isCvg=%d'\
                            %(arg_max1, arg_max2, arg_max3, dmatmax, self.tol, self.bcvg)
            self.record_dmatmax.append(dmatmax)   #attention here, vector push_back in C
        #elif self.matname == 'd':
            #if self.step > 0:
             #   dmatmax = npy.max(npy.abs(self.matlast - matin.nt_sG))
              #  if dmatmax < self.tol:
               #     self.bcvg = 1
               # if self.tol >= 0:
                #    print  'Density: dmatmax= %f tol=%f isCvg=%d'\
                 #           %(dmatmax, self.tol, self.bcvg)                    
            #self.record_dmatmax.append(dmatmax)   #attention here, vector push_back in C            
    def matcvg(self, matin, txt):
        if self.tol >= 0:
            self.cvgjudge(matin, txt)
            if self.bcvg and ((self.other == None) or self.other.bcvg):
                matout = npy.copy(matin)
                return matout

        if self.cvgmethod == 'CVG_Unkown':
            pass
        elif self.cvgmethod == 'CVG_None':
            matout = npy.copy(matin)
        elif self.cvgmethod == 'CVG_Linear':
            matout = self.linear_cvg(matin)
        elif self.cvgmethod == 'CVG_Broydn':
            matout = self.broydn(matin)
            #self.broydn(matin)
        elif self.cvgmethod == 'CVG_Broydn_lnsrch':
            matout = self.broydn_lnsrch(matin)
        #if self.matname == 'f':
        self.matlast = npy.copy(matout)
        #elif self.matname == 'd':
        #    self.matlast = npy.copy(matin.nt_sG)
        self.step = self.step + 1
        #if self.matname =='f':
        return matout
    
    def broydn(self,matin):
        nmaxold = 20
        if self.step >= 2:
            self.dmat[1] = npy.copy(self.dmat[0])
        if self.step > 0:
            self.dmat[0] = matin - self.mat[0]
            fmin = npy.sum(self.dmat[0] * self.dmat[0]) #attention here matDotSum in C
            #print self.matname + 'CvgCtrl: broydn: fmin=', fmin
        if self.step == 0:
            #self.dmat = [npy.empty(matin.shape), npy.empty(matin.shape)]
            #self.mat = [npy.empty(matin.shape), npy.empty(matin.shape)] 
            #self.eta = npy.empty(matin.shape)
            self.dmat = [npy.empty(matin.shape,complex), npy.empty(matin.shape,complex)]
            self.mat = [npy.empty(matin.shape, complex), npy.empty(matin.shape, complex)] 
            self.eta = npy.empty(matin.shape, complex)
            self.c =  []
            self.v = []
            self.u = []
            matout = npy.copy(matin)
            self.mat[0] = npy.copy(matout)
        else:
            if self.step >= 2:
                del self.c[:]
                if len(self.v) >= nmaxold:
                    del self.v[0]
                    del self.u[0]
                self.v.append((self.dmat[0] - self.dmat[1]) / 
                    npy.sum((self.dmat[0] - self.dmat[1]) * (self.dmat[0]-
                                                    self.dmat[1]))) #matDotSum
                if len(self.v) < nmaxold:
                    for i in range(self.step - 1):
                        self.c.append(npy.sum(self.v[i] * self.dmat[0])) 
                else:
                    for i in range(nmaxold):
                        self.c.append(npy.sum(self.v[i] * self.dmat[0]))
                self.u.append(self.alpha * (self.dmat[0] - self.dmat[1]) + 
                                                (self.mat[0]-self.mat[1]))
                usize = len(self.u)     #usize=step-1
                for i in range(usize - 1):
                    a = npy.sum(self.v[i] * (self.dmat[0] - self.dmat[1]))
                    self.u[usize - 1] = self.u[usize - 1] - a * self.u[i]
            self.eta = self.alpha * self.dmat[0]
            usize = len(self.u)  # usize= step-1
            for i in range(usize):
                self.eta = self.eta - self.c[i] * self.u[i]
            matout = self.mat[0] + self.eta
        self.mat[1] = npy.copy(self.mat[0])
        self.mat[0] = npy.copy(matout)
        return matout
    '''
    
    def broydn(self, density):
        nt_G = density.nt_sG
        D_asp = density.D_asp.values()
        D_sap = []
        for s in range(density.nspins):
            D_sap.append([D_sp[s] for D_sp in D_asp])
        D_tap = D_sap[0]            
        if self.step >= 2:
            self.d_nt_G[1] = npy.copy(self.d_nt_G[0])
            for d_D_ap in self.d_D_ap:
                d_D_ap[1] = npy.copy(d_D_ap[0])
        if self.step > 0:
            self.d_nt_G[0] = nt_G - self.nt_iG[0]
            for D_ap, d_D_ap, D_iap in zip(D_tap, self.d_D_ap, self.D_iap):
                d_D_ap[0] = D_ap - D_iap[0]
            fmin = npy.sum(self.d_nt_G[0] * self.d_nt_G[0]) #attention here matDotSum in C
            self.dNt = density.gd.integrate(npy.fabs(self.d_nt_G[0]))
            #print self.matname + 'CvgCtrl: broydn: fmin=', fmin
        if self.step == 0:
            self.d_nt_G = [npy.empty(nt_G.shape), npy.empty(nt_G.shape)]
            #self.d_D_ap = [npy.empty(D_ap.shape), npy.empty(D_ap.shape)]
            self.nt_iG = [npy.empty(nt_G.shape), npy.empty(nt_G.shape)]
            #self.D_iap = [npy.empty(D_ap.shape), npy.empty(D_ap.shape)]
            self.eta_G = npy.empty(nt_G.shape)
            #self.eta_D = npy.empty(D_ap.shape)
            self.len_D_ap = len(D_tap)
            self.d_D_ap = []
            self.D_iap = []
            self.u_D = []
            self.v_D = []
            self.eta_D = []
            self.D_ap_out = []
            for i in range(self.len_D_ap):
                self.d_D_ap.append([npy.empty(D_tap[0].shape), npy.empty(D_tap[0].shape)])
                self.D_iap.append([npy.empty(D_tap[0].shape), npy.empty(D_tap[0].shape)])
                self.eta_D.append(npy.empty(D_tap[0].shape))
                self.u_D.append([])
                #self.v_D.append([])
                self.D_ap_out.append(npy.empty(D_tap[0].shape))
            self.c_G = []
            self.v_G = []
            #self.v_D = []
            self.u_G = []
            #self.u_D = []
            self.nt_iG[0] = npy.copy(nt_G)
            for D_ap, D_iap in zip(D_tap, self.D_iap):
                D_iap[0] = npy.copy(D_ap)
            nt_G_out = npy.copy(nt_G)
            for i in range(self.len_D_ap): 
                self.D_ap_out[i] = npy.copy(D_tap[i])
        else:
            if self.step >= 2:
                del self.c_G[:]
                self.v_G.append((self.d_nt_G[0] - self.d_nt_G[1]) / 
                    npy.sum((self.d_nt_G[0] - self.d_nt_G[1]) * (self.d_nt_G[0]-
                                                    self.d_nt_G[1]))) #matDotSum
                for i in range(self.step - 1):
                    self.c_G.append(npy.sum(self.v_G[i] * self.d_nt_G[0])) #matDotSum
                self.u_G.append(self.alpha * (self.d_nt_G[0] - self.d_nt_G[1]) + 
                                                (self.nt_iG[0]-self.nt_iG[1]))
                for i in range(self.len_D_ap):
                    self.u_D[i].append(self.alpha * (self.d_D_ap[i][0] - self.d_D_ap[i][1]) + 
                                                (self.D_iap[i][0]- self.D_iap[i][1]))
                usize = len(self.u_G)     #usize=step-1
                for i in range(usize - 1):
                    a_G = npy.sum(self.v_G[i] * (self.d_nt_G[0] - self.d_nt_G[1]))
                    self.u_G[usize - 1] = self.u_G[usize - 1] - a_G * self.u_G[i]
                    for j in range(self.len_D_ap):
                        self.u_D[j][usize - 1] = self.u_D[j][usize - 1] - a_G * self.u_D[j][i]
            self.eta_G = self.alpha * self.d_nt_G[0]
            for i in range(self.len_D_ap):
                self.eta_D[i] = self.alpha * self.d_D_ap[i][0]
            usize = len(self.u_G)  # usize= step-1
            for i in range(usize):
                self.eta_G = self.eta_G - self.c_G[i] * self.u_G[i]
                for j in range(self.len_D_ap):
                    self.eta_D[j] -= self.c_G[i] * self.u_D[j][i]
            nt_G_out = self.nt_iG[0] + self.eta_G
            for i in range(self.len_D_ap):
                self.D_ap_out[i] = self.D_iap[i][0] + self.eta_D[i]
        self.nt_iG[1] = npy.copy(self.nt_iG[0])
        for D_iap in self.D_iap:
            D_iap[1] = npy.copy(D_iap[0])
        self.nt_iG[0] = npy.copy(nt_G_out)
        for D_iap, D_ap_out in zip(self.D_iap, self.D_ap_out):
            D_iap[0] = npy.copy(D_ap_out)
        density.nt_sG = nt_G_out
        for i in range(self.len_D_ap):
            density.D_asp[i][0] = npy.copy(self.D_ap_out[i])
        
    '''
        
   
        
        
    def linear_cvg(self, matin):
        if self.step == 0:
            matout = npy.copy(matin)
            self.mat = [npy.empty(matin.shape)]
        else:
            dmatmax = npy.max(abs(self.mat[0]))
            matout = self.alpha * matin + (1 - self.alpha) * self.mat[0]
        if self.step !=0:
            print '%sCvgCtrl, LinearCvg, alpha= %f' %(self.matname, self.alpha)
        self.mat[0] = npy.copy(matout)
        return matout
            
        
    def broydn_lnsrch(self, matin):
        row = matin.shape[-2] 
        col = matin.shape[-1]
        test = 0.0
        tmplam = -100.0

        if self.step == 0:
            self.dmat = [npy.empty(matin.shape), npy.empty(matin.shape)]
            self.mat = [npy.empty(matin.shape), npy.empty(matin.shape)]
            self.c =  []
            self.v = []
            self.u = []
            matout = npy.copy(matin)
            self.mat[0] = npy.copy(matout)
            return matout
        else:
            self.dmat[0] = matin - self.mat[0]
            if self.step > 1:
                self.fmin = npy.sum(self.dmat[0] * self.dmat[0]) #attention here, matDotSum    
            if self.step > self.asbeginstep:
                self.bscalealpha = True
            if self.step == 1 or self.alam < self.alamin \
                or self.fmin < self.fold + self.alf * self.alam * self.slope \
                                                   or not self.bscalealpha:
                if self.step >= 2:
                    del self.c[:]
                    #matDotSum
                    self.v.append((self.dmat[0] - self.dmat[1]) / npy.sum((
                                               self.dmat[0] - self.dmat[1]) *
                                               (self.dmat[0] - self.dmat[1]))) 

                    self.u.append(self.alpha * (self.dmat[0] - self.dmat[1])
                                                + (self.mat[0] - self.mat[1]))
 
                    usize = len(self.u)  #usize=step-1
                    for i in range(usize):
                        self.c.append(npy.sum(self.v[i] * self.dmat[0])) #matDotSum
                    for i in range(usize - 1):
                        a = npy.sum(self.v[i] * (self.dmat[0] - self.dmat[1]))#matDotSum
                        self.u[usize - 1] -=  a * self.u[i]
                self.eta = self.alpha * self.dmat[0]
                usize = len(self.u) # usize= step -1
                for i in range(usize):
                    self.eta -= self.c[i] * self.u[i]
                self.fold = npy.sum(self.dmat[0] * self.dmat[0])  #matDotSum
                self.slope = -self.fold
                
                if len(self.eta.shape) == 2:
                    temp = self.eta
                    temp1 = self.mat[0]
                elif len(self.eta.shape) == 3:
                    temp = self.eta[0]
                    temp.shape = (row, col)
                    temp1 = self.mat[0][0]
                    temp1.shape = (row, col)
                elif len(self.eta.shape) == 4:
                    temp = self.eta[0, 0]
                    temp.shape = (row, col)
                    temp1 = self.mat[0][0, 0]
                    temp1.shape = (row, col)                    
                for i in range(row):
                    for j in range(col):
                        if test < abs(temp[i, j]) / max(
                                                    abs(temp1[i,j]),1):
                            test = abs(temp[i, j]) / max(
                                                    abs(temp1[i,j]),1)
                self.alamin = self.tolx / test
                self.alam = 1.0
                self.dmat[1] = npy.copy(self.dmat[0])
                self.mat[1] = npy.copy(self.mat[0])
                self.xold = npy.copy(self.mat[0])
                matout = self.mat[0] + self.alam * self.eta
                self.mat[0] = npy.copy(matout)
                print self.matname + 'CvgCtrl: broydn: fmin=', self.fmin
                print 'self.alam = ' , self.alam
                return matout
            else:
                if self.alam ==1.0:
                    tmplam = -self.slope / (2.0 * (self.fmin - self.fold -
                                                    self.slope))
                else:
                    rhs1 = self.fmin - self.fold -self.alam * self.slope
                    rhs2 = self.f2 - self.fold2 - self.alam2 * self.slope
                    aa = (rhs1 / (self.alam ** 2) - rhs2 / (self.alam2 ** 2)
                          ) / (self.alam-self.alam2)
                    b = (-self.alam2 * rhs1 / (self.alam ** 2) +
                             self.alam * rhs2 / (self.alam2 ** 2))/(
                             self.alam - self.alam2)
                    if aa == 0:
                        tmplam = -self.slope / (2.0 * b)
                    else:
                        disc = b ** 2 - 3.0 * aa * self.slope
                        if disc < 0.0:
                            print 'Roundoff probelm in lnsrch.'
                        else:
                            tmplam = (-b + npy.sqrt(disc)) / (3.0 * aa)
                    if tmplam > 0.5 * self.alam:
                        tmplam = 0.5 * self.alam
            if self.step > 1:
                self.alam2 = self.alam
                self.f2 = self.fmin
                self.fold2 = self.fold
                self.alam = max(tmplam, 0.1 * self.alam)
                matout = self.xold + self.alam * self.eta
                self.mat[0] = npy.copy(matout)
                print self.matname + 'Cvgctrl: lnsrch: tmplam=', tmplam, \
                                                            'alam=',self.alam
                return matout

            








