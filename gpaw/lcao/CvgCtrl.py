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
    
    def cvgjudge(self, matin):
        dmatmax = 0
        self.bcvglast = self.bcvg
        self.bcvg = 0
        
        nbmol = matin.shape[-1]
        if self.step > 0:
            dmatmax = npy.max(npy.abs(self.matlast - matin))
            arg_max1 = npy.argmax(npy.abs(self.matlast - matin)) / nbmol
            arg_max2 = npy.argmax(npy.abs(self.matlast - matin)) % nbmol
            if dmatmax < self.tol:
                self.bcvg = 1
            if self.tol >= 0:
                if self.matname == 'f' and self.master:
                    print 'Hamiltonian: dmatmax= [%d, %d] %f tol=%f isCvg=%d'\
                          %(arg_max1, arg_max2, dmatmax, self.tol, self.bcvg)
                elif self.matname == 'd' and self.master:
                    print  'Density: dmatmax= [%d %d] %f tol=%f isCvg=%d'\
                        %(arg_max1, arg_max2, dmatmax, self.tol, self.bcvg)                    
        self.record_dmatmax.append(dmatmax)   #attention here, vector push_back in C

    def matcvg(self, matin):
        if self.tol >= 0:
            self.cvgjudge(matin)
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
        elif self.cvgmethod == 'CVG_Broydn_lnsrch':
            matout = self.broydn_lnsrch(matin)
        
        self.matlast = npy.copy(matout)
        self.step = self.step + 1
        return matout
    
    def broydn(self,matin):
        
        if self.step >= 2:
            self.dmat[1] = npy.copy(self.dmat[0])
        if self.step > 0:
            self.dmat[0] = matin - self.mat[0]
            fmin = npy.sum(self.dmat[0] * self.dmat[0]) #attention here matDotSum in C
            #print self.matname + 'CvgCtrl: broydn: fmin=', fmin
        if self.step == 0:
            self.dmat = [npy.empty(matin.shape), npy.empty(matin.shape)]
            self.mat = [npy.empty(matin.shape), npy.empty(matin.shape)] 
            self.eta = npy.empty(matin.shape)
            self.c =  []
            self.v = []
            self.u = []
            matout = npy.copy(matin)
            self.mat[0] = npy.copy(matout)
        else:
            if self.step >= 2:
                del self.c[:]
                self.v.append((self.dmat[0] - self.dmat[1]) / 
                    npy.sum((self.dmat[0] - self.dmat[1]) * (self.dmat[0]-
                                                    self.dmat[1]))) #matDotSum
                for i in range(self.step - 1):
                    self.c.append(npy.sum(self.v[i] * self.dmat[0])) #matDotSum
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

            








