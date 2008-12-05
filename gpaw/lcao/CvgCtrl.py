import numpy as npy

class CvgCtrl:
    def __init__(self):
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
        self.bscalealpha = 0 # bool
        self.tol = 1e-4
        self.bcvg = 0 #bool

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
        print self.matname + 'CvgCtrl: Init:Method=', self.cvgmethod,\
                        'tol0', self.tol, 'ctrlmode=', ctrlmode, \
                                'steady_check0', self.bstdchk
        print 'alpha=', self.alpha, 'ndiis0', self.ndiis, 'tolx0', self.tolx
        print 'alpha_control_method=', self.asmethod, 'alphascaling=', \
                      self.alphascaling, 'allowedmatmax=', self.allowedmatmax
    
    def cvgjudge(self, matin):
        dmatmax = 0
        self.bcvglast = self.bcvg
        self.bcvg = 0
        if self.step > 0:
            dmatmax = npy.max(npy.abs(self.matlast - matin))
            if dmatmax < self.tol:
                self.bcvg = 1
            if self.tol >= 0:
                print self.matname + 'CvgCtrl: CvgJudge: dmatmax=', \
                                dmatmax, 'tol=', self.tol, 'isCvg=', self.bcvg
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
        elif self.cvgmethod == 'CVG_broydn':
            matout = self.broydn(matin)
        self.matlast = npy.copy(matout)
        self.step = self.step + 1
        return matout
    def broydn(self,matin):
        
        if self.step >= 2:
            self.dmat[1] = npy.copy(self.dmat[0])
        if self.step > 0:
            self.dmat[0] = matin - self.mat[0]
            fmin = npy.sum(self.dmat[0] * self.dmat[0]) #attention here matDotSum in C
            print self.matname + 'CvgCtrl: broydn: fmin=', fmin
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


        
    

            








