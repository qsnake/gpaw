import numpy as np

import _gpaw
from gpaw.xc.kernel import XCKernel


class BEE1(XCKernel):
    def __init__(self, parameters=None):
        if parameters is None:
            self.name = 'BEE1'
            parameters = [(-100.0, -5.17947467923, 23.3572146909,
                           48.3293023857, 69.5192796178, 320.0, 10000.0),
                          (9.16531398724, -9.09120195493, 0.392293264316,
                           -0.219567219503, 0.499408969276,
                           0.160492491676, 0.0932604619238)]
        else:
            self.name = 'BEE1?'
            if len(parameters) == 1:
                parameters = [parameters[0], 1.0]
        parameters = np.array(parameters, dtype=float).ravel()
        self.xc = _gpaw.XCFunctional(18, parameters)
        self.type = 'GGA'
