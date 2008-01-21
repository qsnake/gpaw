# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import numpy as npy


def cc(x):
    """Complex conjugate."""
    if isinstance(x, float):
        return x
    if isinstance(x, complex):
        return x.conjugate()
    return x.conj()

    
def real(x):
    """Real part."""
    if isinstance(x, float):
        return x
    return x.real


if __name__ == '__main__':
    pass
