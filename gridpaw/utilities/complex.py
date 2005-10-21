# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import Numeric as num


def cc(x):
    """Complex conjugate."""
    tp = type(x)
    if tp is float:
        return x
    if tp is complex:
        return x.conjugate()
    if x.typecode() == num.Float:
        return x
    else:
        return num.conjugate(x)

    
def real(x):
    """Real part."""
    tp = type(x)
    if tp is float:
        return x
    if tp is complex:
        return x.real
    if x.typecode() == num.Float:
        return x
    else:
        return x.real


if __name__ == '__main__':
    pass
