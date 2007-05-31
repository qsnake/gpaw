from math import sin, cos, pi

import numpy as npy


def rotate(rotations, rotation=npy.diag([1.0, -1, 1])):
    if rotations == '':
        return rotation
    
    for i, a in [('xyz'.index(s[-1]), float(s[:-1]) / 180 * pi)
                 for s in rotations.split(',')]:
        s = sin(a)
        c = cos(a)
        if i == 0:
            rotation = npy.dot(rotation, [( 1,  0,  0),
                                          ( 0,  c, -s),
                                          ( 0,  s,  c)])
        elif i == 1:
            rotation = npy.dot(rotation, [( c,  0, -s),
                                          ( 0,  1,  0),
                                          ( s,  0,  c)])
        else:
            rotation = npy.dot(rotation, [( c, -s,  0),
                                          ( s,  c,  0),
                                          (-0,  0,  1)])
    return rotation
