# Segmentation fault with acml's _dotblas.so
import numpy
from numpy.core.multiarray import dot
b = numpy.ones(13, numpy.complex); dot(b, b)
