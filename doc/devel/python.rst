.. _python:

======
Python
======

Terminology
===========

>>> def f(x, m=2, n=1):
...     y =  x + n
...     return y**k

Here ``f`` is a function, ``x`` is an argument, ``m`` and ``n`` are keywords with default values ``2`` and ``1`` and ``y`` is a variable.

>>> class A:
...     def __init__(self, b):
...         self.c = b
...     def m(self):
...         return f(self.c, n=0)

A class ``A`` is defined, ``__init__`` is a constructor, ``c`` is an attribute and ``m`` is a method.

>>> a = A(7)
>>> a.c
7
>>> a.m()
49
>>> g = a.m
>>> g()
49

Here we make an instance/object ``a`` of type ``A`` and ``g`` is a method bound to ``a``.


Types
=====

 ===========  =====================  ==========================
 type         description            example
 ===========  =====================  ==========================
 ``bool``     boolean                ``False``
 ``int``       integer                ``117``  
 ``float``    floating point number  ``1.78``  
 ``complex``  complex number         ``0.5 + 2.0j``  
 ``str``      string                 ``'abc'``  
 ``tuple``    tuple                  ``(1, 'hmm', 2.0)``  
 ``list``     list                   ``[1, 'hmm', 2.0]``  
 ``dict``     dictionary             ``{'a': 7.0, 23: True}``  
 ``file``     file                   ``open('stuff.dat', 'w')``  
 ===========  =====================  ==========================
