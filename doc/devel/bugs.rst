.. _bugs:

If you have a problem with GPAW, like:

* bugs in the code,
* enhancement proposals
* and problems with the documentation,

you are also very welcome to report the problem to
the `mailing list`_.


.. _mailing list: http://listserv.fysik.dtu.dk/mailman/listinfo/campos


--------------
Bug collection
--------------

* General:

  - Copy-Paste errors.
  - Look for XXX, !!! or ??? in the source code!
  - Numerics default type on an alpha is Int64!!! Use long instead of int!

* Python:

  - Sometimes you need a copy and not a reference.
  - Forgetting a ``n += 1`` statement in a for loop::

      n = 0
      for thing in things:
	  thing.DoStuff(n)
	  n += 1

    Use this instead::

      for n, thing in enumerate(things):
	  thing.DoStuff(n)

  - Indentation errors like this one::

     if ok:
         x = 1.0
     else:
         x = 0.5
         DoStuff(x)

    where ``DoStuff(x)`` should have been reached in both cases!
    Emacs: always use ``C-c >`` and ``C-c <`` for shifting in and out
    blocks of code (mark the block first).

  - Don't use mutables as default values::

     class A:
         def __init__(self, a=[]):
             self.a = a # All instances get the same list!!

  - If ``H`` is a numeric array, then ``H - x`` will subtract ``x``
    from *all* elements - not only the diagonal!

  - If ``a`` is 3D grid of something, and we are using cluster boundary conditions, then we must have zeros at all times in these slices: ``a[0]``, ``a[:, 0]``, and ``a[:, :, 0]``.  So don't do something like ``a -= a.sum() / number_of_grid_points``.

* C:

  - ``if (x = 0)`` should be ``if (x == 0)`` !!!
  - try building from scratch
  - remember ``break`` in switch-case statements!
  - Check ``malloc-free`` pairs.
  - *Never* put function calls inside ``assert``'s.  Compiling with
    ``-DNDEBUG`` will remove the call.

