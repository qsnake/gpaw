.. _profiling:

=========
Profiling
=========

Python has a ``profile`` module to help you find the places in the code where the time is spent.  Let's say you have a script (in ``script.py``) that you want to run through the profiler.  This is what you do:

>>> import profile
>>> profile.run('import script', 'prof')

This will run your script and generate a profile in the file ``prof``.  You can also generate the profile by inserting a line like this in your script::

  ...
  import profile
  profile.run('atoms.GetPotentialEnergy()', 'prof')
  ...

To analyse the results, you do this::

 >>> import pstats
 >>> pstats.Stats('prof').strip_dirs().sort_stats('time').print_stats(20)
 Thu Dec 15 09:09:40 2005    prof 

          54104 function calls (53162 primitive calls) in 5.790 CPU seconds

    Ordered by: internal time
    List reduced from 782 to 20 due to restriction <20>

    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       389    1.490    0.004    1.530    0.004 operators.py:55(apply)
         6    1.160    0.193    1.160    0.193 localized_functions.py:257(__init__)
    295/59    0.620    0.002    3.140    0.053 poisson_solver.py:65(iterate)
       619    0.590    0.001    0.600    0.001 Numeric.py:349(vdot)
       504    0.530    0.001    0.550    0.001 transformers.py:43(apply)
       396    0.490    0.001    0.490    0.001 xc_functional.py:16(calculate_spinpaired)
      2744    0.070    0.000    0.070    0.000 __init__.py:26(is_contiguous)
      7442    0.060    0.000    0.090    0.000 pyexpat.c:457(CharacterData)
      ...
      ...

The list shows the 20 functions where the most time is spent.  Check the pstats_ documentation if you want to do more fancy things.

.. _pstats: http://docs.python.org/lib/module-profile.html


.. tip::

   Since the ``profile`` module does not time calls to C-code, it
   is a good idea to run the code in debug mode - this will wrap
   calls to C-code in Python functions::

     $ python script.py --debug
