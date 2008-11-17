.. _svn:

===
SVN
===

Browse svn online_.

.. _online: http://svn.fysik.dtu.dk/projects/gpaw/


Take a look at this `SVN cheat sheet`_

.. _SVN cheat sheet: ../_static/svn-refcard.pdf

Working with branches
=====================

Creating a new branch::

  $ svn copy https://svn.fysik.dtu.dk/projects/gpaw/trunk https://svn.fysik.dtu.dk/projects/gpaw/branches/mixing -m "Experimental density mixing branch"

Merge changes from trunk into branch::

  $ svn merge -r 869:HEAD https://svn.fysik.dtu.dk/projects/gpaw/trunk
  $ svn ci -m "Merged changes from trunk (869:898) into branch"

Merge branch to trunk::

  $ cd <root directory of branch>
  $ svn log --verbose --stop-on-copy
  ...
  ...
  r667 | jensj | 2007-04-18 19:22:52 +0200 (Wed, 18 Apr 2007) | 1 line
  Changed paths:
     A /branches/new-interface (from /trunk:666)
  $ cd <root directory of trunk>
  $ svn up
  At revision 957.
  $ svn merge -r 667:957 https://svn.fysik.dtu.dk/projects/gpaw/branches/new-interface
