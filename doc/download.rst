.. _download:

========
Download
========

In order to use the code, you need to get 1) the code and 2) the
setups for all your atoms (get the setups :ref:`here <setups>`).  You can
get the code from the latest release or form a fresh checkout from our
svn-server - read below.  When you have the code, go to the
:ref:`installationguide`.



From subversion (SVN) at svn.fysik.dtu.dk
=========================================

The latest revision can be obtained like this::

  $ svn checkout https://USER@svn.fysik.dtu.dk/projects/gpaw/trunk gpaw

where ``USER`` is your svn user-id.  You can also checkout the code anonymously without the ``USER@`` part, but then commits are not possible.
See :ref:`faq` in case of problems.


From tar-file
=============

You should always use the latest version from SVN.  If you for some reason can't get that, you can try this tar-file_, which corresponds to the latest stable release.

.. _tar-file: http://wiki.fysik.dtu.dk/stuff/gpaw-0.4.2171.tar.gz