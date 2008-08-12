.. _newrelease:

===========
New release
===========

XXX This page needs to be updated!!

When it is time for a new release of the code, here is what you have to do:

* Make a fresh checkout from svn::

   svn co https://svn.fysik.dtu.dk/projects/gpaw/trunk ~/gpaw

  and run the test suite (run ``python test.py`` in ``tests`` directory).
* Increase the version number in gpaw/version.py, and commit the change::

   cd ~/gpaw
   svn ci -m"Version 0.3"

  Note the svn release number of this checkin (e.g. 1405).
* Make a tag in svn, using this svn release number::

   svn copy -r 1405 https://USER@svn.fysik.dtu.dk/projects/gpaw/trunk https://USER@svn.fysik.dtu.dk/projects/gpaw/tags/0.3 -m "Version 0.3"

* **Checkout** the source, specyfing the version number in the directory name::

   svn co https://svn.fysik.dtu.dk/projects/gpaw/tags/0.3 ~/gpaw-0.3

* Create the tar file::

   cd ~/gpaw-0.3; rm -f MANIFEST
   python setup.py sdist

  Note that the current svn release number is put into the name of the tar file automatically.
* Put the tar file on web2 (set it read-able for all)::

   scp ~/gpaw-0.3/dist/gpaw-0.3.1425.tar.gz root@web2:/var/www/wiki/stuff

* Change the :ref:`download` link to the new tar file.
* Optionally, update the :ref:`releasenotes`.
* Send announcement email to:

  - campos@listserv.fysik.dtu.dk
  - gridpaw-developer@lists.berlios.de
  - python-announce-list@python.org

* Register with PyPI??
