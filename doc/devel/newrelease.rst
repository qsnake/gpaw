.. _newrelease:

===========
New release
===========

XXX This page needs to be updated!!

When it is time for a new release of the code, here is what you have to do:

* Make a fresh checkout from svn::

   svn co https://svn.fysik.dtu.dk/projects/gpaw/trunk gpaw

  and run the test suite (run ``python test.py`` in ``tests`` directory).

* Make a tag in svn, using the current version number::

    svn copy https://svn.fysik.dtu.dk/projects/gpaw/trunk https://svn.fysik.dtu.dk/projects/gpaw/tags/0.4 -m "Version 0.4"

* **Checkout** the source, specyfing the version number in the directory name::

   svn co https://svn.fysik.dtu.dk/projects/gpaw/tags/0.4 gpaw-0.4

* Create the tar file::

   cd gpaw-0.4
   python setup.py sdist

  Note that the current svn release number is put into the name of the
  tar file automatically.

* Put the tar file on web2 (set it read-able for all)::

   scp dist/gpaw-0.4.2724.tar.gz root@web2:/var/www/wiki/gpaw-files

* Change the :ref:`download` link to the new tar file.

* Optionally, update the :ref:`releasenotes`.

* Increase the version number in gpaw/version.py, and commit the change::

    cd gpaw
    svn ci -m "Version 0.5"

  Now the trunk is ready for work on the new version.

* Send announcement email to:

  - gridpaw-developer@lists.berlios.de
