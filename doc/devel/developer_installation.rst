.. _developer_installation:

======================
Developer installation
======================

The :ref:`standard installation <installationguide>` will copy all the
Python files to the standard place for Python modules (something like
:file:`/usr/lib/python2.5/site-packages/gpaw`) or to
:file:`{<my-directory>}/lib/python/gpaw` if you used the
:file:`--home={<my-directory>}` option.  As a developer, you will want
Python to use the files from the SVN checkout that you are hacking on.

Do the following:

  * Checkout the :ref:`latest_development_release`.

  * build :ref:`c_extension`::

     [~]$ cd gpaw
     [gpaw]$ python setup.py build_ext 2>&1 | tee build_ext.log

  This will build two things:

  * :file:`_gpaw.so`:  A shared library for serial calculations containing
    GPAW's C-extension module.  The module will be in
    :file:`~/gpaw/build/lib.{<platform>}-2.5/`.
  * :file:`gpaw-python`: A special Python interpreter for parallel
    calculations.  The interpreter has GPAW's C-code build in.  The
    :file:`gpaw-python` executable is
    in :file:`~/gpaw/build/bin.{<platform>}-2.5/`.

  **Note** the :file:`gpaw-python` interpreter will only be made if
  :file:`setup.py` found an ``mpicc`` compiler.

  * Prepend :file:`~/gpaw` onto your :envvar:`$PYTHONPATH` and
    :file:`~/gpaw/build/bin.{<platform>}-2.5:~/gpaw/tools` onto
    :envvar:`$PATH`, e.g. put into :file:`~/.tcshrc`::

     setenv PYTHONPATH ${HOME}/gpaw:${PYTHONPATH}
     setenv PATH ${HOME}/gpaw/build/bin.<platform>-2.5:${HOME}/gpaw/tools:${PATH}

    or if you use bash, put these lines into :file:`~/.bashrc`::

     export PYTHONPATH=${HOME}/gpaw:${PYTHONPATH}
     export PATH=${HOME}/gpaw/build/bin.<platform>-2.5:${HOME}/gpaw/tools:${PATH}
