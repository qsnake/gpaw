.. _Ubuntupackage:

==============
Ubuntu package
==============

Here you find information about the system: `<http://www.ubuntu.com/>`_.

TODO: move packages to general campos PPA and rename them to something with 'campos' in it

GPAW is available as an `Ubuntu package
<https://launchpad.net/~askhl/+archive/ppa>`_ on `Launchpad
<https://launchpad.net/>`_ for Ubuntu 9.10 or newer. To install:

- Add the package archive to the system's software
  sources::

    sudo add-apt-repository ppa:askhl/ppa

- Update the package cache::

    sudo apt-get update

- Install GPAW::

    sudo apt-get install gpaw

This will also install ASE, the GPAW setups, MPI and other
dependencies.  ASE and GPAW can be automatically upgraded when a new
stable version is released.
