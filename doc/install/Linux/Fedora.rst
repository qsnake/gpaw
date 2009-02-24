.. _Fedora:

======
Fedora
======

Here you find information about the the system
`<http://fedoraproject.org/>`_.

i386 RPM based systems (fc8, fc9)
=================================

These packages were used during `The CAMD Summer School 2008 <http://www.camd.dtu.dk/English/Events/CAMD_Summer_School_2008.aspx>`_, and are very outdated now.
Users are requested to use :ref:`Installation guide <installationguide>`.

System wide installation
------------------------

The steps described below require root access and assume bash shell:

 - install external packages required by gpaw and ase::

    cd
    REPO="https://wiki.fysik.dtu.dk/stuff/school08/RPMS"
    yum -y install wget
    wget --no-check-certificate https://svn.fysik.dtu.dk/projects/rpmbuild/trunk/SOURCES/RPM-GPG-KEY-fys
    rpm --import RPM-GPG-KEY-fys

   - fc8 and fc9 (Fedora Core 8/9)::

      yum -y install blas-devel lapack-devel
      yum -y install numpy
      yum -y install python-matplotlib

 - install gpaw and ase (**Note**! replace xxx with one of fc8, fc9)::

    yum -y remove campos-gpaw-setups campos-gpaw campos-ase3
    wget --no-check-certificate $REPO/i386/campos-ase3-3.0.0.507-1.xxx.fys.i386.rpm
    wget --no-check-certificate $REPO/i386/campos-gpaw-0.4.2409-1.xxx.fys.gcc.i386.rpm
    wget --no-check-certificate $REPO/noarch/campos-gpaw-setups-0.4.2039-1.xxx.fys.noarch.rpm
    yum -y localinstall campos-ase3-3.0.0.507-1.xxx.fys.i386.rpm
    yum -y localinstall campos-gpaw-0.4.2409-1.xxx.fys.gcc.i386.rpm campos-gpaw-setups-0.4.2039-1.xxx.fys.noarch.rpm
