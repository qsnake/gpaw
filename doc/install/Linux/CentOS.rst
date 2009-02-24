.. _CentOS:

======
CentOS
======

Here you find information about the the system
`<http://www.centos.org/>`_.

i386 RPM based systems (el4, el5)
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

   - el4 (CentOS 4)::

      wget http://packages.sw.be/rpmforge-release/rpmforge-release-0.3.6-1.el4.rf.i386.rpm
      rpm -ivh rpmforge-release-0.3.6-1.el4.rf.i386.rpm
      wget ftp://ftp.scientificlinux.org/linux/scientific/4x/i386/SL/RPMS/numpy-1.0.4-1.i386.rpm
      wget --no-check-certificate https://www.scientificlinux.org/documentation/gpg/RPM-GPG-KEY-dawson
      rpm --import RPM-GPG-KEY-dawson
      yum -y localinstall numpy-1.0.4-1.i386.rpm
      wget --no-check-certificate $REPO/i386/python-matplotlib-0.91.2-3.el4.fys.i386.rpm
      wget --no-check-certificate $REPO/i386/pytz-2006p-1.el4.fys.i386.rpm
      yum -y localinstall python-matplotlib-0.91.2-3.el4.fys.i386.rpm pytz-2006p-1.el4.fys.i386.rpm

   - el5 (CentOS 5)::

      wget http://packages.sw.be/rpmforge-release/rpmforge-release-0.3.6-1.el5.rf.i386.rpm
      yum -ivh rpmforge-release-0.3.6-1.el5.rf.i386.rpm
      rpm --import http://download.fedora.redhat.com/pub/epel/RPM-GPG-KEY-EPEL
      yum -y install blas-devel lapack-devel
      wget http://download.fedora.redhat.com/pub/epel/5/i386/numpy-1.0.4-1.el5.i386.rpm
      yum -y localinstall numpy-1.0.4-1.el5.i386.rpm
      yum -y update numpy
      wget http://download.fedora.redhat.com/pub/epel/5/i386/python-matplotlib-0.90.1-1.el5.i386.rpm
      wget http://download.fedora.redhat.com/pub/epel/5/i386/pytz-2006p-1.el5.noarch.rpm
      yum -y localinstall python-matplotlib-0.90.1-1.el5.i386.rpm pytz-2006p-1.el5.noarch.rpm
      yum -y update python-matplotlib

 - install gpaw and ase (**Note**! replace xxx with one of el4, el5)::

    yum -y remove campos-gpaw-setups campos-gpaw campos-ase3
    wget --no-check-certificate $REPO/i386/campos-ase3-3.0.0.507-1.xxx.fys.i386.rpm
    wget --no-check-certificate $REPO/i386/campos-gpaw-0.4.2409-1.xxx.fys.gcc.i386.rpm
    wget --no-check-certificate $REPO/noarch/campos-gpaw-setups-0.4.2039-1.xxx.fys.noarch.rpm
    yum -y localinstall campos-ase3-3.0.0.507-1.xxx.fys.i386.rpm
    yum -y localinstall campos-gpaw-0.4.2409-1.xxx.fys.gcc.i386.rpm campos-gpaw-setups-0.4.2039-1.xxx.fys.noarch.rpm
