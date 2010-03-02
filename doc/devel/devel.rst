.. _devel:

===========
Development
===========

GPAW development can be done by anyone! Just take a look at the
:ref:`todo` list and find something that suits your talents!

The primary source of information is still the :ref:`manual` and
:ref:`documentation`, but as a developer you might need additional
information which can be found here. For example the :ref:`code_overview`.

As a developer, you should subscribe to all GPAW related :ref:`mailing_lists`.

Now you are ready to to perfom a :ref:`developer_installation` and
start development!

.. toctree::
   :maxdepth: 1

   developer_installation

.. note --- below toctrees are defined in separate files to make sure that the line spacing doesn't get very large (which is of course a bad hack)

Development topics
==================

When committing significant changes to the code, remember to add a
note in the :ref:`releasenotes` at the top (current svn) - the version
to become the next release.

.. toctree::
   :maxdepth: 1

   toc-general

* The latest report_ from PyLint_ about the GPAW coding standard.

.. spacer

* Details about supported :ref:`platforms_and_architectures`.

.. _report: http://dcwww.camd.dtu.dk/~s052580/pylint/gpaw/
.. _PyLint: http://www.logilab.org/857


.. _code_overview:

Code Overview
=============

Keep this picture under your pillow:

.. _the_big_picture:

.. image:: bigpicture.png
   :target: ../bigpicture.pdf

The developer guide provides an overview of the PAW quantities and how
the corresponding objects are defined in the code:

.. toctree::
   :maxdepth: 2

   overview
   developersguide
   proposals/proposals
   paw
   symmetry
   wavefunctions
   setups
   density_and_hamiltonian
   communicators
   others


The GPAW logo
=============

The GPAW-logo is available in the odg_ and svg_ file formats:
gpaw-logo.odg_, gpaw-logo.svg_

.. _odg: http://www.openoffice.org/product/draw.html
.. _svg: http://en.wikipedia.org/wiki/Scalable_Vector_Graphics
.. _gpaw-logo.odg: ../_static/gpaw-logo.odg
.. _gpaw-logo.svg: ../_static/gpaw-logo.svg


Statistics
==========

The image below shows the development in the volume of the code as per
February 11 2010.

.. image:: ../_static/stat.png

*Documentation* refers solely the contents of this homepage. Inline
documentation is included in the other line counts.


Contributing to GPAW
====================

Getting commit access to our SVN repository works the same way as for
the `ASE project`_.  Here is the list of current committers:


==============  ============================
id              real name
==============  ============================
askhl           Ask Hjorth Larsen
carstenr        Carsten Rostgaard
dlandis         David Landis
dulak           Marcin Dulak
georg           Poul Georg Moses
getri           George Tritsaris
hahansen        Heine Anton Hansen
haiping         Haiping Lin
hhk05           Henrik Kristoffersen
jensj           Jens Jørgen Mortensen
jingzhe         Jingzhe Chen
jsm             Jess Stausholm-Møller
jstenlund       Jonathan Stenlund
jussie          Jussi Enkovaara
juya            Jun Yan
kelkkane        Andre Kelkkanen
kuismam         Mikael Kuisma
lara            Lara Ferrighi
lauri           Lauri Lethovaara
lopeza          Olga Lopez
madsbk          Mads Burgdorff Kristensen
marsalek        Ondrej Marsalek
mathiasl        Mathias Ljungberg
miwalter        Michael Walter
moses           Poul Georg Moses
mvanin          Marco Vanin
naromero        Nichols Romero
peterklue       Peter Kluepfel
rostgaard       Carsten Rostgaard
s032082         Christian Glinsvad
s042606         Janosch Michael Rauba
s052580         Troels Kofoed Jacobsen
schiotz         Jakob Schiotz
strange         Mikkel Strange
tolsen          Thomas Olsen
==============  ============================

.. _ASE project: https://wiki.fysik.dtu.dk/ase/development/contribute.html

.. epost={'askhl': 'askhl fysik,dtu,dk', 'tolsen': 'tolsen fysik,dtu,dk', 'jussie': 'jussi,enkovaara csc,fi', 'dulak': 'dulak fysik,dtu,dk', 'hhk05': 'hhk05 inano,dk', 'carstenr': 'carstenr fysik,dtu,dk', 'lara': 'laraf phys,au,dk', 'lauri': 'lauri,lehtovaara iki,fi', 'naromero': 'naromero alcf,anl,gov', 'kuismam': 'mikael,kuisma tut,fi', 'mathiasl': 'mathiasl physto,se', 'haiping': 'H,Lin1 liverpool,ac,uk', 'georg': 'georg fysik,dtu,dk', 'jingzhe': 'jingzhe fysik,dtu,dk', 'strange': 'strange fysik,dtu,dk', 'rostgaard': 'rostgaard fysik,dtu,dk', 'schiotz': 'schiotz fysik,dtu,dk', 'peterklue': 'peter theochem,org', 'moses': 'poulgeorgmoses gmail,com', 's032082': 's032082 fysik,dtu,dk', 'jensj': 'jensj fysik,dtu,dk', 'jstenlund': 'jonathan,stenlund abo,fi', 'jsm': 'jsm phys,au,dk', 'dlandis': 'dlandis fysik,dtu,dk', 'getri': 'getri fysik,dtu,dk', 'marsalek': 'ondrej,marsalek gmail,com', 's052580': 's052580 fysik,dtu,dk', 's042606': 's042606 fysik,dtu,dk', 'hahansen': 'hahansen fysik,dtu,dk', 'miwalter': 'Michael,Walter fmf,uni-freiburg,de', 'mvanin': 'mvanin fysik,dtu,dk', 'juya': 'juya fysik,dtu,dk', 'lopeza': 'lopez cc,jyu,fi', 'kelkkane': 'kelkkane fysik,dtu,dk', 'madsbk': 'madsbk diku,dk'}
