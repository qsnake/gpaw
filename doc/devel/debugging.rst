.. _debugging:

=========
Debugging
=========

Python debugging
================

Even though some debugging can done just with print statements, a real debugger offers several advantages. It is possible, for example, to set breakpoints in certain files or functions, execute the code step by step, examine and change values of variables. Python contains a standard debugger *pdb*. A script can be started under the debugger control as *python -m pdb script.py* (python 2.4) or *python /path_to_pdb/pdb.py script.py* (python 2.3). Now, before the execution of the script starts one enters the debugger prompt. The most important debugger commands are:

h(elp) [command]

b(reak) [[filename:]lineno|function[, condition]]

  Set a breakpoint.

s(tep)

  Execute the current line, stop at the first possible occasion (either in a function that is called or on the next line 
  in the current function).

n(ext)

  Continue execution until the next line in the current function is reached or it returns. 

r(eturn)

  Continue execution until the current function returns.

c(ont(inue))

  Continue execution, only stop when a breakpoint is encountered. 

l(ist) [first[, last]]

  List source code for the current file. 

p expression

  Evaluate the expression in the current context and print its value. Note: "print" can also be used, but is not a   
  debugger command -- this executes the Python print statement

Most commands can be invoked with only the first letter. All the commands and their full documentation can be found from http://docs.python.org/lib/module-pdb.html


An example session might look like::

  corona1 ~/gpaw/trunk/test> python -m pdb H.py
  > /home/csc/jenkovaa/gpaw/trunk/test/H.py(1)?()
  -> from gpaw import GPAW
  (Pdb) l 11,5
   11     hydrogen.SetCalculator(calc)
   12     e1 = hydrogen.GetPotentialEnergy()
   13
   14     calc.Set(kpts=(1, 1, 1))
   15     e2 = hydrogen.GetPotentialEnergy()
   16     equal(e1, e2)
  (Pdb) break 12
  Breakpoint 1 at /home/csc/jenkovaa/gpaw/trunk/test/H.py:12
  (Pdb) c

    ... output from the script...

  > /home/csc/jenkovaa/gpaw/trunk/test/H.py(12)?()
  -> e1 = hydrogen.GetPotentialEnergy()
  (Pdb) s
  --Call--
  > /v/solaris9/appl/chem/CamposASE/ASE/ListOfAtoms.py(224)GetPotentialEnergy()
  -> def GetPotentialEnergy(self):
  (Pdb) p self
  [Atom('H', (2.0, 2.0, 2.0))]


Emacs has a special mode for python debugging which can be invoked as *M-x pdb*. After that one has to give the command to start the debugger (e.g. python -m pdb script.py). Emacs opens two windows, one for the debugger command prompt and one which shows the source code and the current point of execution. Breakpoints can be set also on the source-code window.

C debugging
===========

First of all, the c-extension should be compiled with the *-g* flag in order to get the debug information into the library. 
Also, the optimizations should be switched of which could be done in `customize.py` as::

   extra_link_args += ['-g']
   extra_compile_args += ['-O0 -g']

There are several debuggers available, the following example session applies to *gdb*::

  sepeli ~/gpaw/trunk/test> gdb python
  GNU gdb Red Hat Linux (6.1post-1.20040607.52rh)
  (gdb) break Operator_apply
  Function "Operator_apply" not defined.
  Make breakpoint pending on future shared library load? (y or [n]) y

  Breakpoint 1 (Operator_apply) pending.
  (gdb) run H.py
  Starting program: /usr/bin/python2.4 H.py

    ... output ...
  
  Breakpoint 2, Operator_apply (self=0x2a98f8f670, args=0x2a9af73b78)
    at c/operators.c:83
  (gdb)

One can also do combined C and python debugging by starting the input script as `run -m pdb H.py` i.e::

  sepeli ~/gpaw/trunk/test> gdb python
  GNU gdb Red Hat Linux (6.1post-1.20040607.52rh)
  (gdb) break Operator_apply
  Function "Operator_apply" not defined.
  Make breakpoint pending on future shared library load? (y or [n]) y

  Breakpoint 1 (Operator_apply) pending.
  (gdb) run -m pdb H.py
  Starting program: /usr/bin/python2.4 -m pdb H.py
  [Thread debugging using libthread_db enabled]
  [New Thread -1208371520 (LWP 1575)]
  > /home/jenkovaa/test/H.py(1)?()
  -> from gpaw import GPAW
  (Pdb)


The basic gdb commands are the same as in pdb (or vice versa). Gdb-documentation can be found for example from http://www.gnu.org/software/gdb/documentation/

Emacs can be used also with gdb. Start with *M-x gdb* and then continue as when starting from the command line.
