==========================================
NEB: Self-diffusion on the Al(110) surface
==========================================

In this exercise, we will find minimum energy paths and transition
states using the "Nudged Elastic Band" method.

Look at the Al(110) surface in the file Al-Al110.traj_ It may help to repeat the structure with the keyword -r: 
e.g. ``ag -r 2,2,1 Al-Al110.traj``. 
The adatom can jump along the rows or across the rows. 

  .. _Al-Al110.traj : attachment: Al-Al110.traj.

* Which of the two jumps do you think will have the largest energy
  barrier?

The template script neb1.py_ will find the minimum energy path for a jump
along the rows.  Read, understand and run the script.

* Make sure you understand what is going on (make a good sketch of the
  110 surface).

* What is the energy barrier?

* Copy the script to ``neb2.py`` and modify it to find the barrier for
  diffusion across one of the rows.  What is the barrier for this
  process?

* Can you think of a third type of diffusion process?  Hint: it is
  called an exchange process.  Find the barrier for this process, and
  compare the energy barrier with the two other ones.


.. _neb1.py : wiki:ASESVN:ase/examples/neb1.py
