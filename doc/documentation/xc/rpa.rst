.. _rpa:

=======================
RPA correlation energy
=======================

.. default-role:: math

The correlation energy within the Random Phase Approximation (RPA) can be written

.. math::

  E_c^{RPA} = \int_0^{\infty}\frac{d\omega}{2\pi}\text{Tr}\Big[\text{ln}\{1-\chi^0(i\omega)v\}+\chi^0(i\omega)v\Big],
 
where `\chi^0(i\omega)` is the non-interacting (Kohn-Sham) response function evaluated at complex frequencies, `\text{Tr}` is the Trace and `\it{v}` is the Coulomb interaction. The response function and Coulomb interaction are evaluted in a plane wave basis as described in :ref:`df_tutorial` and :ref:`df_theory` and for periodic systems the Trace therefore involves a summation over `\mathbf{q}`-points, which are determined from the Brillouin zone sampling used when calculating `\chi^0(i\omega)`. 

The RPA correlation energy is obtained by::
    
    from gpaw.xc.rpa_correlation_energy import RPACorrelation
    rpa = RPACorrelation(calc, txt='rpa_correlation.txt')   
    E_rpa = rpa.get_rpa_correlation_energy(ecut=ecut, w=ws, kcommsize=size)

where calc is a calculator object containing converged wavefunctions from a ground state calculation, txt denotes the output file, ecut (eV) determines the plane wave cutoff used to represent the response function, w is an equidistant array of frequencies (eV) on which the response function is calculated and  kcommsize is the number of k-point domains used for parallelization. Default parallelization is over frequency points, but it is often an advantage to use a combination of frequency and k-point parallelization. For example, having a system with 20 irreducible k-points and a list with 32 frequency points, one could use kcommsize=4 on 32 processors, which would give 4 frequency points to each processor, which then belongs to one of the 4 k-point domains. 

A major complication with the RPA correlation energy is that it converges very slowly with the number of unoccupied bands included in the evaluation of `\chi^0(i\omega)`. However, as described in Ref. \ [#Harl]_ the high energy part of the response function resembles the Lindhard function, which for high energies gives a correlation energy converging as

.. math::

  E_c^{Lindhard}(G^{\chi}_{cut}) = E_c^{\infty}+\frac{A}{(G^{\chi}_{cut})^3},

where `G^{\chi}_{cut}` is the largest reciprocal lattice vector used in the evaluation of `\chi^0`. With an external potential, the number of unoccupied bands is an additional convergence parameter, but for reproducing the scaling of the Lindhard function, it is natural to set the total number of bands equal to the number of plane waves used. Thus, to obtain a converged RPA correlation energy one should proceed in three steps.

* Perform a ground state calculation with a lot of converged unoccupied bands.
  
* Define a list of cutoff energies - typically something like [200, 225, 250, 275, 300] (eV). For each cutoff energy perform an RPA correlation energy calculation with the number bands `n` set equal to the number of plane waves defined by that cutoff energy. 

* Use that `n\propto (G^{\chi}_{cut})^3` and fit the list of obtained correlation energies to `E_c^{RPA}(n) = E_c^{\infty}+A/n` to obtain `E_c^{\infty}=E_c^{RPA}`.

It is possible to specify the number of bands used to calculate the response function by the keyword "nbands" in get_rpa_correlation_energy(). The default value of nbands is the number of bands, which correspond to ecut and is saved in the variable rpa.nbands after the calculation. This value needs to be smaller than the total number of (converged) bands from the ground state calculation. 

Below is shown an example of a calculation of the correlation part of the atomization energy of an N2 molecule represented in a unit cell of side length 6 Å. 

.. image:: E_rpa.png
	   :height: 400 px

The calculated points were generated with the script :svn:`~doc/documentation/xc/rpa_n2.py`, where N2_4000.gpw and N_4000.gpw contain ground state calculations with 4000 converged bands of N2 and N respectively. The number of bands used for the six points correspond to cutoff energies of 150, 200, 250, 300, 350 and 400 eV and the green line is the best fit to `E(n)=E_c+A/n` using the last 4 points. The script takes 16 hours on 32 Intel Xeon X5570 2.93GHz CPUs. 

The extrapolated value at infinity of 4.98 eV exactly match the value found in Ref. \ [#Harl]_. However, it should be noted that since the RPA functional contains long range correlation effects, one needs to carefully converge the result with respect to the unit cell volume. Typically, van der Waals interaction behave as `V^{-2}` and one can extrapolate to infinite volume using a few different super cells.

.. [#Harl] J. Harl and G. Kresse,
           *Phys. Rev. B* **77**, 045136 (2008)
