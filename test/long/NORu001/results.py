"""
E_ad_O = Oad-clean-0.5*moli['O2']
E_ad_N = Nad-clean-0.5*moli['N2']
E_rebond = Oad+Nad-2*clean-moli['NO']
delta = moli['O2']+moli['N2']-moli['NO']

print '%9s %5.2f %14s' % ('E_ad_NO:', E_ad_NO, '(VASP: -4.56)')
print '%9s %5.2f %14s' % ('E_ad_N:', E_ad_N, '(VASP: -0.94)')
print '%9s %5.2f %14s' % ('E_ad_O:', E_ad_O, '(VASP: -2.67)')
print '%9s %5.2f %14s' % ('delta:', delta, '(VASP: -0.95)')
NO@Ru(0001):
This test calculates N and O adsorption energies and a rebond energy, in a
manner geared to J.Phys.: Condens. Matter 18 (2006) 41-54. Instead of
considering a c(2x2) surface cell, as done in the paper, a smaller p(2x2) cell
is considered here, which should not seriously affect the results.
Default values of the function NO_Ru0001 resemble the settings described in the
paper. Only PBE is used instead of PW91.
"""
