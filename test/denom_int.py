# This test takes approximately 0.0 seconds
"""
Here is a simple test program, which tests for radial integration
of Function1D class.
"""

from gpaw.Function1D import Function1D
import Numeric as num

# Use 150 radial points to integrate
N = 150

# Create the radial grid    
# Copy-pasted from all_electron.py
beta = 0.4
g = num.arange(N, typecode=num.Float)
r = beta * g / (N - g)
dr = beta * N / (N - g)**2

# Fix division by zero (Grr!)
r[0] = r[1]

""" The radial parts are from Gasiorowicz, Quantum Physics, 2nd edition, p. 208 """

print "Checking that norm of Hydrogen atom 1s-state is 1:"
H1s = Function1D(0, 0, 2 * num.exp(-r))
result = (H1s * H1s).integrateRY(r, dr)
print result
assert abs(result-1.0) <1e-6

print "Checking that norm of Hydrogen atom 2p-state is 1:"
H2p = Function1D(1,0, ( 1.0 / num.sqrt(3) * (1.0/2.0)**(3.0/2.0) * r * num.exp(-r/2)))
result = (H2p * H2p).integrateRY(r, dr)
print result
assert abs(result-1.0) <1e-6

print "Checking that the cross-density integrates to zero:"
result = (H2p * H1s).integrateRY(r, dr)
print result
assert abs(result-0.0) <1e-6

# Testing for the integration with fraction of spherical harmonics

n1s = H1s * H1s
n2p = H2p * H2p
nom = (H1s*H1s + (H2p*H2p).scalar_mul(0.5)) * H1s*H1s
denom = H1s*H1s + H2p*H2p

print "Checking for certain denominator integration... (result from coarse Matlab integration was: 0.93000"
result = nom.integrate_with_denominator(denom, r, dr)
print result
assert abs(result-0.93000) <1e-2

result = (H1s*H1s*H1s*H1s).integrate_with_denominator(H1s*H1s, r, dr)
print "Norm of H1s with denominator...:"
print result

print "The hartree-energy of hydrogen 1s. Should be 5/16Ha: "
result = 0.5*((H1s*H1s).solve_poisson(r, dr, beta, N) * H1s*H1s).integrateRY(r, dr)
print result
assert abs(result-0.3125) <1e-4

"""
%Here is a simple Matlab code which tests for these same integrals
%Note that N=300 consumes over 2Gb of memory

N = 300
k = linspace(-15,15,N);

[x,y,z] = ndgrid(k,k,k);

r = sqrt(x.^2+y.^2+z.^2);

Y00 = 1/sqrt(4*pi);
Y10 = sqrt(3/(4*pi)) * z ./ r;

H1s = 2 * exp(-r) * Y00;
H2p = 1.0 / sqrt(3) * (1.0/2.0)^(3.0/2.0) * r .* exp(-r/2) .* Y10;

dr = (k(2)-k(1))^3;
I = sum(sum(sum(H1s .* H1s))) * dr
I = sum(sum(sum(H2p .* H2p))) * dr

nom = (H1s .* H1s + 0.5 * H2p.*H2p) .* H1s.*H1s;
denom = H1s.*H1s + H2p.*H2p;

I = sum(sum(sum(nom./denom))) * dr

% Results:
% 0.99999599090205
% 0.99981623497608
% 0.93000397523946

"""
