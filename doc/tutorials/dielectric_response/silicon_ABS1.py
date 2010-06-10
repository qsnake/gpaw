import numpy as np
from gpaw import GPAW
from gpaw.response.df import DF

w = np.linspace(0, 24., 481)    # 0-24 eV with 0.05 eV spacing
q = np.array([0.0, 0.00001, 0.])

df = DF(calc='si.gpw',
        q=q,
        w=w,
        eta=0.2,           # Broadening parameter 
        ecut=150,          # Energy cutoff for planewaves
        optical_limit=True,
        txt='df_2.out')    # Output text

df1, df2 = df.get_dielectric_function()
df.get_absorption_spectrum(df1, df2, filename='si_abs.dat')
df.check_sum_rule(df1, df2)
df.write('df_2.pckl')      # Save important parameters and data 
