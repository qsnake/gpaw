"""
The following contains the G2-1 and G2-2 neutral test sets.

* G2-1 neutral test set: 55 molecules
* G2-2 neutral test set: 93 molecules
* MP2(full)/6-31G(d) optimized geometries

Reference:
'Assessment of Gaussian-2 and Density Functional Theories for the
Computation of Enthalpies of Formation' by Larry A. Curtiss, 
Krishnan Raghavachari, Paul Redfern, and John A. Pople, 
J. Chem. Phys. Vol. 106, 1063 (1997).
"""
from ASE import  ListOfAtoms, Atom
from gpaw.utilities import center

# The magnetic moments of the 14 atoms involved in the g2 molecules:
atoms = {'H': 1, 'Li': 1, 'Be': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 2, 'F': 1,
         'Na': 1, 'Al': 1, 'Si': 2, 'P': 3, 'S': 2,'Cl': 1}

# The 148 molecules of the test set
molecules = {
     'Cl4C2': "C2Cl4 (Cl2C=CCl2), D2h symm.",
      'H2O2': "Hydrogen peroxide (HO-OH), C2 symm.",
    'ClC2H3': "Vinyl chloride, H2C=CHCl, Cs symm.",
    'ClC2H5': "Ethyl chloride (CH3-CH2-Cl), Cs symm.",
      'H2CO': "Formaldehyde (H2C=O), C2v symm.",
        'O3': "O3 (Ozone), C2v symm.",
        'O2': "O2 molecule, D*h symm, Triplet.",
       'F3B': "BF3, Planar D3h symm.",
      'F4C2': "C2F4 (F2C=CF2), D2H symm.",
       'BeH': "Beryllium hydride (BeH), D*h symm.",
       'F3N': "NF3, C3v symm.",
       'HCO': "HCO radical, Bent Cs symm.",
       'OF2': "F2O, C2v symm.",
       'Cl2': "Cl2 molecule, D*h symm.",
    'OFC2H3': "Acetyl fluoride (CH3COF), HCCO cis, Cs symm.",
    'SOC2H6': "Dimethylsulfoxide (CH3)2SO, Cs symm.",
        'SH': "SH radical, C*v symm.",
    'H10C4x': "Isobutane (C4H10), C3v symm.",
      'C2N2': "Cyanogen (NCCN). D*h symm.",
       'O2N': "NO2 radical, C2v symm, 2-A1.",
    'O2H4C2': "Methyl formate (HCOOCH3), Cs symm.",
      'OH3C': "H2COH radical, C1 symm.",
    'O2H2C2': "Glyoxal (O=CH-CH=O). Trans, C2h symm.",
       'CO2': "Carbon dioxide (CO2), D*h symm.",
       'H2O': "Water (H2O), C2v symm.",
     'H5C5N': "Pyridine (cyclic C5H5N), C2v symm.",
        'NO': "NO radical, C*v symm, 2-Pi.",
      'H4C3': "Propyne (C3H4), C3v symm.",
      'C2H4': "Ethylene (H2C=CH2), D2h symm.",
        'OH': "OH radical, C*v symm.",
    'Cl2CH2': "Dichloromethane (H2CCl2), C2v symm.",
      'H6C6': "Benzene (C6H6), D6h symm.",
     'Cl4Si': "SiCl4, Td symm.",
      'H6C4': "Trans-1,3-butadiene (C4H6), C2h symm.",
     'H6C4x': "Dimethylacetylene (2-butyne, C4H6), D3h symm (eclipsed).",
    'H6C4xx': "Methylenecyclopropane (C4H6), C2v symm.",
   'H6C4xxx': "Bicyclo[1.1.0]butane (C4H6), C2v symm.",
  'H6C4xxxx': "Cyclobutene (C4H6), C2v symm.",
      'H6C3': "Propene (C3H6), Cs symm.",
      'C2H6': "Ethane (H3C-CH3), D3d symm.",
      'HCF3': "Trifluoromethane (HCF3), C3v symm.",
     'HCCl3': "Chloroform (HCCl3), C3v symm.",
        'SO': "Sulfur monoxide (SO), C*v symm, triplet.",
       'CH3': "Methyl radical (CH3), D3h symm.",
        'CO': "Carbon monoxide (CO), C*v symm.",
      'F3Al': "AlF3, Planar D3h symm.",
       'SiO': "Silicon monoxide (SiO), C*v symm.",
        'NH': "NH, triplet, C*v symm.",
      'Cl4C': "CCl4, Td symm.",
     'H10C4': "Trans-butane (C4H10), C2h symm.",
        'CH': "CH radical. Doublet, C*v symm.",
    'H4C3xx': "Cyclopropene (C3H4), C2v symm.",
       'SH2': "Hydrogen sulfide (H2S), C2v symm.",
       'ClO': "ClO radical, C*v symm, 2-PI.",
    'H7C2Nx': "Trans-Ethylamine (CH3-CH2-NH2), Cs symm.",
    'OH4C2x': "Acetaldehyde (CH3CHO), Cs symm.",
       'LiF': "Lithium Fluoride (LiF), C*v symm.",
     'SH6C2': "ThioEthanol (CH3-CH2-SH), Cs symm.",
    'OH5C2N': "Acetamide (CH3CONH2), C1 symm.",
     'OH3Cx': "CH3O radical, Cs symm, 2-A'.",
     'OH3C2': "CH3CO radical, HCCO cis, Cs symm, 2-A'.",
       'OSC': "O=C=S, Linear, C*v symm.",
        'P2': "P2 molecule, D*h symm.",
     'O2H2C': "Formic Acid (HCOOH), HOCO cis, Cs symm.",
     'H4C3x': "Allene (C3H4), D2d symm.",
       'LiH': "Lithium hydride (LiH), C*v symm.",
     'OH8C3': "Isopropyl alcohol, (CH3)2CH-OH, Gauche isomer, C1 symm.",
       'HCN': "Hydrogen cyanide (HCN), C*v symm.",
      'H9C4': "t-Butyl radical, (CH3)3C, C3v symm.",
      'SiH4': "Silane (SiH4), Td symm.",
    'H7C3Cl': "Propyl chloride (CH3CH2CH2Cl), Cs symm.",
     'OH4C2': "Oxirane (cyclic CH2-O-CH2 ring), C2v symm.",
     'OH4C4': "Furan (cyclic C4H4O), C2v symm.",
        'S2': "S2 molecule, D*h symm, triplet.",
     'OH2C2': "Ketene (H2C=C=O), C2v symm.",
     'Si2H6': "Disilane (H3Si-SiH3), D3d symm.",
     'F2CH2': "Difluoromethane (H2CF2), C2v symm.",
     'H9C3N': "Trimethyl Amine, (CH3)3N, C3v symm.",
        'HF': "Hydrogen fluoride (HF), C*v symm.",
       'ClF': "ClF molecule, C*v symm, 1-SG.",
      'OF2C': "COF2, C2v symm.",
       'F3P': "PF3, C3v symm.",
       'HCl': "Hydrogen chloride (HCl), C*v symm.",
      'N2H4': "Hydrazine (H2N-NH2), C2 symm.",
     'H8C4x': "Isobutene (C4H8), Single bonds trans, C2v symm.",
     'OH6C3': "Acetone (CH3-CO-CH3), C2v symm.",
     'OH6C2': "Ethanol (trans, CH3CH2OH), Cs symm.",
     'H5C2N': "Aziridine (cyclic CH2-NH-CH2 ring), C2v symm.",
'SiH2_s1A1d': "Singlet silylene (SiH2), C2v symm, 1-A1.",
'SiH2_s3B1d': "Triplet silylene (SiH2), C2v symm, 3-B1.",
      'OClN': "ClNO, Cs symm.",
     'F3C2N': "CF3CN, C3v symm.",
     'H6C3x': "Cyclopropane (C3H6), D3h symm.",
       'HC2': "CCH radical, C*v symm.",
     'CH3Cl': "Methyl chloride (CH3Cl), C3v symm.",
     'SH4C4': "Thiophene (cyclic C4H4S), C2v symm.",
     'H3C3N': "CyanoEthylene (H2C=CHCN), Cs symm.",
   'O2H3CNx': "Methylnitrite (CH3-O-N=O), NOCH trans, ONOC cis, Cs symm.",
     'Cl3Al': "AlCl3, Planar D3h symm.",
      'H8C5': "Spiropentane (C5H8), D2d symm.",
      'H8C4': "Cyclobutane (C4H8), D2d symm.",
        'N2': "N2 molecule, D*h symm.",
      'H8C3': "Propane (C3H8), C2v symm.",
     'H6SiC': "Methylsilane (CH3-SiH3), C3v symm.",
     'OH5C2': "CH3CH2O radical, Cs symm, 2-A''.",
        'F2': "F2 molecule, D*h symm.",
       'PH2': "PH2 radical, C2v symm.",
        'H2': "H2. D*h symm.",
       'NH2': "NH2 radical, C2v symm, 2-B1.",
        'CS': "Carbon monosulfide (CS), C*v symm.",
 'CH2_s1A1d': "Singlet methylene (CH2), C2v symm, 1-A1.",
 'CH2_s3B1d': "Triplet methylene (CH2), C2v symm, 3-B1.",
    'OH6C2x': "DimethylEther (CH3-O-CH3), C2v symm.",
      'C2H2': "Acetylene (C2H2), D*h symm.",
   'O2H4C2x': "Acetic Acid (CH3COOH), Single bonds trans, Cs symm.",
       'CH4': "Methane (CH4), Td symm.",
      'H5CN': "Methylamine (H3C-NH2), Cs symm.",
     'H3C2N': "Acetonitrile (CH3-CN), C3v symm.",
      'ClF3': "ClF3, C2v symm.",
      'F4Si': "SiF4, Td symm.",
     'CH3OH': "Methanol (CH3-OH), Cs symm.",
     'FC2H3': "Vinyl fluoride (H2C=CHF), Cs symm.",
      'SiH3': "Silyl radical (SiH3), C3v symm.",
    'SH6C2x': "Dimethyl ThioEther (CH3-S-CH3), C2v symm.",
        'CN': "Cyano radical (CN), C*v symm, 2-Sigma+.",
       'S2C': "CS2, Linear, D*h symm.",
     'H5C4N': "Pyrrole (Planar cyclic C4H4NH), C2v symm.",
      'SH3C': "CH3S radical, Cs symm, 2-A'.",
     'CH3SH': "Methanethiol (H3C-SH), Staggered, Cs symm.",
    'O2H3CN': "Nitromethane (CH3-NO2), Cs symm.",
       'Si2': "Si2 molecule, D*h symm, Triplet (3-Sigma-G-).",
     'H7C2N': "Dimethylamine, (CH3)2NH, Cs symm.",
      'NaCl': "Sodium Chloride (NaCl), C*v symm.",
      'H7C3': "(CH3)2CH radical, Cs symm, 2-A'.",
       'NH3': "Ammonia (NH3), C3v symm.",
       'Na2': "Disodium (Na2), D*h symm.",
       'PH3': "Phosphine (PH3), C3v symm.",
       'SO2': "Sulfur dioxide (SO2), C2v symm.",
       'Li2': "Dilithium (Li2), D*h symm.",
       'ON2': "N2O, Cs symm.",
    'OH8C3x': "Methyl ethyl ether (CH3-CH2-O-CH3), Trans, Cs symm.",
      'HOCl': "HOCl molecule, Cs symm.",
      'Cl3B': "BCl3, Planar D3h symm.",
      'H3C2': "C2H3 radical, Cs symm, 2-A'.",
       'F4C': "CF4, Td symm.",
   'OClC2H3': "Acetyl,Chloride (CH3COCl), HCCO cis, Cs symm.",
      'H5C2': "C2H5 radical, Staggered, Cs symm, 2-A'.",
     'SH4C2': "Thiooxirane (cyclic CH2-S-CH2 ring), C2v symm.",
}

# Lithium hydride (LiH), C*v symm.
# MP2 energy = -7.9965108 Hartree
# Charge = 0, multiplicity = 1
LiH = ListOfAtoms([
    Atom('Li', [.000000, .000000, .410000]),
    Atom('H', [.000000, .000000, -1.230000]),
    ])

# Beryllium hydride (BeH), D*h symm.
# MP2 energy = -15.171409 Hartree
# Charge = 0, multiplicity = 2
BeH = ListOfAtoms([
    Atom('Be', [.000000, .000000, .269654]),
    Atom('H', [.000000, .000000, -1.078616]),
    ])

# CH radical. Doublet, C*v symm.
# MP2 energy = -38.3423986 Hartree
# Charge = 0, multiplicity = 2
CH = ListOfAtoms([
    Atom('C', [.000000, .000000, .160074]),
    Atom('H', [.000000, .000000, -.960446]),
    ])

# Triplet methylene (CH2), C2v symm, 3-B1.
# MP2 energy = -39.0074352 Hartree
# Charge = 0, multiplicity = 3
CH2_s3B1d = ListOfAtoms([
    Atom('C', [.000000, .000000, .110381]),
    Atom('H', [.000000, .982622, -.331142]),
    Atom('H', [.000000, -.982622, -.331142]),
    ])

# Singlet methylene (CH2), C2v symm, 1-A1.
# MP2 energy = -38.9740078 Hartree
# Charge = 0, multiplicity = 1
CH2_s1A1d = ListOfAtoms([
    Atom('C', [.000000, .000000, .174343]),
    Atom('H', [.000000, .862232, -.523029]),
    Atom('H', [.000000, -.862232, -.523029]),
    ])

# Methyl radical (CH3), D3h symm.
# MP2 energy = -39.6730312 Hartree
# Charge = 0, multiplicity = 2
CH3 = ListOfAtoms([
    Atom('C', [.000000, .000000, .000000]),
    Atom('H', [.000000, 1.078410, .000000]),
    Atom('H', [.933930, -.539205, .000000]),
    Atom('H', [-.933930, -.539205, .000000]),
    ])

# Methane (CH4), Td symm.
# MP2 energy = -40.3370426 Hartree
# Charge = 0, multiplicity = 1
CH4 = ListOfAtoms([
    Atom('C', [.000000, .000000, .000000]),
    Atom('H', [.629118, .629118, .629118]),
    Atom('H', [-.629118, -.629118, .629118]),
    Atom('H', [.629118, -.629118, -.629118]),
    Atom('H', [-.629118, .629118, -.629118]),
    ])

# NH, triplet, C*v symm.
# MP2 energy = -55.0614242 Hartree
# Charge = 0, multiplicity = 3
NH = ListOfAtoms([
    Atom('N', [.000000, .000000, .129929], magmom=0.883),
    Atom('H', [.000000, .000000, -.909501], magmom=-0.010),
    ])

# NH2 radical, C2v symm, 2-B1.
# MP2 energy = -55.6937452 Hartree
# Charge = 0, multiplicity = 2
NH2 = ListOfAtoms([
    Atom('N', [.000000, .000000, .141690]),
    Atom('H', [.000000, .806442, -.495913]),
    Atom('H', [.000000, -.806442, -.495913]),
    ])

# Ammonia (NH3), C3v symm.
# MP2 energy = -56.3573777 Hartree
# Charge = 0, multiplicity = 1
NH3 = ListOfAtoms([
    Atom('N', [.000000, .000000, .116489]),
    Atom('H', [.000000, .939731, -.271808]),
    Atom('H', [.813831, -.469865, -.271808]),
    Atom('H', [-.813831, -.469865, -.271808]),
    ])

# OH radical, C*v symm.
# MP2 energy = -75.5232063 Hartree
# Charge = 0, multiplicity = 2
OH = ListOfAtoms([
    Atom('O', [.000000, .000000, .108786], magmom=0.5),
    Atom('H', [.000000, .000000, -.870284], magmom=0.5),
    ])

# Water (H2O), C2v symm.
# MP2 energy = -76.1992442 Hartree
# Charge = 0, multiplicity = 1
H2O = ListOfAtoms([
    Atom('O', [.000000, .000000, .119262]),
    Atom('H', [.000000, .763239, -.477047]),
    Atom('H', [.000000, -.763239, -.477047]),
    ])

# Hydrogen fluoride (HF), C*v symm.
# MP2 energy = -100.1841614 Hartree
# Charge = 0, multiplicity = 1
HF = ListOfAtoms([
    Atom('F', [.000000, .000000, .093389]),
    Atom('H', [.000000, .000000, -.840502]),
    ])

# Singlet silylene (SiH2), C2v symm, 1-A1.
# MP2 energy = -290.0772034 Hartree
# Charge = 0, multiplicity = 1
SiH2_s1A1d = ListOfAtoms([
    Atom('Si', [.000000, .000000, .131272]),
    Atom('H', [.000000, 1.096938, -.918905]),
    Atom('H', [.000000, -1.096938, -.918905]),
    ])

# Triplet silylene (SiH2), C2v symm, 3-B1.
# MP2 energy = -290.0561783 Hartree
# Charge = 0, multiplicity = 3
SiH2_s3B1d = ListOfAtoms([
    Atom('Si', [.000000, .000000, .094869]),
    Atom('H', [.000000, 1.271862, -.664083]),
    Atom('H', [.000000, -1.271862, -.664083]),
    ])

# Silyl radical (SiH3), C3v symm.
# MP2 energy = -290.6841563 Hartree
# Charge = 0, multiplicity = 2
SiH3 = ListOfAtoms([
    Atom('Si', [.000000, .000000, .079299]),
    Atom('H', [.000000, 1.413280, -.370061]),
    Atom('H', [1.223937, -.706640, -.370061]),
    Atom('H', [-1.223937, -.706640, -.370061]),
    ])

# Silane (SiH4), Td symm.
# MP2 energy = -291.3168497 Hartree
# Charge = 0, multiplicity = 1
SiH4 = ListOfAtoms([
    Atom('Si', [.000000, .000000, .000000]),
    Atom('H', [.856135, .856135, .856135]),
    Atom('H', [-.856135, -.856135, .856135]),
    Atom('H', [-.856135, .856135, -.856135]),
    Atom('H', [.856135, -.856135, -.856135]),
    ])

# PH2 radical, C2v symm.
# MP2 energy = -341.9457892 Hartree
# Charge = 0, multiplicity = 2
PH2 = ListOfAtoms([
    Atom('P', [.000000, .000000, .115396]),
    Atom('H', [.000000, 1.025642, -.865468]),
    Atom('H', [.000000, -1.025642, -.865468]),
    ])

# Phosphine (PH3), C3v symm.
# MP2 energy = -342.562259 Hartree
# Charge = 0, multiplicity = 1
PH3 = ListOfAtoms([
    Atom('P', [.000000, .000000, .124619]),
    Atom('H', [.000000, 1.200647, -.623095]),
    Atom('H', [1.039791, -.600323, -.623095]),
    Atom('H', [-1.039791, -.600323, -.623095]),
    ])

# Hydrogen sulfide (H2S), C2v symm.
# MP2 energy = -398.7986975 Hartree
# Charge = 0, multiplicity = 1
SH2 = ListOfAtoms([
    Atom('S', [.000000, .000000, .102135]),
    Atom('H', [.000000, .974269, -.817083]),
    Atom('H', [.000000, -.974269, -.817083]),
    ])

# Hydrogen chloride (HCl), C*v symm.
# MP2 energy = -460.2021493 Hartree
# Charge = 0, multiplicity = 1
HCl = ListOfAtoms([
    Atom('Cl', [.000000, .000000, .071110]),
    Atom('H', [.000000, .000000, -1.208868]),
    ])

# Dilithium (Li2), D*h symm.
# MP2 energy = -14.8868485 Hartree
# Charge = 0, multiplicity = 1
Li2 = ListOfAtoms([
    Atom('Li', [.000000, .000000, 1.386530]),
    Atom('Li', [.000000, .000000, -1.386530]),
    ])

# Lithium Fluoride (LiF), C*v symm.
# MP2 energy = -107.1294652 Hartree
# Charge = 0, multiplicity = 1
LiF = ListOfAtoms([
    Atom('Li', [.000000, .000000, -1.174965]),
    Atom('F', [.000000, .000000, .391655]),
    ])

# Acetylene (C2H2), D*h symm.
# MP2 energy = -77.0762154 Hartree
# Charge = 0, multiplicity = 1
C2H2 = ListOfAtoms([
    Atom('C', [.000000, .000000, .608080]),
    Atom('C', [.000000, .000000, -.608080]),
    Atom('H', [.000000, .000000, -1.673990]),
    Atom('H', [.000000, .000000, 1.673990]),
    ])

# Ethylene (H2C=CH2), D2h symm.
# MP2 energy = -78.2942862 Hartree
# Charge = 0, multiplicity = 1
C2H4 = ListOfAtoms([
    Atom('C', [.000000, .000000, .667480]),
    Atom('C', [.000000, .000000, -.667480]),
    Atom('H', [.000000, .922832, 1.237695]),
    Atom('H', [.000000, -.922832, 1.237695]),
    Atom('H', [.000000, .922832, -1.237695]),
    Atom('H', [.000000, -.922832, -1.237695]),
    ])

# Ethane (H3C-CH3), D3d symm.
# MP2 energy = -79.5039697 Hartree
# Charge = 0, multiplicity = 1
C2H6 = ListOfAtoms([
    Atom('C', [.000000, .000000, .762209]),
    Atom('C', [.000000, .000000, -.762209]),
    Atom('H', [.000000, 1.018957, 1.157229]),
    Atom('H', [-.882443, -.509479, 1.157229]),
    Atom('H', [.882443, -.509479, 1.157229]),
    Atom('H', [.000000, -1.018957, -1.157229]),
    Atom('H', [-.882443, .509479, -1.157229]),
    Atom('H', [.882443, .509479, -1.157229]),
    ])

# Cyano radical (CN), C*v symm, 2-Sigma+.
# MP2 energy = -92.441963 Hartree
# Charge = 0, multiplicity = 2
CN = ListOfAtoms([
    Atom('C', [.000000, .000000, -.611046]),
    Atom('N', [.000000, .000000, .523753]),
    ])

# Hydrogen cyanide (HCN), C*v symm.
# MP2 energy = -93.1669402 Hartree
# Charge = 0, multiplicity = 1
HCN = ListOfAtoms([
    Atom('C', [.000000, .000000, -.511747]),
    Atom('N', [.000000, .000000, .664461]),
    Atom('H', [.000000, .000000, -1.580746]),
    ])

# Carbon monoxide (CO), C*v symm.
# MP2 energy = -113.0281795 Hartree
# Charge = 0, multiplicity = 1
CO = ListOfAtoms([
    Atom('O', [.000000, .000000, .493003]),
    Atom('C', [.000000, .000000, -.657337]),
    ])

# HCO radical, Bent Cs symm.
# MP2 energy = -113.540332 Hartree
# Charge = 0, multiplicity = 2
HCO = ListOfAtoms([
    Atom('C', [.062560, .593926, .000000]),
    Atom('O', [.062560, -.596914, .000000]),
    Atom('H', [-.875835, 1.211755, .000000]),
    ])

# Formaldehyde (H2C=O), C2v symm.
# MP2 energy = -114.1749578 Hartree
# Charge = 0, multiplicity = 1
H2CO = ListOfAtoms([
    Atom('O', [.000000, .000000, .683501]),
    Atom('C', [.000000, .000000, -.536614]),
    Atom('H', [.000000, .934390, -1.124164]),
    Atom('H', [.000000, -.934390, -1.124164]),
    ])

# Methanol (CH3-OH), Cs symm.
# MP2 energy = -115.3532948 Hartree
# Charge = 0, multiplicity = 1
CH3OH = ListOfAtoms([
    Atom('C', [-.047131, .664389, .000000]),
    Atom('O', [-.047131, -.758551, .000000]),
    Atom('H', [-1.092995, .969785, .000000]),
    Atom('H', [.878534, -1.048458, .000000]),
    Atom('H', [.437145, 1.080376, .891772]),
    Atom('H', [.437145, 1.080376, -.891772]),
    ])

# N2 molecule, D*h symm.
# MP2 energy = -109.2615742 Hartree
# Charge = 0, multiplicity = 1
N2 = ListOfAtoms([
    Atom('N', [.000000, .000000, .564990]),
    Atom('N', [.000000, .000000, -.564990]),
    ])

# Hydrazine (H2N-NH2), C2 symm.
# MP2 energy = -111.5043953 Hartree
# Charge = 0, multiplicity = 1
N2H4 = ListOfAtoms([
    Atom('N', [.000000, .718959, -.077687]),
    Atom('N', [.000000, -.718959, -.077687]),
    Atom('H', [.211082, 1.092752, .847887]),
    Atom('H', [-.948214, 1.005026, -.304078]),
    Atom('H', [-.211082, -1.092752, .847887]),
    Atom('H', [.948214, -1.005026, -.304078]),
    ])

# NO radical, C*v symm, 2-Pi.
# MP2 energy = -129.564464 Hartree
# Charge = 0, multiplicity = 2
NO = ListOfAtoms([
    Atom('N', [.000000, .000000, -.609442], magmom=0.6),
    Atom('O', [.000000, .000000, .533261], magmom=0.4),
    ])

# O2 molecule, D*h symm, Triplet.
# MP2 energy = -149.9543197 Hartree
# Charge = 0, multiplicity = 3
O2 = ListOfAtoms([
    Atom('O', [.000000, .000000, .622978], magmom=1.),
    Atom('O', [.000000, .000000, -.622978], magmom=1.),
    ])

# Hydrogen peroxide (HO-OH), C2 symm.
# MP2 energy = -151.1349184 Hartree
# Charge = 0, multiplicity = 1
H2O2 = ListOfAtoms([
    Atom('O', [.000000, .734058, -.052750]),
    Atom('O', [.000000, -.734058, -.052750]),
    Atom('H', [.839547, .880752, .422001]),
    Atom('H', [-.839547, -.880752, .422001]),
    ])

# F2 molecule, D*h symm.
# MP2 energy = -199.0388236 Hartree
# Charge = 0, multiplicity = 1
F2 = ListOfAtoms([
    Atom('F', [.000000, .000000, .710304]),
    Atom('F', [.000000, .000000, -.710304]),
    ])

# Carbon dioxide (CO2), D*h symm.
# MP2 energy = -188.1183633 Hartree
# Charge = 0, multiplicity = 1
CO2 = ListOfAtoms([
    Atom('C', [.000000, .000000, .000000]),
    Atom('O', [.000000, .000000, 1.178658]),
    Atom('O', [.000000, .000000, -1.178658]),
    ])

# Disodium (Na2), D*h symm.
# MP2 energy = -323.7039996 Hartree
# Charge = 0, multiplicity = 1
Na2 = ListOfAtoms([
    Atom('Na', [.000000, .000000, 1.576262]),
    Atom('Na', [.000000, .000000, -1.576262]),
    ])

# Si2 molecule, D*h symm, Triplet (3-Sigma-G-).
# MP2 energy = -577.8606556 Hartree
# Charge = 0, multiplicity = 3
Si2 = ListOfAtoms([
    Atom('Si', [.000000, .000000, 1.130054]),
    Atom('Si', [.000000, .000000, -1.130054]),
    ])

# P2 molecule, D*h symm.
# MP2 energy = -681.6646966 Hartree
# Charge = 0, multiplicity = 1
P2 = ListOfAtoms([
    Atom('P', [.000000, .000000, .966144]),
    Atom('P', [.000000, .000000, -.966144]),
    ])

# S2 molecule, D*h symm, triplet.
# MP2 energy = -795.2628131 Hartree
# Charge = 0, multiplicity = 3
S2 = ListOfAtoms([
    Atom('S', [.000000, .000000, .960113], magmom=0.558),
    Atom('S', [.000000, .000000, -.960113], magmom=0.558),
    ])

# Cl2 molecule, D*h symm.
# MP2 energy = -919.191224 Hartree
# Charge = 0, multiplicity = 1
Cl2 = ListOfAtoms([
    Atom('Cl', [.000000, .000000, 1.007541]),
    Atom('Cl', [.000000, .000000, -1.007541]),
    ])

# Sodium Chloride (NaCl), C*v symm.
# MP2 energy = -621.5463469 Hartree
# Charge = 0, multiplicity = 1
NaCl = ListOfAtoms([
    Atom('Na', [.000000, .000000, -1.451660]),
    Atom('Cl', [.000000, .000000, .939310]),
    ])

# Silicon monoxide (SiO), C*v symm.
# MP2 energy = -364.0594076 Hartree
# Charge = 0, multiplicity = 1
SiO = ListOfAtoms([
    Atom('Si', [.000000, .000000, .560846]),
    Atom('O', [.000000, .000000, -.981480]),
    ])

# Carbon monosulfide (CS), C*v symm.
# MP2 energy = -435.5576809 Hartree
# Charge = 0, multiplicity = 1
CS = ListOfAtoms([
    Atom('C', [.000000, .000000, -1.123382]),
    Atom('S', [.000000, .000000, .421268]),
    ])

# Sulfur monoxide (SO), C*v symm, triplet.
# MP2 energy = -472.6266876 Hartree
# Charge = 0, multiplicity = 3
SO = ListOfAtoms([
    Atom('O', [.000000, .000000, -1.015992], magmom=.491),
    Atom('S', [.000000, .000000, .507996], magmom=.709),
    ])

# ClO radical, C*v symm, 2-PI.
# MP2 energy = -534.5186484 Hartree
# Charge = 0, multiplicity = 2
ClO = ListOfAtoms([
    Atom('Cl', [.000000, .000000, .514172], magmom=1.), # XXXX ??
    Atom('O', [.000000, .000000, -1.092615], magmom=0), # XXXX ??
    ])

# ClF molecule, C*v symm, 1-SG.
# MP2 energy = -559.1392996 Hartree
# Charge = 0, multiplicity = 1
ClF = ListOfAtoms([
    Atom('F', [.000000, .000000, -1.084794]),
    Atom('Cl', [.000000, .000000, .574302]),
    ])

# Disilane (H3Si-SiH3), D3d symm.
# MP2 energy = -581.4851067 Hartree
# Charge = 0, multiplicity = 1
Si2H6 = ListOfAtoms([
    Atom('Si', [.000000, .000000, 1.167683]),
    Atom('Si', [.000000, .000000, -1.167683]),
    Atom('H', [.000000, 1.393286, 1.686020]),
    Atom('H', [-1.206621, -.696643, 1.686020]),
    Atom('H', [1.206621, -.696643, 1.686020]),
    Atom('H', [.000000, -1.393286, -1.686020]),
    Atom('H', [-1.206621, .696643, -1.686020]),
    Atom('H', [1.206621, .696643, -1.686020]),
    ])

# Methyl chloride (CH3Cl), C3v symm.
# MP2 energy = -499.3690844 Hartree
# Charge = 0, multiplicity = 1
CH3Cl = ListOfAtoms([
    Atom('C', [.000000, .000000, -1.121389]),
    Atom('Cl', [.000000, .000000, .655951]),
    Atom('H', [.000000, 1.029318, -1.474280]),
    Atom('H', [.891415, -.514659, -1.474280]),
    Atom('H', [-.891415, -.514659, -1.474280]),
    ])

# Methanethiol (H3C-SH), Staggered, Cs symm.
# MP2 energy = -437.9678831 Hartree
# Charge = 0, multiplicity = 1
CH3SH = ListOfAtoms([
    Atom('C', [-.047953, 1.149519, .000000]),
    Atom('S', [-.047953, -.664856, .000000]),
    Atom('H', [1.283076, -.823249, .000000]),
    Atom('H', [-1.092601, 1.461428, .000000]),
    Atom('H', [.432249, 1.551207, .892259]),
    Atom('H', [.432249, 1.551207, -.892259]),
    ])

# HOCl molecule, Cs symm.
# MP2 energy = -535.1694444 Hartree
# Charge = 0, multiplicity = 1
HOCl = ListOfAtoms([
    Atom('O', [.036702, 1.113517, .000000]),
    Atom('H', [-.917548, 1.328879, .000000]),
    Atom('Cl', [.036702, -.602177, .000000]),
    ])

# Sulfur dioxide (SO2), C2v symm.
# MP2 energy = -547.700099 Hartree
# Charge = 0, multiplicity = 1
SO2 = ListOfAtoms([
    Atom('S', [.000000, .000000, .370268]),
    Atom('O', [.000000, 1.277617, -.370268]),
    Atom('O', [.000000, -1.277617, -.370268]),
    ])

# BF3, Planar D3h symm.
# MP2 energy = -323.7915374. Hartree
# Charge = 0, multiplicity = 1
F3B = ListOfAtoms([
    Atom('B', [.000000, .000000, .000000]),
    Atom('F', [.000000, 1.321760, .000000]),
    Atom('F', [1.144678, -.660880, .000000]),
    Atom('F', [-1.144678, -.660880, .000000]),
    ])

# BCl3, Planar D3h symm.
# MP2 energy = -1403.7595806 Hartree
# Charge = 0, multiplicity = 1
Cl3B = ListOfAtoms([
    Atom('B', [.000000, .000000, .000000]),
    Atom('Cl', [.000000, 1.735352, .000000]),
    Atom('Cl', [1.502859, -.867676, .000000]),
    Atom('Cl', [-1.502859, -.867676, .000000]),
    ])

# AlF3, Planar D3h symm.
# MP2 energy = -541.0397296 Hartree
# Charge = 0, multiplicity = 1
F3Al = ListOfAtoms([
    Atom('Al', [.000000, .000000, .000000]),
    Atom('F', [.000000, 1.644720, .000000]),
    Atom('F', [1.424369, -.822360, .000000]),
    Atom('F', [-1.424369, -.822360, .000000]),
    ])

# AlCl3, Planar D3h symm.
# MP2 energy = -1621.0484142 Hartree
# Charge = 0, multiplicity = 1
Cl3Al = ListOfAtoms([
    Atom('Al', [.000000, .000000, .000000]),
    Atom('Cl', [.000000, 2.069041, .000000]),
    Atom('Cl', [1.791842, -1.034520, .000000]),
    Atom('Cl', [-1.791842, -1.034520, .000000]),
    ])

# CF4, Td symm.
# MP2 energy = -436.4622308 Hartree
# Charge = 0, multiplicity = 1
F4C = ListOfAtoms([
    Atom('C', [.000000, .000000, .000000]),
    Atom('F', [.767436, .767436, .767436]),
    Atom('F', [-.767436, -.767436, .767436]),
    Atom('F', [-.767436, .767436, -.767436]),
    Atom('F', [.767436, -.767436, -.767436]),
    ])

# CCl4, Td symm.
# MP2 energy = -1876.4528012 Hartree
# Charge = 0, multiplicity = 1
Cl4C = ListOfAtoms([
    Atom('C', [.000000, .000000, .000000]),
    Atom('Cl', [1.021340, 1.021340, 1.021340]),
    Atom('Cl', [-1.021340, -1.021340, 1.021340]),
    Atom('Cl', [-1.021340, 1.021340, -1.021340]),
    Atom('Cl', [1.021340, -1.021340, -1.021340]),
    ])

# O=C=S, Linear, C*v symm.
# MP2 energy = -510.704382 Hartree
# Charge = 0, multiplicity = 1
OSC = ListOfAtoms([
    Atom('O', [.000000, .000000, -1.699243]),
    Atom('C', [.000000, .000000, -.520492]),
    Atom('S', [.000000, .000000, 1.044806]),
    ])

# CS2, Linear, D*h symm.
# MP2 energy = -833.2916974 Hartree
# Charge = 0, multiplicity = 1
S2C = ListOfAtoms([
    Atom('S', [.000000, .000000, 1.561117]),
    Atom('C', [.000000, .000000, .000000]),
    Atom('S', [.000000, .000000, -1.561117]),
    ])

# COF2, C2v symm.
# MP2 energy = -312.2651646 Hartree
# Charge = 0, multiplicity = 1
OF2C = ListOfAtoms([
    Atom('O', [.000000, .000000, 1.330715]),
    Atom('C', [.000000, .000000, .144358]),
    Atom('F', [.000000, 1.069490, -.639548]),
    Atom('F', [.000000, -1.069490, -.639548]),
    ])

# SiF4, Td symm.
# MP2 energy = -687.7406597 Hartree
# Charge = 0, multiplicity = 1
F4Si = ListOfAtoms([
    Atom('Si', [.000000, .000000, .000000]),
    Atom('F', [.912806, .912806, .912806]),
    Atom('F', [-.912806, -.912806, .912806]),
    Atom('F', [-.912806, .912806, -.912806]),
    Atom('F', [.912806, -.912806, -.912806]),
    ])

# SiCl4, Td symm.
# MP2 energy = -2127.6916411 Hartree
# Charge = 0, multiplicity = 1
Cl4Si = ListOfAtoms([
    Atom('Si', [.000000, .000000, .000000]),
    Atom('Cl', [1.169349, 1.169349, 1.169349]),
    Atom('Cl', [-1.169349, -1.169349, 1.169349]),
    Atom('Cl', [1.169349, -1.169349, -1.169349]),
    Atom('Cl', [-1.169349, 1.169349, -1.169349]),
    ])

# N2O, Cs symm.
# MP2 energy = -184.2136838 Hartree
# Charge = 0, multiplicity = 1
ON2 = ListOfAtoms([
    Atom('N', [.000000, .000000, -1.231969]),
    Atom('N', [.000000, .000000, -.060851]),
    Atom('O', [.000000, .000000, 1.131218]),
    ])

# ClNO, Cs symm.
# MP2 energy = -589.1833856 Hartree
# Charge = 0, multiplicity = 1
OClN = ListOfAtoms([
    Atom('Cl', [-.537724, -.961291, .000000]),
    Atom('N', [.000000, .997037, .000000]),
    Atom('O', [1.142664, 1.170335, .000000]),
    ])

# NF3, C3v symm.
# MP2 energy = -353.2366115 Hartree
# Charge = 0, multiplicity = 1
F3N = ListOfAtoms([
    Atom('N', [.000000, .000000, .489672]),
    Atom('F', [.000000, 1.238218, -.126952]),
    Atom('F', [1.072328, -.619109, -.126952]),
    Atom('F', [-1.072328, -.619109, -.126952]),
    ])

# PF3, C3v symm.
# MP2 energy = -639.7725739 Hartree
# Charge = 0, multiplicity = 1
F3P = ListOfAtoms([
    Atom('P', [.000000, .000000, .506767]),
    Atom('F', [.000000, 1.383861, -.281537]),
    Atom('F', [1.198459, -.691931, -.281537]),
    Atom('F', [-1.198459, -.691931, -.281537]),
    ])

# O3 (Ozone), C2v symm.
# MP2 energy = -224.8767539 Hartree
# Charge = 0, multiplicity = 1
O3 = ListOfAtoms([
    Atom('O', [.000000, 1.103810, -.228542]),
    Atom('O', [.000000, .000000, .457084]),
    Atom('O', [.000000, -1.103810, -.228542]),
    ])

# F2O, C2v symm.
# MP2 energy = -273.9997434 Hartree
# Charge = 0, multiplicity = 1
OF2 = ListOfAtoms([
    Atom('F', [.000000, 1.110576, -.273729]),
    Atom('O', [.000000, .000000, .615890]),
    Atom('F', [.000000, -1.110576, -.273729]),
    ])

# ClF3, C2v symm.
# MP2 energy = .2017685 Hartree
# Charge = 0, multiplicity = 1
ClF3 = ListOfAtoms([
    Atom('Cl', [.000000, .000000, .376796]),
    Atom('F', [.000000, .000000, -1.258346]),
    Atom('F', [.000000, 1.714544, .273310]),
    Atom('F', [.000000, -1.714544, .273310]),
    ])

# C2F4 (F2C=CF2), D2H symm.
# MP2 energy = -474.3606919 Hartree
# Charge = 0, multiplicity = 1
F4C2 = ListOfAtoms([
    Atom('C', [.000000, .000000, .663230]),
    Atom('C', [.000000, .000000, -.663230]),
    Atom('F', [.000000, 1.112665, 1.385652]),
    Atom('F', [.000000, -1.112665, 1.385652]),
    Atom('F', [.000000, 1.112665, -1.385652]),
    Atom('F', [.000000, -1.112665, -1.385652]),
    ])

# C2Cl4 (Cl2C=CCl2), D2h symm.
# MP2 energy = -1914.4397862 Hartree
# Charge = 0, multiplicity = 1
Cl4C2 = ListOfAtoms([
    Atom('C', [.000000, .000000, .675402]),
    Atom('C', [.000000, .000000, -.675402]),
    Atom('Cl', [.000000, 1.448939, 1.589701]),
    Atom('Cl', [.000000, -1.448939, 1.589701]),
    Atom('Cl', [.000000, -1.448939, -1.589701]),
    Atom('Cl', [.000000, 1.448939, -1.589701]),
    ])

# CF3CN, C3v symm.
# MP2 energy = -429.4170926 Hartree
# Charge = 0, multiplicity = 1
F3C2N = ListOfAtoms([
    Atom('C', [.000000, .000000, -.326350]),
    Atom('C', [.000000, .000000, 1.150830]),
    Atom('F', [.000000, 1.257579, -.787225]),
    Atom('F', [1.089096, -.628790, -.787225]),
    Atom('F', [-1.089096, -.628790, -.787225]),
    Atom('N', [.000000, .000000, 2.329741]),
    ])

# Propyne (C3H4), C3v symm.
# MP2 energy = -116.2562366 Hartree
# Charge = 0, multiplicity = 1
H4C3 = ListOfAtoms([
    Atom('C', [.000000, .000000, .214947]),
    Atom('C', [.000000, .000000, 1.433130]),
    Atom('C', [.000000, .000000, -1.246476]),
    Atom('H', [.000000, .000000, 2.498887]),
    Atom('H', [.000000, 1.021145, -1.636167]),
    Atom('H', [.884337, -.510572, -1.636167]),
    Atom('H', [-.884337, -.510572, -1.636167]),
    ])

# Allene (C3H4), D2d symm.
# MP2 energy = -116.2485221 Hartree
# Charge = 0, multiplicity = 1
H4C3x = ListOfAtoms([
    Atom('C', [.000000, .000000, .000000]),
    Atom('C', [.000000, .000000, 1.311190]),
    Atom('C', [.000000, .000000, -1.311190]),
    Atom('H', [.000000, .926778, 1.876642]),
    Atom('H', [.000000, -.926778, 1.876642]),
    Atom('H', [.926778, .000000, -1.876642]),
    Atom('H', [-.926778, .000000, -1.876642]),
    ])

# Cyclopropene (C3H4), C2v symm.
# MP2 energy = -116.2195708 Hartree
# Charge = 0, multiplicity = 1
H4C3xx = ListOfAtoms([
    Atom('C', [.000000, .000000, .858299]),
    Atom('C', [.000000, -.650545, -.498802]),
    Atom('C', [.000000, .650545, -.498802]),
    Atom('H', [.912438, .000000, 1.456387]),
    Atom('H', [-.912438, .000000, 1.456387]),
    Atom('H', [.000000, -1.584098, -1.038469]),
    Atom('H', [.000000, 1.584098, -1.038469]),
    ])

# Propene (C3H6), Cs symm.
# MP2 energy = -117.4696582 Hartree
# Charge = 0, multiplicity = 1
H6C3 = ListOfAtoms([
    Atom('C', [1.291290, .133682, .000000]),
    Atom('C', [.000000, .479159, .000000]),
    Atom('H', [1.601160, -.907420, .000000]),
    Atom('H', [2.080800, .877337, .000000]),
    Atom('H', [-.263221, 1.536098, .000000]),
    Atom('C', [-1.139757, -.492341, .000000]),
    Atom('H', [-.776859, -1.523291, .000000]),
    Atom('H', [-1.775540, -.352861, .880420]),
    Atom('H', [-1.775540, -.352861, -.880420]),
    ])

# Cyclopropane (C3H6), D3h symm.
# MP2 energy = -117.4628345 Hartree
# Charge = 0, multiplicity = 1
H6C3x = ListOfAtoms([
    Atom('C', [.000000, .866998, .000000]),
    Atom('C', [.750842, -.433499, .000000]),
    Atom('C', [-.750842, -.433499, .000000]),
    Atom('H', [.000000, 1.455762, .910526]),
    Atom('H', [.000000, 1.455762, -.910526]),
    Atom('H', [1.260727, -.727881, -.910526]),
    Atom('H', [1.260727, -.727881, .910526]),
    Atom('H', [-1.260727, -.727881, .910526]),
    Atom('H', [-1.260727, -.727881, -.910526]),
    ])

# Propane (C3H8), C2v symm.
# MP2 energy = -118.6744132 Hartree
# Charge = 0, multiplicity = 1
H8C3 = ListOfAtoms([
    Atom('C', [.000000, .000000, .587716]),
    Atom('C', [.000000, 1.266857, -.260186]),
    Atom('C', [.000000, -1.266857, -.260186]),
    Atom('H', [-.876898, .000000, 1.244713]),
    Atom('H', [.876898, .000000, 1.244713]),
    Atom('H', [.000000, 2.166150, .362066]),
    Atom('H', [.000000, -2.166150, .362066]),
    Atom('H', [.883619, 1.304234, -.904405]),
    Atom('H', [-.883619, 1.304234, -.904405]),
    Atom('H', [-.883619, -1.304234, -.904405]),
    Atom('H', [.883619, -1.304234, -.904405]),
    ])

# Trans-1,3-butadiene (C4H6), C2h symm.
# MP2 energy = -155.4417118 Hartree
# Charge = 0, multiplicity = 1
H6C4 = ListOfAtoms([
    Atom('C', [.605711, 1.746550, .000000]),
    Atom('C', [.605711, .404083, .000000]),
    Atom('C', [-.605711, -.404083, .000000]),
    Atom('C', [-.605711, -1.746550, .000000]),
    Atom('H', [1.527617, 2.317443, .000000]),
    Atom('H', [-.321132, 2.313116, .000000]),
    Atom('H', [1.553503, -.133640, .000000]),
    Atom('H', [-1.553503, .133640, .000000]),
    Atom('H', [.321132, -2.313116, .000000]),
    Atom('H', [-1.527617, -2.317443, .000000]),
    ])

# Dimethylacetylene (2-butyne, C4H6), D3h symm (eclipsed).
# MP2 energy =  -155.435151. Hartree
# Charge = 0, multiplicity = 1
H6C4x = ListOfAtoms([
    Atom('C', [.000000, .000000, 2.071955]),
    Atom('C', [.000000, .000000, .609970]),
    Atom('C', [.000000, .000000, -.609970]),
    Atom('C', [.000000, .000000, -2.071955]),
    Atom('H', [.000000, 1.020696, 2.464562]),
    Atom('H', [-.883949, -.510348, 2.464562]),
    Atom('H', [.883949, -.510348, 2.464562]),
    Atom('H', [.000000, 1.020696, -2.464562]),
    Atom('H', [.883949, -.510348, -2.464562]),
    Atom('H', [-.883949, -.510348, -2.464562]),
    ])

# Methylenecyclopropane (C4H6), C2v symm.
# MP2 energy = -155.4160189 Hartree
# Charge = 0, multiplicity = 1
H6C4xx = ListOfAtoms([
    Atom('C', [.000000, .000000, .315026]),
    Atom('C', [.000000, -.767920, -.932032]),
    Atom('C', [.000000, .767920, -.932032]),
    Atom('C', [.000000, .000000, 1.640027]),
    Atom('H', [-.912794, -1.271789, -1.239303]),
    Atom('H', [.912794, -1.271789, -1.239303]),
    Atom('H', [.912794, 1.271789, -1.239303]),
    Atom('H', [-.912794, 1.271789, -1.239303]),
    Atom('H', [.000000, -.926908, 2.205640]),
    Atom('H', [.000000, .926908, 2.205640]),
    ])

# Bicyclo[1.1.0]butane (C4H6), C2v symm.
# MP2 energy = -155.4094811 Hartree
# Charge = 0, multiplicity = 1
H6C4xxx = ListOfAtoms([
    Atom('C', [.000000, 1.131343, .310424]),
    Atom('C', [.000000, -1.131343, .310424]),
    Atom('C', [.747952, .000000, -.311812]),
    Atom('C', [-.747952, .000000, -.311812]),
    Atom('H', [.000000, 1.237033, 1.397617]),
    Atom('H', [.000000, 2.077375, -.227668]),
    Atom('H', [.000000, -1.237033, 1.397617]),
    Atom('H', [.000000, -2.077375, -.227668]),
    Atom('H', [1.414410, .000000, -1.161626]),
    Atom('H', [-1.414410, .000000, -1.161626]),
    ])

# Cyclobutene (C4H6), C2v symm.
# MP2 energy = -155.4293322 Hartree
# Charge = 0, multiplicity = 1
H6C4xxxx = ListOfAtoms([
    Atom('C', [.000000, -.672762, .811217]),
    Atom('C', [.000000, .672762, .811217]),
    Atom('C', [.000000, -.781980, -.696648]),
    Atom('C', [.000000, .781980, -.696648]),
    Atom('H', [.000000, -1.422393, 1.597763]),
    Atom('H', [.000000, 1.422393, 1.597763]),
    Atom('H', [-.889310, -1.239242, -1.142591]),
    Atom('H', [.889310, -1.239242, -1.142591]),
    Atom('H', [.889310, 1.239242, -1.142591]),
    Atom('H', [-.889310, 1.239242, -1.142591]),
    ])

# Cyclobutane (C4H8), D2d symm.
# MP2 energy = -156.6370628 Hartree
# Charge = 0, multiplicity = 1
H8C4 = ListOfAtoms([
    Atom('C', [.000000, 1.071142, .147626]),
    Atom('C', [.000000, -1.071142, .147626]),
    Atom('C', [-1.071142, .000000, -.147626]),
    Atom('C', [1.071142, .000000, -.147626]),
    Atom('H', [.000000, 1.986858, -.450077]),
    Atom('H', [.000000, 1.342921, 1.207520]),
    Atom('H', [.000000, -1.986858, -.450077]),
    Atom('H', [.000000, -1.342921, 1.207520]),
    Atom('H', [-1.986858, .000000, .450077]),
    Atom('H', [-1.342921, .000000, -1.207520]),
    Atom('H', [1.986858, .000000, .450077]),
    Atom('H', [1.342921, .000000, -1.207520]),
    ])

# Isobutene (C4H8), Single bonds trans, C2v symm.
# MP2 energy = 156.646397 Hartree
# Charge = 0, multiplicity = 1
H8C4x = ListOfAtoms([
    Atom('C', [.000000, .000000, 1.458807]),
    Atom('C', [.000000, .000000, .119588]),
    Atom('H', [.000000, .924302, 2.028409]),
    Atom('H', [.000000, -.924302, 2.028409]),
    Atom('C', [.000000, 1.272683, -.678803]),
    Atom('H', [.000000, 2.153042, -.031588]),
    Atom('H', [.880211, 1.323542, -1.329592]),
    Atom('H', [-.880211, 1.323542, -1.329592]),
    Atom('C', [.000000, -1.272683, -.678803]),
    Atom('H', [.000000, -2.153042, -.031588]),
    Atom('H', [-.880211, -1.323542, -1.329592]),
    Atom('H', [.880211, -1.323542, -1.329592]),
    ])

# Trans-butane (C4H10), C2h symm.
# MP2 energy = -157.8449716 Hartree
# Charge = 0, multiplicity = 1
H10C4 = ListOfAtoms([
    Atom('C', [.702581, 1.820873, .000000]),
    Atom('C', [.702581, .296325, .000000]),
    Atom('C', [-.702581, -.296325, .000000]),
    Atom('C', [-.702581, -1.820873, .000000]),
    Atom('H', [1.719809, 2.222340, .000000]),
    Atom('H', [-1.719809, -2.222340, .000000]),
    Atom('H', [.188154, 2.210362, .883614]),
    Atom('H', [.188154, 2.210362, -.883614]),
    Atom('H', [-.188154, -2.210362, .883614]),
    Atom('H', [-.188154, -2.210362, -.883614]),
    Atom('H', [1.247707, -.072660, -.877569]),
    Atom('H', [1.247707, -.072660, .877569]),
    Atom('H', [-1.247707, .072660, -.877569]),
    Atom('H', [-1.247707, .072660, .877569]),
    ])

# Isobutane (C4H10), C3v symm.
# MP2 energy = -157.8477683 Hartree
# Charge = 0, multiplicity = 1
H10C4x = ListOfAtoms([
    Atom('C', [.000000, .000000, .376949]),
    Atom('H', [.000000, .000000, 1.475269]),
    Atom('C', [.000000, 1.450290, -.096234]),
    Atom('H', [.000000, 1.493997, -1.190847]),
    Atom('H', [-.885482, 1.984695, .261297]),
    Atom('H', [.885482, 1.984695, .261297]),
    Atom('C', [1.255988, -.725145, -.096234]),
    Atom('H', [1.293839, -.746998, -1.190847]),
    Atom('H', [2.161537, -.225498, .261297]),
    Atom('H', [1.276055, -1.759198, .261297]),
    Atom('C', [-1.255988, -.725145, -.096234]),
    Atom('H', [-1.293839, -.746998, -1.190847]),
    Atom('H', [-1.276055, -1.759198, .261297]),
    Atom('H', [-2.161537, -.225498, .261297]),
    ])

# Spiropentane (C5H8), D2d symm.
# MP2 energy = -194.5892415 Hartree
# Charge = 0, multiplicity = 1
H8C5 = ListOfAtoms([
    Atom('C', [.000000, .000000, .000000]),
    Atom('C', [.000000, .762014, 1.265752]),
    Atom('C', [.000000, -.762014, 1.265752]),
    Atom('C', [.762014, .000000, -1.265752]),
    Atom('C', [-.762014, .000000, -1.265752]),
    Atom('H', [-.914023, 1.265075, 1.568090]),
    Atom('H', [.914023, 1.265075, 1.568090]),
    Atom('H', [-.914023, -1.265075, 1.568090]),
    Atom('H', [.914023, -1.265075, 1.568090]),
    Atom('H', [1.265075, -.914023, -1.568090]),
    Atom('H', [1.265075, .914023, -1.568090]),
    Atom('H', [-1.265075, -.914023, -1.568090]),
    Atom('H', [-1.265075, .914023, -1.568090]),
    ])

# Benzene (C6H6), D6h symm.
# MP2 energy = -231.4871881 Hartree
# Charge = 0, multiplicity = 1
H6C6 = ListOfAtoms([
    Atom('C', [.000000, 1.395248, .000000]),
    Atom('C', [1.208320, .697624, .000000]),
    Atom('C', [1.208320, -.697624, .000000]),
    Atom('C', [.000000, -1.395248, .000000]),
    Atom('C', [-1.208320, -.697624, .000000]),
    Atom('C', [-1.208320, .697624, .000000]),
    Atom('H', [.000000, 2.482360, .000000]),
    Atom('H', [2.149787, 1.241180, .000000]),
    Atom('H', [2.149787, -1.241180, .000000]),
    Atom('H', [.000000, -2.482360, .000000]),
    Atom('H', [-2.149787, -1.241180, .000000]),
    Atom('H', [-2.149787, 1.241180, .000000]),
    ])

# Difluoromethane (H2CF2), C2v symm.
# MP2 energy = -238.3733057 Hartree
# Charge = 0, multiplicity = 1
F2CH2 = ListOfAtoms([
    Atom('C', [.000000, .000000, .502903]),
    Atom('F', [.000000, 1.109716, -.290601]),
    Atom('F', [.000000, -1.109716, -.290601]),
    Atom('H', [-.908369, .000000, 1.106699]),
    Atom('H', [.908369, .000000, 1.106699]),
    ])

# Trifluoromethane (HCF3), C3v symm.
# MP2 energy = -337.4189848 Hartree
# Charge = 0, multiplicity = 1
HCF3 = ListOfAtoms([
    Atom('C', [.000000, .000000, .341023]),
    Atom('H', [.000000, .000000, 1.429485]),
    Atom('F', [.000000, 1.258200, -.128727]),
    Atom('F', [1.089633, -.629100, -.128727]),
    Atom('F', [-1.089633, -.629100, -.128727]),
    ])

# Dichloromethane (H2CCl2), C2v symm.
# MP2 energy = -958.4007187 Hartree
# Charge = 0, multiplicity = 1
Cl2CH2 = ListOfAtoms([
    Atom('C', [.000000, .000000, .759945]),
    Atom('Cl', [.000000, 1.474200, -.215115]),
    Atom('Cl', [.000000, -1.474200, -.215115]),
    Atom('H', [-.894585, .000000, 1.377127]),
    Atom('H', [.894585, .000000, 1.377127]),
    ])

# Chloroform (HCCl3), C3v symm.
# MP2 energy = -1417.4294497 Hartree
# Charge = 0, multiplicity = 1
HCCl3 = ListOfAtoms([
    Atom('C', [.000000, .000000, .451679]),
    Atom('H', [.000000, .000000, 1.537586]),
    Atom('Cl', [.000000, 1.681723, -.083287]),
    Atom('Cl', [1.456415, -.840862, -.083287]),
    Atom('Cl', [-1.456415, -.840862, -.083287]),
    ])

# Methylamine (H3C-NH2), Cs symm.
# MP2 energy = -95.5144387 Hartree
# Charge = 0, multiplicity = 1
H5CN = ListOfAtoms([
    Atom('C', [.051736, .704422, .000000]),
    Atom('N', [.051736, -.759616, .000000]),
    Atom('H', [-.941735, 1.176192, .000000]),
    Atom('H', [-.458181, -1.099433, .812370]),
    Atom('H', [-.458181, -1.099433, -.812370]),
    Atom('H', [.592763, 1.056727, .880670]),
    Atom('H', [.592763, 1.056727, -.880670]),
    ])

# Acetonitrile (CH3-CN), C3v symm.
# MP2 energy = -132.3513069 Hartree
# Charge = 0, multiplicity = 1
H3C2N = ListOfAtoms([
    Atom('C', [.000000, .000000, -1.186930]),
    Atom('C', [.000000, .000000, .273874]),
    Atom('N', [.000000, .000000, 1.452206]),
    Atom('H', [.000000, 1.024986, -1.562370]),
    Atom('H', [.887664, -.512493, -1.562370]),
    Atom('H', [-.887664, -.512493, -1.562370]),
    ])

# Nitromethane (CH3-NO2), Cs symm.
# MP2 energy = -244.3453346 Hartree
# Charge = 0, multiplicity = 1
O2H3CN = ListOfAtoms([
    Atom('C', [-.114282, -1.314565, .000000]),
    Atom('N', [.000000, .166480, .000000]),
    Atom('H', [.899565, -1.715256, .000000]),
    Atom('H', [-.640921, -1.607212, .904956]),
    Atom('H', [-.640921, -1.607212, -.904956]),
    Atom('O', [.066748, .728232, -1.103775]),
    Atom('O', [.066748, .728232, 1.103775]),
    ])

# Methylnitrite (CH3-O-N=O), NOCH trans, ONOC cis, Cs symm.
# MP2 energy = -244.3391134 Hartree
# Charge = 0, multiplicity = 1
O2H3CNx = ListOfAtoms([
    Atom('C', [-1.316208, .309247, .000000]),
    Atom('O', [.000000, .896852, .000000]),
    Atom('H', [-1.985538, 1.166013, .000000]),
    Atom('H', [-1.464336, -.304637, .890672]),
    Atom('H', [-1.464336, -.304637, -.890672]),
    Atom('N', [1.045334, -.022815, .000000]),
    Atom('O', [.686764, -1.178416, .000000]),
    ])

# Methylsilane (CH3-SiH3), C3v symm.
# MP2 energy = -330.5003988 Hartree
# Charge = 0, multiplicity = 1
H6SiC = ListOfAtoms([
    Atom('C', [.000000, .000000, -1.244466]),
    Atom('Si', [.000000, .000000, .635703]),
    Atom('H', [.000000, -1.019762, -1.636363]),
    Atom('H', [-.883140, .509881, -1.636363]),
    Atom('H', [.883140, .509881, -1.636363]),
    Atom('H', [.000000, 1.391234, 1.158682]),
    Atom('H', [-1.204844, -.695617, 1.158682]),
    Atom('H', [1.204844, -.695617, 1.158682]),
    ])

# Formic Acid (HCOOH), HOCO cis, Cs symm.
# MP2 energy = -189.2518734 Hartree
# Charge = 0, multiplicity = 1
O2H2C = ListOfAtoms([
    Atom('O', [-1.040945, -.436432, .000000]),
    Atom('C', [.000000, .423949, .000000]),
    Atom('O', [1.169372, .103741, .000000]),
    Atom('H', [-.649570, -1.335134, .000000]),
    Atom('H', [-.377847, 1.452967, .000000]),
    ])

# Methyl formate (HCOOCH3), Cs symm.
# MP2 energy = -228.4116599 Hartree
# Charge = 0, multiplicity = 1
O2H4C2 = ListOfAtoms([
    Atom('C', [-.931209, -.083866, .000000]),
    Atom('O', [-.711019, -1.278209, .000000]),
    Atom('O', [.000000, .886841, .000000]),
    Atom('H', [-1.928360, .374598, .000000]),
    Atom('C', [1.356899, .397287, .000000]),
    Atom('H', [1.980134, 1.288164, .000000]),
    Atom('H', [1.541121, -.206172, .889397]),
    Atom('H', [1.541121, -.206172, -.889397]),
    ])

# Acetamide (CH3CONH2), C1 symm.
# MP2 energy = -208.5849862 Hartree
# Charge = 0, multiplicity = 1
OH5C2N = ListOfAtoms([
    Atom('O', [.424546, 1.327024, .008034]),
    Atom('C', [.077158, .149789, -.004249]),
    Atom('N', [.985518, -.878537, -.048910]),
    Atom('C', [-1.371475, -.288665, -.000144]),
    Atom('H', [.707952, -1.824249, .169942]),
    Atom('H', [-1.997229, .584922, -.175477]),
    Atom('H', [-1.560842, -1.039270, -.771686]),
    Atom('H', [-1.632113, -.723007, .969814]),
    Atom('H', [1.953133, -.631574, .111866]),
    ])

# Aziridine (cyclic CH2-NH-CH2 ring), C2v symm.
# MP2 energy = -133.4730917 Hartree
# Charge = 0, multiplicity = 1
H5C2N = ListOfAtoms([
    Atom('C', [-.038450, -.397326, .739421]),
    Atom('N', [-.038450, .875189, .000000]),
    Atom('C', [-.038450, -.397326, -.739421]),
    Atom('H', [.903052, 1.268239, .000000]),
    Atom('H', [-.955661, -.604926, 1.280047]),
    Atom('H', [-.955661, -.604926, -1.280047]),
    Atom('H', [.869409, -.708399, 1.249033]),
    Atom('H', [.869409, -.708399, -1.249033]),
    ])

# Cyanogen (NCCN). D*h symm.
# MP2 energy = -185.1746395 Hartree
# Charge = 0, multiplicity = 1
C2N2 = ListOfAtoms([
    Atom('N', [.000000, .000000, 1.875875]),
    Atom('C', [.000000, .000000, .690573]),
    Atom('C', [.000000, .000000, -.690573]),
    Atom('N', [.000000, .000000, -1.875875]),
    ])

# Dimethylamine, (CH3)2NH, Cs symm.
# MP2 energy = -134.6781011 Hartree
# Charge = 0, multiplicity = 1
H7C2N = ListOfAtoms([
    Atom('C', [-.027530, -.224702, 1.204880]),
    Atom('N', [-.027530, .592470, .000000]),
    Atom('C', [-.027530, -.224702, -1.204880]),
    Atom('H', [.791501, -.962742, 1.248506]),
    Atom('H', [.039598, .421182, 2.083405]),
    Atom('H', [-.972220, -.772987, 1.261750]),
    Atom('H', [.805303, 1.178220, .000000]),
    Atom('H', [.791501, -.962742, -1.248506]),
    Atom('H', [.039598, .421182, -2.083405]),
    Atom('H', [-.972220, -.772987, -1.261750]),
    ])

# Trans-Ethylamine (CH3-CH2-NH2), Cs symm.
# MP2 energy = -134.6882447 Hartree
# Charge = 0, multiplicity = 1
H7C2Nx = ListOfAtoms([
    Atom('C', [1.210014, -.353598, .000000]),
    Atom('C', [.000000, .575951, .000000]),
    Atom('N', [-1.305351, -.087478, .000000]),
    Atom('H', [2.149310, .208498, .000000]),
    Atom('H', [1.201796, -.997760, .884909]),
    Atom('H', [1.201796, -.997760, -.884909]),
    Atom('H', [.034561, 1.230963, -.876478]),
    Atom('H', [.034561, 1.230963, .876478]),
    Atom('H', [-1.372326, -.698340, .813132]),
    Atom('H', [-1.372326, -.698340, -.813132]),
    ])

# Ketene (H2C=C=O), C2v symm.
# MP2 energy = -152.1600778 Hartree
# Charge = 0, multiplicity = 1
OH2C2 = ListOfAtoms([
    Atom('C', [.000000, .000000, -1.219340]),
    Atom('C', [.000000, .000000, .098920]),
    Atom('H', [.000000, .938847, -1.753224]),
    Atom('H', [.000000, -.938847, -1.753224]),
    Atom('O', [.000000, .000000, 1.278620]),
    ])

# Oxirane (cyclic CH2-O-CH2 ring), C2v symm.
# MP2 energy = -153.3156907 Hartree
# Charge = 0, multiplicity = 1
OH4C2 = ListOfAtoms([
    Atom('C', [.000000, .731580, -.375674]),
    Atom('O', [.000000, .000000, .860950]),
    Atom('C', [.000000, -.731580, -.375674]),
    Atom('H', [.919568, 1.268821, -.594878]),
    Atom('H', [-.919568, 1.268821, -.594878]),
    Atom('H', [-.919568, -1.268821, -.594878]),
    Atom('H', [.919568, -1.268821, -.594878]),
    ])

# Acetaldehyde (CH3CHO), Cs symm.
# MP2 energy = -153.3589689 Hartree
# Charge = 0, multiplicity = 1
OH4C2x = ListOfAtoms([
    Atom('O', [1.218055, .361240, .000000]),
    Atom('C', [.000000, .464133, .000000]),
    Atom('H', [-.477241, 1.465295, .000000]),
    Atom('C', [-.948102, -.700138, .000000]),
    Atom('H', [-.385946, -1.634236, .000000]),
    Atom('H', [-1.596321, -.652475, .880946]),
    Atom('H', [-1.596321, -.652475, -.880946]),
    ])

# Glyoxal (O=CH-CH=O). Trans, C2h symm.
# MP2 energy = -227.2037251 Hartree
# Charge = 0, multiplicity = 1
O2H2C2 = ListOfAtoms([
    Atom('C', [.000000, .756430, .000000]),
    Atom('C', [.000000, -.756430, .000000]),
    Atom('O', [1.046090, 1.389916, .000000]),
    Atom('H', [-.999940, 1.228191, .000000]),
    Atom('O', [-1.046090, -1.389916, .000000]),
    Atom('H', [.999940, -1.228191, .000000]),
    ])

# Ethanol (trans, CH3CH2OH), Cs symm.
# MP2 energy = -154.5289541 Hartree
# Charge = 0, multiplicity = 1
OH6C2 = ListOfAtoms([
    Atom('C', [1.168181, -.400382, .000000]),
    Atom('C', [.000000, .559462, .000000]),
    Atom('O', [-1.190083, -.227669, .000000]),
    Atom('H', [-1.946623, .381525, .000000]),
    Atom('H', [.042557, 1.207508, .886933]),
    Atom('H', [.042557, 1.207508, -.886933]),
    Atom('H', [2.115891, .144800, .000000]),
    Atom('H', [1.128599, -1.037234, .885881]),
    Atom('H', [1.128599, -1.037234, -.885881]),
    ])

# DimethylEther (CH3-O-CH3), C2v symm.
# MP2 energy = -154.5155453 Hartree
# Charge = 0, multiplicity = 1
OH6C2x = ListOfAtoms([
    Atom('C', [.000000, 1.165725, -.199950]),
    Atom('O', [.000000, .000000, .600110]),
    Atom('C', [.000000, -1.165725, -.199950]),
    Atom('H', [.000000, 2.017769, .480203]),
    Atom('H', [.891784, 1.214320, -.840474]),
    Atom('H', [-.891784, 1.214320, -.840474]),
    Atom('H', [.000000, -2.017769, .480203]),
    Atom('H', [-.891784, -1.214320, -.840474]),
    Atom('H', [.891784, -1.214320, -.840474]),
    ])

# Thiooxirane (cyclic CH2-S-CH2 ring), C2v symm.
# MP2 energy = 475.9496155 Hartree
# Charge = 0, multiplicity = 1
SH4C2 = ListOfAtoms([
    Atom('C', [.000000, -.739719, -.792334]),
    Atom('S', [.000000, .000000, .863474]),
    Atom('C', [.000000, .739719, -.792334]),
    Atom('H', [-.913940, -1.250142, -1.076894]),
    Atom('H', [.913940, -1.250142, -1.076894]),
    Atom('H', [.913940, 1.250142, -1.076894]),
    Atom('H', [-.913940, 1.250142, -1.076894]),
    ])

# Dimethylsulfoxide (CH3)2SO, Cs symm.
# MP2 energy = -552.1363114 Hartree
# Charge = 0, multiplicity = 1
SOC2H6 = ListOfAtoms([
    Atom('S', [.000002, .231838, -.438643]),
    Atom('O', [.000020, 1.500742, .379819]),
    Atom('C', [1.339528, -.809022, .180717]),
    Atom('C', [-1.339548, -.808992, .180718]),
    Atom('H', [1.255835, -.896385, 1.266825]),
    Atom('H', [-2.279404, -.313924, -.068674]),
    Atom('H', [1.304407, -1.793327, -.292589]),
    Atom('H', [2.279395, -.313974, -.068674]),
    Atom('H', [-1.304447, -1.793298, -.292587]),
    Atom('H', [-1.255857, -.896355, 1.266826]),
    ])

# ThioEthanol (CH3-CH2-SH), Cs symm.
# MP2 energy = -477.139659 Hartree
# Charge = 0, multiplicity = 1
SH6C2 = ListOfAtoms([
    Atom('C', [1.514343, .679412, .000000]),
    Atom('C', [.000000, .826412, .000000]),
    Atom('S', [-.756068, -.831284, .000000]),
    Atom('H', [-2.035346, -.427738, .000000]),
    Atom('H', [-.324970, 1.376482, .885793]),
    Atom('H', [-.324970, 1.376482, -.885793]),
    Atom('H', [1.986503, 1.665082, .000000]),
    Atom('H', [1.854904, .137645, .885494]),
    Atom('H', [1.854904, .137645, -.885494]),
    ])

# Dimethyl ThioEther (CH3-S-CH3), C2v symm.
# MP2 energy = -477.1413207 Hartree
# Charge = 0, multiplicity = 1
SH6C2x = ListOfAtoms([
    Atom('C', [.000000, 1.366668, -.513713]),
    Atom('S', [.000000, .000000, .664273]),
    Atom('C', [.000000, -1.366668, -.513713]),
    Atom('H', [.000000, 2.296687, .057284]),
    Atom('H', [.891644, 1.345680, -1.144596]),
    Atom('H', [-.891644, 1.345680, -1.144596]),
    Atom('H', [.000000, -2.296687, .057284]),
    Atom('H', [-.891644, -1.345680, -1.144596]),
    Atom('H', [.891644, -1.345680, -1.144596]),
    ])

# Vinyl fluoride (H2C=CHF), Cs symm.
# MP2 energy = -177.3151594 Hartree
# Charge = 0, multiplicity = 1
FC2H3 = ListOfAtoms([
    Atom('C', [.000000, .437714, .000000]),
    Atom('C', [1.191923, -.145087, .000000]),
    Atom('F', [-1.148929, -.278332, .000000]),
    Atom('H', [-.186445, 1.505778, .000000]),
    Atom('H', [1.291348, -1.222833, .000000]),
    Atom('H', [2.083924, .466279, .000000]),
    ])

# Ethyl chloride (CH3-CH2-Cl), Cs symm.
# MP2 energy = -538.5434131 Hartree
# Charge = 0, multiplicity = 1
ClC2H5 = ListOfAtoms([
    Atom('C', [.000000, .807636, .000000]),
    Atom('C', [1.505827, .647832, .000000]),
    Atom('Cl', [-.823553, -.779970, .000000]),
    Atom('H', [-.344979, 1.341649, .885248]),
    Atom('H', [-.344979, 1.341649, -.885248]),
    Atom('H', [1.976903, 1.634877, .000000]),
    Atom('H', [1.839246, .104250, .885398]),
    Atom('H', [1.839246, .104250, -.885398]),
    ])

# Vinyl chloride, H2C=CHCl, Cs symm.
# MP2 energy = -537.3360622 Hartree
# Charge = 0, multiplicity = 1
ClC2H3 = ListOfAtoms([
    Atom('C', [.000000, .756016, .000000]),
    Atom('C', [1.303223, 1.028507, .000000]),
    Atom('Cl', [-.631555, -.854980, .000000]),
    Atom('H', [-.771098, 1.516963, .000000]),
    Atom('H', [2.056095, .249427, .000000]),
    Atom('H', [1.632096, 2.061125, .000000]),
    ])

# CyanoEthylene (H2C=CHCN), Cs symm.
# MP2 energy = -170.3161069 Hartree
# Charge = 0, multiplicity = 1
H3C3N = ListOfAtoms([
    Atom('C', [-.161594, -1.638625, .000000]),
    Atom('C', [.584957, -.524961, .000000]),
    Atom('C', [.000000, .782253, .000000]),
    Atom('H', [-1.245203, -1.598169, .000000]),
    Atom('H', [.305973, -2.616405, .000000]),
    Atom('H', [1.669863, -.572107, .000000]),
    Atom('N', [-.467259, 1.867811, .000000]),
    ])

# Acetone (CH3-CO-CH3), C2v symm.
# MP2 energy = -192.5408724 Hartree
# Charge = 0, multiplicity = 1
OH6C3 = ListOfAtoms([
    Atom('O', [.000000, .000000, 1.405591]),
    Atom('C', [.000000, .000000, .179060]),
    Atom('C', [.000000, 1.285490, -.616342]),
    Atom('C', [.000000, -1.285490, -.616342]),
    Atom('H', [.000000, 2.134917, .066535]),
    Atom('H', [.000000, -2.134917, .066535]),
    Atom('H', [-.881086, 1.331548, -1.264013]),
    Atom('H', [.881086, 1.331548, -1.264013]),
    Atom('H', [.881086, -1.331548, -1.264013]),
    Atom('H', [-.881086, -1.331548, -1.264013]),
    ])

# Acetic Acid (CH3COOH), Single bonds trans, Cs symm.
# MP2 energy = -228.4339789 Hartree
# Charge = 0, multiplicity = 1
O2H4C2x = ListOfAtoms([
    Atom('C', [.000000, .154560, .000000]),
    Atom('O', [.166384, 1.360084, .000000]),
    Atom('O', [-1.236449, -.415036, .000000]),
    Atom('H', [-1.867646, .333582, .000000]),
    Atom('C', [1.073776, -.892748, .000000]),
    Atom('H', [2.048189, -.408135, .000000]),
    Atom('H', [.968661, -1.528353, .881747]),
    Atom('H', [.968661, -1.528353, -.881747]),
    ])

# Acetyl fluoride (CH3COF), HCCO cis, Cs symm.
# MP2 energy = -252.4133329 Hartree
# Charge = 0, multiplicity = 1
OFC2H3 = ListOfAtoms([
    Atom('C', [.000000, .186396, .000000]),
    Atom('O', [.126651, 1.377199, .000000]),
    Atom('F', [-1.243950, -.382745, .000000]),
    Atom('C', [1.049454, -.876224, .000000]),
    Atom('H', [2.035883, -.417099, .000000]),
    Atom('H', [.924869, -1.508407, .881549]),
    Atom('H', [.924869, -1.508407, -.881549]),
    ])

# Acetyl,Chloride (CH3COCl), HCCO cis, Cs symm.
# MP2 energy = -612.4186269 Hartree
# Charge = 0, multiplicity = 1
OClC2H3 = ListOfAtoms([
    Atom('C', [.000000, .523878, .000000]),
    Atom('C', [1.486075, .716377, .000000]),
    Atom('Cl', [-.452286, -1.217999, .000000]),
    Atom('O', [-.845539, 1.374940, .000000]),
    Atom('H', [1.701027, 1.784793, .000000]),
    Atom('H', [1.917847, .240067, .882679]),
    Atom('H', [1.917847, .240067, -.882679]),
    ])

# Propyl chloride (CH3CH2CH2Cl), Cs symm.
# MP2 energy = -577.7144239 Hartree
# Charge = 0, multiplicity = 1
H7C3Cl = ListOfAtoms([
    Atom('C', [.892629, -.642344, .000000]),
    Atom('C', [2.365587, -.245168, .000000]),
    Atom('C', [.000000, .582921, .000000]),
    Atom('H', [.663731, -1.252117, .879201]),
    Atom('H', [.663731, -1.252117, -.879201]),
    Atom('H', [3.005476, -1.130924, .000000]),
    Atom('Cl', [-1.732810, .139743, .000000]),
    Atom('H', [2.614882, .347704, -.884730]),
    Atom('H', [2.614882, .347704, .884730]),
    Atom('H', [.172881, 1.195836, .886460]),
    Atom('H', [.172881, 1.195836, -.886460]),
    ])

# Isopropyl alcohol, (CH3)2CH-OH, Gauche isomer, C1 symm.
# MP2 energy = -193.706552 Hartree
# Charge = 0, multiplicity = 1
OH8C3 = ListOfAtoms([
    Atom('O', [.027191, 1.363691, -.167516]),
    Atom('C', [-.000926, .036459, .370128]),
    Atom('H', [.859465, 1.775647, .121307]),
    Atom('H', [.007371, .082145, 1.470506]),
    Atom('C', [-1.313275, -.563514, -.088979]),
    Atom('C', [1.200721, -.764480, -.104920]),
    Atom('H', [-1.334005, -.607253, -1.181009]),
    Atom('H', [1.202843, -.807817, -1.197189]),
    Atom('H', [-2.147812, .054993, .247676]),
    Atom('H', [2.136462, -.299324, .223164]),
    Atom('H', [-1.438709, -1.574275, .308340]),
    Atom('H', [1.177736, -1.784436, .289967]),
    ])

# Methyl ethyl ether (CH3-CH2-O-CH3), Trans, Cs symm.
# MP2 energy = -193.6914772 Hartree
# Charge = 0, multiplicity = 1
OH8C3x = ListOfAtoms([
    Atom('O', [.006429, -.712741, .000000]),
    Atom('C', [.000000, .705845, .000000]),
    Atom('C', [1.324518, -1.226029, .000000]),
    Atom('C', [-1.442169, 1.160325, .000000]),
    Atom('H', [.530962, 1.086484, .886881]),
    Atom('H', [.530962, 1.086484, -.886881]),
    Atom('H', [1.241648, -2.313325, .000000]),
    Atom('H', [1.881329, -.905925, -.891710]),
    Atom('H', [1.881329, -.905925, .891710]),
    Atom('H', [-1.954863, .780605, -.885855]),
    Atom('H', [-1.954863, .780605, .885855]),
    Atom('H', [-1.502025, 2.252083, .000000]),
    ])

# Trimethyl Amine, (CH3)3N, C3v symm.
# MP2 energy = -173.8464634 Hartree
# Charge = 0, multiplicity = 1
H9C3N = ListOfAtoms([
    Atom('N', [.000000, .000000, .395846]),
    Atom('C', [.000000, 1.378021, -.065175]),
    Atom('C', [1.193401, -.689011, -.065175]),
    Atom('C', [-1.193401, -.689011, -.065175]),
    Atom('H', [.000000, 1.461142, -1.167899]),
    Atom('H', [.886156, 1.891052, .317655]),
    Atom('H', [-.886156, 1.891052, .317655]),
    Atom('H', [1.265386, -.730571, -1.167899]),
    Atom('H', [1.194621, -1.712960, .317655]),
    Atom('H', [2.080777, -.178092, .317655]),
    Atom('H', [-1.265386, -.730571, -1.167899]),
    Atom('H', [-2.080777, -.178092, .317655]),
    Atom('H', [-1.194621, -1.712960, .317655]),
    ])

# Furan (cyclic C4H4O), C2v symm.
# MP2 energy = -229.3327814 Hartree
# Charge = 0, multiplicity = 1
OH4C4 = ListOfAtoms([
    Atom('O', [.000000, .000000, 1.163339]),
    Atom('C', [.000000, 1.094700, .348039]),
    Atom('C', [.000000, -1.094700, .348039]),
    Atom('C', [.000000, .713200, -.962161]),
    Atom('C', [.000000, -.713200, -.962161]),
    Atom('H', [.000000, 2.049359, .851113]),
    Atom('H', [.000000, -2.049359, .851113]),
    Atom('H', [.000000, 1.370828, -1.819738]),
    Atom('H', [.000000, -1.370828, -1.819738]),
    ])

# Thiophene (cyclic C4H4S), C2v symm.
# MP2 energy = -551.9559715 Hartree
# Charge = 0, multiplicity = 1
SH4C4 = ListOfAtoms([
    Atom('S', [.000000, .000000, 1.189753]),
    Atom('C', [.000000, 1.233876, -.001474]),
    Atom('C', [.000000, -1.233876, -.001474]),
    Atom('C', [.000000, .709173, -1.272322]),
    Atom('C', [.000000, -.709173, -1.272322]),
    Atom('H', [.000000, 2.275343, .291984]),
    Atom('H', [.000000, -2.275343, .291984]),
    Atom('H', [.000000, 1.321934, -2.167231]),
    Atom('H', [.000000, -1.321934, -2.167231]),
    ])

# Pyrrole (Planar cyclic C4H4NH), C2v symm.
# MP2 energy = -209.5041766 Hartree
# Charge = 0, multiplicity = 1
H5C4N = ListOfAtoms([
    Atom('H', [.000000, .000000, 2.129296]),
    Atom('N', [.000000, .000000, 1.118684]),
    Atom('C', [.000000, 1.124516, .333565]),
    Atom('C', [.000000, -1.124516, .333565]),
    Atom('C', [.000000, .708407, -.983807]),
    Atom('C', [.000000, -.708407, -.983807]),
    Atom('H', [.000000, 2.112872, .770496]),
    Atom('H', [.000000, -2.112872, .770496]),
    Atom('H', [.000000, 1.357252, -1.849085]),
    Atom('H', [.000000, -1.357252, -1.849085]),
    ])

# Pyridine (cyclic C5H5N), C2v symm.
# MP2 energy = -247.5106791 Hartree
# Charge = 0, multiplicity = 1
H5C5N = ListOfAtoms([
    Atom('N', [.000000, .000000, 1.424672]),
    Atom('C', [.000000, .000000, -1.386178]),
    Atom('C', [.000000, 1.144277, .720306]),
    Atom('C', [.000000, -1.144277, .720306]),
    Atom('C', [.000000, -1.196404, -.672917]),
    Atom('C', [.000000, 1.196404, -.672917]),
    Atom('H', [.000000, .000000, -2.473052]),
    Atom('H', [.000000, 2.060723, 1.307477]),
    Atom('H', [.000000, -2.060723, 1.307477]),
    Atom('H', [.000000, -2.155293, -1.183103]),
    Atom('H', [.000000, 2.155293, -1.183103]),
    ])

# H2. D*h symm.
# MP2 energy = -1.1441408 Hartree
# Charge = 0, multiplicity = 1
H2 = ListOfAtoms([
    Atom('H', [.000000, .000000, .368583]),
    Atom('H', [.000000, .000000, -.368583]),
    ])

# SH radical, C*v symm.
# MP2 energy = -398.1720853 Hartree
# Charge = 0, multiplicity = 2
SH = ListOfAtoms([
    Atom('S', [.000000, .000000, .079083]),
    Atom('H', [.000000, .000000, -1.265330]),
    ])

# CCH radical, C*v symm.
# MP2 energy = -76.3534702 Hartree
# Charge = 0, multiplicity = 2
HC2 = ListOfAtoms([
    Atom('C', [.000000, .000000, -.462628]),
    Atom('C', [.000000, .000000, .717162]),
    Atom('H', [.000000, .000000, -1.527198]),
    ])

# C2H3 radical, Cs symm, 2-A'.
# MP2 energy = -77.613258 Hartree
# Charge = 0, multiplicity = 2
H3C2 = ListOfAtoms([
    Atom('C', [.049798, -.576272, .000000]),
    Atom('C', [.049798, .710988, .000000]),
    Atom('H', [-.876750, -1.151844, .000000]),
    Atom('H', [.969183, -1.154639, .000000]),
    Atom('H', [-.690013, 1.498185, .000000]),
    ])

# CH3CO radical, HCCO cis, Cs symm, 2-A'.
# MP2 energy = -152.7226518 Hartree
# Charge = 0, multiplicity = 2
OH3C2 = ListOfAtoms([
    Atom('C', [-.978291, -.647814, .000000]),
    Atom('C', [.000000, .506283, .000000]),
    Atom('H', [-.455551, -1.607837, .000000]),
    Atom('H', [-1.617626, -.563271, .881061]),
    Atom('H', [-1.617626, -.563271, -.881061]),
    Atom('O', [1.195069, .447945, .000000]),
    ])

# H2COH radical, C1 symm.
# MP2 energy = -114.7033977 Hartree
# Charge = 0, multiplicity = 2
OH3C = ListOfAtoms([
    Atom('C', [.687448, .029626, -.082014]),
    Atom('O', [-.672094, -.125648, .030405]),
    Atom('H', [-1.091850, .740282, -.095167]),
    Atom('H', [1.122783, .975263, .225993]),
    Atom('H', [1.221131, -.888116, .118015]),
    ])

# CH3O radical, Cs symm, 2-A'.
# MP2 energy = -114.6930929 Hartree
# Charge = 0, multiplicity = 2
OH3Cx = ListOfAtoms([
    Atom('C', [-.008618, -.586475, .000000]),
    Atom('O', [-.008618, .799541, .000000]),
    Atom('H', [1.055363, -.868756, .000000]),
    Atom('H', [-.467358, -1.004363, .903279]),
    Atom('H', [-.467358, -1.004363, -.903279]),
    ])

# CH3CH2O radical, Cs symm, 2-A''.
# MP2 energy = -153.8670598 Hartree
# Charge = 0, multiplicity = 2
OH5C2 = ListOfAtoms([
    Atom('C', [1.004757, -.568263, .000000]),
    Atom('C', [.000000, .588691, .000000]),
    Atom('O', [-1.260062, .000729, .000000]),
    Atom('H', [.146956, 1.204681, .896529]),
    Atom('H', [.146956, 1.204681, -.896529]),
    Atom('H', [2.019363, -.164100, .000000]),
    Atom('H', [.869340, -1.186832, .888071]),
    Atom('H', [.869340, -1.186832, -.888071]),
    ])

# CH3S radical, Cs symm, 2-A'.
# MP2 energy = -437.3459808 Hartree
# Charge = 0, multiplicity = 2
SH3C = ListOfAtoms([
    Atom('C', [-.003856, 1.106222, .000000]),
    Atom('S', [-.003856, -.692579, .000000]),
    Atom('H', [1.043269, 1.427057, .000000]),
    Atom('H', [-.479217, 1.508437, .895197]),
    Atom('H', [-.479217, 1.508437, -.895197]),
    ])

# C2H5 radical, Staggered, Cs symm, 2-A'.
# MP2 energy = -78.8446639 Hartree
# Charge = 0, multiplicity = 2
H5C2 = ListOfAtoms([
    Atom('C', [-.014359, -.694617, .000000]),
    Atom('C', [-.014359, .794473, .000000]),
    Atom('H', [1.006101, -1.104042, .000000]),
    Atom('H', [-.517037, -1.093613, .884839]),
    Atom('H', [-.517037, -1.093613, -.884839]),
    Atom('H', [.100137, 1.346065, .923705]),
    Atom('H', [.100137, 1.346065, -.923705]),
    ])

# (CH3)2CH radical, Cs symm, 2-A'.
# MP2 energy = -118.0192311 Hartree
# Charge = 0, multiplicity = 2
H7C3 = ListOfAtoms([
    Atom('C', [.014223, .543850, .000000]),
    Atom('C', [.014223, -.199742, 1.291572]),
    Atom('C', [.014223, -.199742, -1.291572]),
    Atom('H', [-.322890, 1.575329, .000000]),
    Atom('H', [.221417, .459174, 2.138477]),
    Atom('H', [.221417, .459174, -2.138477]),
    Atom('H', [-.955157, -.684629, 1.484633]),
    Atom('H', [.767181, -.995308, 1.286239]),
    Atom('H', [.767181, -.995308, -1.286239]),
    Atom('H', [-.955157, -.684629, -1.484633]),
    ])

# t-Butyl radical, (CH3)3C, C3v symm.
# MP2 energy = -157.1957937 Hartree
# Charge = 0, multiplicity = 2
H9C4 = ListOfAtoms([
    Atom('C', [.000000, .000000, .191929]),
    Atom('C', [.000000, 1.478187, -.020866]),
    Atom('C', [1.280147, -.739093, -.020866]),
    Atom('C', [-1.280147, -.739093, -.020866]),
    Atom('H', [.000000, 1.731496, -1.093792]),
    Atom('H', [-.887043, 1.945769, .417565]),
    Atom('H', [.887043, 1.945769, .417565]),
    Atom('H', [1.499520, -.865748, -1.093792]),
    Atom('H', [2.128607, -.204683, .417565]),
    Atom('H', [1.241564, -1.741086, .417565]),
    Atom('H', [-1.499520, -.865748, -1.093792]),
    Atom('H', [-1.241564, -1.741086, .417565]),
    Atom('H', [-2.128607, -.204683, .417565]),
    ])

# NO2 radical, C2v symm, 2-A1.
# MP2 energy = -204.5685941 Hartree
# Charge = 0, multiplicity = 2
O2N = ListOfAtoms([
    Atom('N', [.000000, .000000, .332273]),
    Atom('O', [.000000, 1.118122, -.145370]),
    Atom('O', [.000000, -1.118122, -.145370]),
    ])

def get_g2(name, cell):
    if name in atoms:
        loa =  ListOfAtoms([Atom(name, magmom=atoms[name])], cell, False)
    elif name in molecules:
        loa = eval(name)
        loa.SetUnitCell(cell, fix=True)
        loa.SetBoundaryConditions(periodic=False)
    else:
        raise NotImplementedError, 'System %s not in database.' % name

    center(loa)
    return loa.Copy()
