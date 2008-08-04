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
from ase.atoms import Atoms, Atom

raise DeprecationWarning, ('g2 module is outdated. '
                           'Use ase.data.molecules instead')

"""
Atomic enthalpies of formation at 0K, H_exp(OK), and thermal
corrections, H(298)-H(0), from Curtiss et al. JCP 106, 1063 (1997).
"""
atoms = {
    # Atom, magmom, H_exp(0K), H(298K) - H_exp(0K)
    'H' : (1,  51.63, 1.01),
    'Li': (1,  37.69, 1.10),
    'Be': (0,  76.48, 0.46),
    'B' : (1, 136.20, 0.29),
    'C' : (2, 169.98, 0.25),
    'N' : (3, 112.53, 1.04),
    'O' : (2,  58.99, 1.04),
    'F' : (1,  18.47, 1.05),
    'Na': (1,  25.69, 1.54),
    'Mg': (0,  34.87, 1.19),
    'Al': (1,  78.23, 1.08),
    'Si': (2, 106.60, 0.76),
    'P' : (3,  75.42, 1.28),
    'S' : (2,  65.66, 1.05),
    'Cl': (1,  28.59, 1.10),
    }

"""
Experimental enthalpies of formation at 298K, zero-point energies (ZPE), and
thermal enthalpy corrections for molecules from
Staroverov et al. JCP 119, 12129 (2003)

ZPE and thermal corrections are estimated from B3LYP geometries and vibrations.

Data for extra systems are from CCCBDB: http://srdata.nist.gov/cccbdb/
"""
molecules = {
    # variable / key: (Desciption, LaTeX name, H_exp(298K), ZPE, H(298K) - H_exp(0K) )
    # Extra systems
       'Be2': ("Diatomic Beryllium"                                       , r"$\rm{Be}_2$"                                                 ,  155.1,   1.0000, 5.0600),
    # The G1 test set
       'LiH': ("Lithium hydride (LiH), C*v symm."                         , r"$\rm{LiH}$"                                                  ,   33.3,   2.0149, 2.0783),
       'BeH': ("Beryllium hydride (BeH), D*h symm."                       , r"$\rm{BeH}$"                                                  ,   81.7,   2.9073, 2.0739),
        'CH': ("CH radical. Doublet, C*v symm."                           , r"$\rm{CH}$"                                                   ,  142.5,   3.9659, 2.0739),
 'CH2_s3B1d': ("Triplet methylene (CH2), C2v symm, 3-B1."                 , r"$\rm{CH}_2\ (^3B_1)$"                                        ,   93.7,  10.6953, 2.3877),
 'CH2_s1A1d': ("Singlet methylene (CH2), C2v symm, 1-A1."                 , r"$\rm{CH}_2\ (^1A_1)$"                                        ,  102.8,  10.2422, 2.3745),
       'CH3': ("Methyl radical (CH3), D3h symm."                          , r"$\rm{CH}_3$"                                                 ,   35.0,  18.3383, 2.5383),
       'CH4': ("Methane (CH4), Td symm."                                  , r"$\rm{CH}_4$"                                                 ,  -17.9,  27.6744, 2.3939),
        'NH': ("NH, triplet, C*v symm."                                   , r"$\rm{NH}$"                                                   ,   85.2,   4.5739, 2.0739),
       'NH2': ("NH2 radical, C2v symm, 2-B1."                             , r"$\rm{NH}_2$"                                                 ,   45.1,  11.7420, 2.3726),
       'NH3': ("Ammonia (NH3), C3v symm."                                 , r"$\rm{NH}_3$"                                                 ,  -11.0,  21.2462, 2.3896),
        'OH': ("OH radical, C*v symm."                                    , r"$\rm{OH}$"                                                   ,    9.4,   5.2039, 2.0739),
       'H2O': ("Water (H2O), C2v symm."                                   , r"$\rm{H}_2\rm{O}$"                                            ,  -57.8,  13.2179, 2.3720),
        'HF': ("Hydrogen fluoride (HF), C*v symm."                        , r"$\rm{HF}$"                                                   ,  -65.1,   5.7994, 2.0733),
'SiH2_s1A1d': ("Singlet silylene (SiH2), C2v symm, 1-A1."                 , r"$\rm{SiH}_2\ (^1A_1)$"                                       ,   65.2,   7.1875, 2.3927),
'SiH2_s3B1d': ("Triplet silylene (SiH2), C2v symm, 3-B1."                 , r"$\rm{SiH}_2\ (^3B_1)$"                                       ,   86.2,   7.4203, 2.4078),
      'SiH3': ("Silyl radical (SiH3), C3v symm."                          , r"$\rm{SiH}_3$"                                                ,   47.9,  13.0898, 2.4912),
      'SiH4': ("Silane (SiH4), Td symm."                                  , r"$\rm{SiH}_4$"                                                ,    8.2,  19.2664, 2.5232),
       'PH2': ("PH2 radical, C2v symm."                                   , r"$\rm{PH}_2$"                                                 ,   33.1,   8.2725, 2.3845),
       'PH3': ("Phosphine (PH3), C3v symm."                               , r"$\rm{PH}_3$"                                                 ,    1.3,  14.7885, 2.4203),
       'SH2': ("Hydrogen sulfide (H2S), C2v symm."                        , r"$\rm{SH}_2$"                                                 ,   -4.9,   9.3129, 2.3808),
       'HCl': ("Hydrogen chloride (HCl), C*v symm."                       , r"$\rm{HCl}$"                                                  ,  -22.1,   4.1673, 2.0739),
       'Li2': ("Dilithium (Li2), D*h symm."                               , r"$\rm{Li}_2$"                                                 ,   51.6,   0.4838, 2.3086),
       'LiF': ("Lithium Fluoride (LiF), C*v symm."                        , r"$\rm{LiF}$"                                                  ,  -80.1,   1.4019, 2.0990),
      'C2H2': ("Acetylene (C2H2), D*h symm."                              , r"$\rm{C}_2\rm{H}_2$"                                          ,   54.2,  16.6001, 2.4228),
      'C2H4': ("Ethylene (H2C=CH2), D2h symm."                            , r"$\rm{C}_2\rm{H}_4$"                                          ,   12.5,  31.5267, 2.5100),
      'C2H6': ("Ethane (H3C-CH3), D3d symm."                              , r"$\rm{C}_2\rm{H}_6$"                                          ,  -20.1,  46.0950, 2.7912),
        'CN': ("Cyano radical (CN), C*v symm, 2-Sigma+."                  , r"$\rm{CN}$"                                                   ,  104.9,   3.0183, 2.0739),
       'HCN': ("Hydrogen cyanide (HCN), C*v symm."                        , r"$\rm{HCN}$"                                                  ,   31.5,  10.2654, 2.1768),
        'CO': ("Carbon monoxide (CO), C*v symm."                          , r"$\rm{CO}$"                                                   ,  -26.4,   3.1062, 2.0739),
       'HCO': ("HCO radical, Bent Cs symm."                               , r"$\rm{HCO}$"                                                  ,   10.0,   8.0290, 2.3864),
      'H2CO': ("Formaldehyde (H2C=O), C2v symm."                          , r"$\rm{H}_2\rm{CO}$"                                           ,  -26.0,  16.4502, 2.3927),
     'CH3OH': ("Methanol (CH3-OH), Cs symm."                              , r"$\rm{H}_3\rm{COH}$"                                          ,  -48.0,  31.6635, 2.6832),
        'N2': ("N2 molecule, D*h symm."                                   , r"$\rm{N}_2$"                                                  ,    0.0,   3.4243, 2.0733),
      'N2H4': ("Hydrazine (H2N-NH2), C2 symm."                            , r"$\rm{H}_2\rm{NNH}_2$"                                        ,   22.8,  32.9706, 2.6531),
        'NO': ("NO radical, C*v symm, 2-Pi."                              , r"$\rm{NO}$"                                                   ,   21.6,   2.7974, 2.0745),
        'O2': ("O2 molecule, D*h symm, Triplet."                          , r"$\rm{O}_2$"                                                  ,    0.0,   2.3444, 2.0752),
      'H2O2': ("Hydrogen peroxide (HO-OH), C2 symm."                      , r"$\rm{HOOH}$"                                                 ,  -32.5,  16.4081, 2.6230),
        'F2': ("F2 molecule, D*h symm."                                   , r"$\rm{F}_2$"                                                  ,    0.0,   1.5179, 2.0915),
       'CO2': ("Carbon dioxide (CO2), D*h symm."                          , r"$\rm{CO}_2$"                                                 ,  -94.1,   7.3130, 2.2321),
       'Na2': ("Disodium (Na2), D*h symm."                                , r"$\rm{Na}_2$"                                                 ,   34.0,   0.2246, 2.4699),
       'Si2': ("Si2 molecule, D*h symm, Triplet (3-Sigma-G-)."            , r"$\rm{Si}_2$"                                                 ,  139.9,   0.7028, 2.2182),
        'P2': ("P2 molecule, D*h symm."                                   , r"$\rm{P}_2$"                                                  ,   34.3,   1.1358, 2.1235),
        'S2': ("S2 molecule, D*h symm, triplet."                          , r"$\rm{S}_2$"                                                  ,   30.7,   1.0078, 2.1436),
       'Cl2': ("Cl2 molecule, D*h symm."                                  , r"$\rm{Cl}_2$"                                                 ,    0.0,   0.7737, 2.1963),
      'NaCl': ("Sodium Chloride (NaCl), C*v symm."                        , r"$\rm{NaCl}$"                                                 ,  -43.6,   0.5152, 2.2935),
       'SiO': ("Silicon monoxide (SiO), C*v symm."                        , r"$\rm{SiO}$"                                                  ,  -24.6,   1.7859, 2.0821),
        'CS': ("Carbon monosulfide (CS), C*v symm."                       , r"$\rm{SC}$"                                                   ,   66.9,   1.8242, 2.0814),
        'SO': ("Sulfur monoxide (SO), C*v symm, triplet."                 , r"$\rm{SO}$"                                                   ,    1.2,   1.6158, 2.0877),
       'ClO': ("ClO radical, C*v symm, 2-PI."                             , r"$\rm{ClO}$"                                                  ,   24.2,   1.1923, 2.1172),
       'ClF': ("ClF molecule, C*v symm, 1-SG."                            , r"$\rm{FCl}$"                                                  ,  -13.2,   1.1113, 2.1273),
     'Si2H6': ("Disilane (H3Si-SiH3), D3d symm."                          , r"$\rm{Si}_2\rm{H}_6$"                                         ,   19.1,  30.2265, 3.7927),
     'CH3Cl': ("Methyl chloride (CH3Cl), C3v symm."                       , r"$\rm{CH}_3\rm{Cl}$"                                          ,  -19.6,  23.3013, 2.4956),
     'CH3SH': ("Methanethiol (H3C-SH), Staggered, Cs symm."               , r"$\rm{H}_3\rm{CSH}$"                                          ,   -5.5,  28.3973, 2.8690),
      'HOCl': ("HOCl molecule, Cs symm."                                  , r"$\rm{HOCl}$"                                                 ,  -17.8,   8.1539, 2.4416),
       'SO2': ("Sulfur dioxide (SO2), C2v symm."                          , r"$\rm{SO}_2$"                                                 ,  -71.0,   4.3242, 2.5245),
       # The G2 test set:
       'BF3': ("BF3, Planar D3h symm."                                    , r"$\rm{BF}_3$"                                                 , -271.4,   7.8257, 2.7893),
      'BCl3': ("BCl3, Planar D3h symm."                                   , r"$\rm{BCl}_3$"                                                ,  -96.3,   4.6536, 3.3729),
      'AlF3': ("AlF3, Planar D3h symm."                                   , r"$\rm{AlF}_3$"                                                , -289.0,   4.8645, 3.3986),
     'AlCl3': ("AlCl3, Planar D3h symm."                                  , r"$\rm{AlCl}_3$"                                               , -139.7,   2.9687, 3.9464),
       'CF4': ("CF4, Td symm."                                            , r"$\rm{CF}_4$"                                                 , -223.0,  10.5999, 3.0717),
      'CCl4': ("CCl4, Td symm."                                           , r"$\rm{CCl}_4$"                                                ,  -22.9,   5.7455, 4.1754),
       'OCS': ("O=C=S, Linear, C*v symm."                                 , r"$\rm{COS}$"                                                  ,  -33.1,   5.7706, 2.3663),
       'CS2': ("CS2, Linear, D*h symm."                                   , r"$\rm{CS}_2$"                                                 ,   28.0,   4.3380, 2.5326),
      'COF2': ("COF2, C2v symm."                                          , r"$\rm{COF}_2$"                                                , -149.1,   8.8215, 2.6619),
      'SiF4': ("SiF4, Td symm."                                           , r"$\rm{SiF}_4$"                                                , -386.0,   7.8771, 3.7054),
     'SiCl4': ("SiCl4, Td symm."                                          , r"$\rm{SiCl}_4$"                                               , -158.4,   4.4396, 4.7182),
       'N2O': ("N2O, Cs symm."                                            , r"$\rm{N}_2\rm{O}$"                                            ,   19.6,   6.9748, 2.2710),
      'ClNO': ("ClNO, Cs symm."                                           , r"$\rm{ClNO}$"                                                 ,   12.4,   4.0619, 2.7039),
       'NF3': ("NF3, C3v symm."                                           , r"$\rm{NF}_3$"                                                 ,  -31.6,   6.4477, 2.8301),
       'PF3': ("PF3, C3v symm."                                           , r"$\rm{PF}_3$"                                                 , -229.1,   5.2981, 3.1288),
        'O3': ("O3 (Ozone), C2v symm."                                    , r"$\rm{O}_3$"                                                  ,   34.1,   4.6178, 2.4479),
       'F2O': ("F2O, C2v symm."                                           , r"$\rm{F}_2\rm{O}$"                                            ,    5.9,   3.4362, 2.5747),
      'ClF3': ("ClF3, C2v symm."                                          , r"$\rm{ClF}_3$"                                                ,  -38.0,   4.2922, 3.3289),
      'C2F4': ("C2F4 (F2C=CF2), D2H symm."                                , r"$\rm{C}_2\rm{F}_4$"                                          , -157.4,  13.4118, 3.9037),
     'C2Cl4': ("C2Cl4 (Cl2C=CCl2), D2h symm."                             , r"$\rm{C}_2\rm{Cl}_4$"                                         ,   -3.0,   9.4628, 4.7132),
     'CF3CN': ("CF3CN, C3v symm."                                         , r"$\rm{CF}_3\rm{CN}$"                                          , -118.4,  14.1020, 3.7996),
  'C3H4_C3v': ("Propyne (C3H4), C3v symm."                                , r"$\rm{CH}_3\rm{CCH}\ \rm{(propyne)}$"                         ,   44.2,  34.2614, 3.1193),
  'C3H4_D2d': ("Allene (C3H4), D2d symm."                                 , r"$\rm{CH}_2\rm{=C=CH}_2\ \rm{(allene)}$"                      ,   45.5,  34.1189, 2.9744),
  'C3H4_C2v': ("Cyclopropene (C3H4), C2v symm."                           , r"$\rm{C}_3\rm{H}_4\ \rm{(cyclopropene)}$"                     ,   66.2,  34.7603, 2.6763),
   'C3H6_Cs': ("Propene (C3H6), Cs symm."                                 , r"$\rm{CH}_3\rm{CH=CH}_2\ \rm{(propylene)}$"                   ,    4.8,  49.1836, 3.1727),
  'C3H6_D3h': ("Cyclopropane (C3H6), D3h symm."                           , r"$\rm{C}_3\rm{H}_6\ \rm{(cyclopropane)}$"                     ,   12.7,  50.2121, 2.7272),
      'C3H8': ("Propane (C3H8), C2v symm."                                , r"$\rm{C}_3\rm{H}_8\ \rm{(propane)}$"                          ,  -25.0,  63.8008, 3.4632),
     'C4H6x': ("Trans-1,3-butadiene (C4H6), C2h symm."                    , r"$\rm{CH}_2\rm{CHCHCH}_2\ \rm{(butadiene)}$"                  ,   26.3,  52.6273, 3.5341),
    'C4H6xx': ("Dimethylacetylene (2-butyne, C4H6), D3h symm (eclipsed)." , r"$\rm{C}_4\rm{H}_6\ \rm{(2-butyne)}$"                         ,   34.8,  51.8731, 4.2344),
   'C4H6xxx': ("Methylenecyclopropane (C4H6), C2v symm."                  , r"$\rm{C}_4\rm{H}_6\ \rm{(methylene cyclopropane)}$"           ,   47.9,  52.6230, 3.2881),
  'C4H6xxxx': ("Bicyclo[1.1.0]butane (C4H6), C2v symm."                   , r"$\rm{C}_4\rm{H}_6\ \rm{(bicyclobutane)}$"                    ,   51.9,  53.3527, 2.9637),
 'C4H6xxxxx': ("Cyclobutene (C4H6), C2v symm."                            , r"$\rm{C}_4\rm{H}_6\ \rm{(cyclobutene)}$"                      ,   37.4,  53.4105, 3.0108),
     'C4H8x': ("Cyclobutane (C4H8), D2d symm."                            , r"$\rm{C}_4\rm{H}_8\ \rm{(cyclobutane)}$"                      ,    6.8,  68.3314, 3.2310),
    'C4H8xx': ("Isobutene (C4H8), Single bonds trans, C2v symm."          , r"$\rm{C}_4\rm{H}_8\ \rm{(isobutene)}$"                        ,   -4.0,  66.5693, 3.9495),
    'C4H10x': ("Trans-butane (C4H10), C2h symm."                          , r"$\rm{C}_4\rm{H}_{10}\ \rm{(trans butane)}$"                  ,  -30.0,  81.3980, 4.2633),
   'C4H10xx': ("Isobutane (C4H10), C3v symm."                             , r"$\rm{C}_4\rm{H}_{10}\ \rm{(isobutane)}$"                     ,  -32.1,  81.1050, 4.2282),
      'C5H8': ("Spiropentane (C5H8), D2d symm."                           , r"$\rm{C}_5\rm{H}_8\ \rm{(spiropentane)}$"                     ,   44.3,  70.9964, 3.7149),
      'C6H6': ("Benzene (C6H6), D6h symm."                                , r"$\rm{C}_6\rm{H}_6\ \rm{(benzene)}$"                          ,   19.7,  61.9252, 3.3886),
     'H2CF2': ("Difluoromethane (H2CF2), C2v symm."                       , r"$\rm{CH}_2\rm{F}_2$"                                         , -107.7,  20.2767, 2.5552),
      'HCF3': ("Trifluoromethane (HCF3), C3v symm."                       , r"$\rm{CHF}_3$"                                                , -166.6,  15.7072, 2.7717),
    'H2CCl2': ("Dichloromethane (H2CCl2), C2v symm."                      , r"$\rm{CH}_2\rm{Cl}_2$"                                        ,  -22.8,  18.0930, 2.8527),
     'HCCl3': ("Chloroform (HCCl3), C3v symm."                            , r"$\rm{CHCl}_3$"                                               ,  -24.7,  12.1975, 3.4262),
    'H3CNH2': ("Methylamine (H3C-NH2), Cs symm."                          , r"$\rm{CH}_3\rm{NH}_2\ \rm{(methylamine)}$"                    ,   -5.5,  39.5595, 2.7428),
     'CH3CN': ("Acetonitrile (CH3-CN), C3v symm."                         , r"$\rm{CH}_3\rm{CN}\ \rm{(methyl cyanide)}$"                   ,   18.0,  28.0001, 2.8552),
    'CH3NO2': ("Nitromethane (CH3-NO2), Cs symm."                         , r"$\rm{CH}_3\rm{NO}_2\ \rm{(nitromethane)}$"                   ,  -17.8,  30.7568, 2.7887),
    'CH3ONO': ("Methylnitrite (CH3-O-N=O), NOCH trans, ONOC cis, Cs symm.", r"$\rm{CH}_3\rm{ONO}\ \rm{(methyl nitrite)}$"                  ,  -15.9,  29.9523, 3.3641),
   'CH3SiH3': ("Methylsilane (CH3-SiH3), C3v symm."                       , r"$\rm{CH}_3\rm{SiH}_3\ \rm{(methyl silane)}$"                 ,   -7.0,  37.6606, 3.2486),
     'HCOOH': ("Formic Acid (HCOOH), HOCO cis, Cs symm."                  , r"$\rm{HCOOH}\ \rm{(formic acid)}$"                            ,  -90.5,  20.9525, 2.5853),
   'HCOOCH3': ("Methyl formate (HCOOCH3), Cs symm."                       , r"$\rm{HCOOCH}_3\ \rm{(methyl formate)}$"                      ,  -85.0,  38.3026, 3.4726),
  'CH3CONH2': ("Acetamide (CH3CONH2), C1 symm."                           , r"$\rm{CH}_3\rm{CONH}_2\ \rm{(acetamide)}$"                    ,  -57.0,  45.2566, 3.9313),
  'CH2NHCH2': ("Aziridine (cyclic CH2-NH-CH2 ring), C2v symm."            , r"$\rm{C}_2\rm{H}_4\rm{NH}\ \rm{(aziridine)}$"                 ,   30.2,  43.3728, 2.6399),
      'NCCN': ("Cyanogen (NCCN). D*h symm."                               , r"$\rm{NCCN}\ \rm{(cyanogen)}$"                                ,   73.3,  10.2315, 2.9336),
    'C2H6NH': ("Dimethylamine, (CH3)2NH, Cs symm."                        , r"$\rm{(CH}_3\rm{)}_2\rm{NH}\ \rm{(dimethylamine)}$"           ,   -4.4,  57.0287, 3.3760),
 'CH3CH2NH2': ("Trans-Ethylamine (CH3-CH2-NH2), Cs symm."                 , r"$\rm{CH}_3\rm{CH}_2\rm{NH}_2\ \rm{(trans ethylamine)}$"      ,  -11.3,  57.2420, 3.3678),
     'H2CCO': ("Ketene (H2C=C=O), C2v symm."                              , r"$\rm{CH}_2\rm{CO}\ \rm{(ketene)}$"                           ,  -11.4,  19.5984, 2.8075),
   'CH2OCH2': ("Oxirane (cyclic CH2-O-CH2 ring), C2v symm."               , r"$\rm{C}_2\rm{H}_4\rm{O}\ \rm{(oxirane)}$"                    ,  -12.6,  35.4204, 2.5816),
    'CH3CHO': ("Acetaldehyde (CH3CHO), Cs symm."                          , r"$\rm{CH}_3\rm{CHO}\ \rm{(acetaldehyde)}$"                    ,  -39.7,  34.2288, 3.0428),
    'OCHCHO': ("Glyoxal (O=CH-CH=O). Trans, C2h symm."                    , r"$\rm{HCOCOH}\ \rm{(glyoxal)}$"                               ,  -50.7,  22.8426, 3.2518),
  'CH3CH2OH': ("Ethanol (trans, CH3CH2OH), Cs symm."                      , r"$\rm{CH}_3\rm{CH}_2\rm{OH}\ \rm{(ethanol)}$"                 ,  -56.2,  49.3072, 3.3252),
   'CH3OCH3': ("DimethylEther (CH3-O-CH3), C2v symm."                     , r"$\rm{CH}_3\rm{OCH}_3\ \rm{(dimethylether)}$"                 ,  -44.0,  49.1911, 3.3139),
   'CH2SCH2': ("Thiooxirane (cyclic CH2-S-CH2 ring), C2v symm."           , r"$\rm{C}_2\rm{H}_4\rm{S}\ \rm{(thiirane)}$"                   ,   19.6,  33.9483, 2.7290),
    'C2H6SO': ("Dimethylsulfoxide (CH3)2SO, Cs symm."                     , r"$\rm{(CH}_3\rm{)}_2\rm{SO}\ \rm{(dimethyl sulfoxide)}$"      ,  -36.2,  48.8479, 4.1905),
  'CH3CH2SH': ("ThioEthanol (CH3-CH2-SH), Cs symm."                       , r"$\rm{C}_2\rm{H}_5\rm{SH}\ \rm{(ethanethiol)}$"               ,  -11.1,  46.1583, 3.5900),
   'CH3SCH3': ("Dimethyl ThioEther (CH3-S-CH3), C2v symm."                , r"$\rm{CH}_3\rm{SCH}_3\ \rm{(dimethyl sulfide)}$"              ,   -8.9,  46.6760, 3.6929),
    'H2CCHF': ("Vinyl fluoride (H2C=CHF), Cs symm."                       , r"$\rm{CH}_2\rm{=CHF}\ \rm{(vinyl fluoride)}$"                 ,  -33.2,  27.2785, 2.7039),
  'CH3CH2Cl': ("Ethyl chloride (CH3-CH2-Cl), Cs symm."                    , r"$\rm{C}_2\rm{H}_5\rm{Cl}\ \rm{(ethyl chloride)}$"            ,  -26.8,  41.0686, 3.1488),
   'H2CCHCl': ("Vinyl chloride, H2C=CHCl, Cs symm."                       , r"$\rm{CH}_2\rm{=CHCl}\ \rm{(vinyl chloride)}$"                ,    8.9,  26.3554, 2.8269),
   'H2CCHCN': ("CyanoEthylene (H2C=CHCN), Cs symm."                       , r"$\rm{CH}_2\rm{=CHCN}\ \rm{(acrylonitrile)}$"                 ,   43.2,  31.4081, 3.2034),
  'CH3COCH3': ("Acetone (CH3-CO-CH3), C2v symm."                          , r"$\rm{CH}_3\rm{COCH}_3\ \rm{(acetone)}$"                      ,  -51.9,  51.5587, 3.9878),
   'CH3COOH': ("Acetic Acid (CH3COOH), Single bonds trans, Cs symm."      , r"$\rm{CH}_3\rm{COOH}\ \rm{(acetic acid)}$"                    , -103.4,  38.1670, 3.4770),
    'CH3COF': ("Acetyl fluoride (CH3COF), HCCO cis, Cs symm."             , r"$\rm{CH}_3\rm{COF}\ \rm{(acetyl fluoride)}$"                 , -105.7,  30.2742, 3.3126),
   'CH3COCl': ("Acetyl,Chloride (CH3COCl), HCCO cis, Cs symm."            , r"$\rm{CH}_3\rm{COCl}\ \rm{(acetyl chloride)}$"                ,  -58.0,  29.1855, 3.5235),
    'C3H7Cl': ("Propyl chloride (CH3CH2CH2Cl), Cs symm."                  , r"$\rm{CH}_3\rm{CH}_2\rm{CH}_2\rm{Cl}\ \rm{(propyl chloride)}$",  -31.5,  58.6696, 3.9885),
  'C2H6CHOH': ("Isopropyl alcohol, (CH3)2CH-OH, Gauche isomer, C1 symm."  , r"$\rm{(CH}_3\rm{)}_2\rm{CHOH}\ \rm{(isopropanol)}$"           ,  -65.2,  66.5612, 4.0732),
'CH3CH2OCH3': ("Methyl ethyl ether (CH3-CH2-O-CH3), Trans, Cs symm."      , r"$\rm{C}_2\rm{H}_5\rm{OCH}_3\ \rm{(methyl ethyl ether)}$"     ,  -51.7,  66.6936, 4.1058),
     'C3H9N': ("Trimethyl Amine, (CH3)3N, C3v symm."                      , r"$\rm{(CH}_3\rm{)}_3\rm{N}\ \rm{(trimethylamine)}$"           ,   -5.7,  74.1584, 4.0631),
     'C4H4O': ("Furan (cyclic C4H4O), C2v symm."                          , r"$\rm{C}_4\rm{H}_4\rm{O}\ \rm{(furan)}$"                      ,   -8.3,  43.2116, 2.9480),
     'C4H4S': ("Thiophene (cyclic C4H4S), C2v symm."                      , r"$\rm{C}_4\rm{H}_4\rm{S}\ \rm{(thiophene)}$"                  ,   27.5,  41.2029, 3.1702),
    'C4H4NH': ("Pyrrole (Planar cyclic C4H4NH), C2v symm."                , r"$\rm{C}_4\rm{H}_5\rm{N}\ \rm{(pyrrole)}$"                    ,   25.9,  50.9688, 3.1156),
     'C5H5N': ("Pyridine (cyclic C5H5N), C2v symm."                       , r"$\rm{C}_5\rm{H}_5\rm{N}\ \rm{(pyridine)}$"                   ,   33.6,  54.8230, 3.3007),
        'H2': ("H2. D*h symm."                                            , r"$\rm{H}_2$"                                                  ,    0.0,   6.2908, 2.0739),
        'SH': ("SH radical, C*v symm."                                    , r"$\rm{HS}$"                                                   ,   34.2,   3.7625, 2.0739),
       'CCH': ("CCH radical, C*v symm."                                   , r"$\rm{CCH}$"                                                  ,  135.1,   7.8533, 2.7830),
      'C2H3': ("C2H3 radical, Cs symm, 2-A'."                             , r"$\rm{C}_2\rm{H}_3\ \rm{(2A')}$"                              ,   71.6,  22.5747, 2.5483),
     'CH3CO': ("CH3CO radical, HCCO cis, Cs symm, 2-A'."                  , r"$\rm{CH}_3\rm{CO}\ \rm{(2A')}$"                              ,   -2.4,  26.6070, 3.0842),
     'H2COH': ("H2COH radical, C1 symm."                                  , r"$\rm{H}_2\rm{COH}\ \rm{(2A)}$"                               ,   -4.1,  23.1294, 2.6726),
      'CH3O': ("CH3O radical, Cs symm, 2-A'."                             , r"$\rm{CH}_3\rm{O}\ \rm{CS (2A')}$"                            ,    4.1,  22.4215, 2.4969),
   'CH3CH2O': ("CH3CH2O radical, Cs symm, 2-A''."                         , r"$\rm{CH}_3\rm{CH}_2\rm{O}\ \rm{(2A'')}$"                     ,   -3.7,  39.4440, 3.0158),
      'CH3S': ("CH3S radical, Cs symm, 2-A'."                             , r"$\rm{CH}_3\rm{S}\ \rm{(2A')}$"                               ,   29.8,  21.9415, 2.6054),
      'C2H5': ("C2H5 radical, Staggered, Cs symm, 2-A'."                  , r"$\rm{C}_2\rm{H}_5\ \rm{(2A')}$"                              ,   28.9,  36.5675, 3.0942),
      'C3H7': ("(CH3)2CH radical, Cs symm, 2-A'."                         , r"$\rm{(CH}_3\rm{)}_2\rm{CH}\ \rm{(2A')}$"                     ,   21.5,  54.2928, 3.8435),
     'C3H9C': ("t-Butyl radical, (CH3)3C, C3v symm."                      , r"$\rm{(CH}_3\rm{)}_3\rm{C}\ \rm{(t-butyl radical)}$"          ,   12.3,  71.7833, 4.6662),
       'NO2': ("NO2 radical, C2v symm, 2-A1."                             , r"$\rm{NO}_2$"                                                 ,    7.9,   5.4631, 2.4366),
}

"""
Experimental ionization energies from CCCBDB at
http://srdata.nist.gov/cccbdb/default.htm
"""
IP = {# System     IE    IE_vert
    'H'         : (13.60,  None),
    'Li'        : ( 5.39,  None),
    'Be'        : ( 9.32,  None),
    'B'         : ( 8.30,  None),
    'C'         : (11.26,  None),
    'N'         : (14.53,  None),
    'O'         : (13.62,  None),
    'F'         : (17.42,  None),
    'Na'        : ( 5.14,  None),
    'Mg'        : ( 7.65,  None),
    'Al'        : ( 5.99,  None),
    'Si'        : ( 8.15,  None),
    'P'         : (10.49,  None),
    'S'         : (10.36,  None),
    'Cl'        : (12.97,  None),      
    'LiH'       : ( 7.90,  None),
    'BeH'       : ( 8.21,  None),
    'CH'        : (10.64,  None),  
    'CH2_s3B1d' : (10.40,  None),
    'CH3'       : ( 9.84,  None),
    'CH4'       : (12.61, 13.60),
    'NH'        : (13.10, 13.49),
    'NH2'       : (10.78, 12.00),
    'NH3'       : (10.07, 10.82),
    'OH'        : (13.02,  None),
    'H2O'       : (12.62,  None),
    'HF'        : (16.03, 16.12),
    'SiH2_s1A1d': ( 8.92,  None),
    'SiH3'      : ( 8.14,  8.74),
    'SiH4'      : (11.00, 12.30),
    'PH2'       : ( 9.82,  None),
    'PH3'       : ( 9.87, 10.95),
    'SH2'       : (10.46, 10.50),
    'HCl'       : (12.74,  None),
    'Li2'       : ( 5.11,  None),
    'LiF'       : (11.30,  None),
    'C2H2'      : (11.40, 11.49),
    'C2H4'      : (10.51, 10.68),
    'CN'        : (13.60,  None),
    'HCN'       : (13.60, 13.61),
    'CO'        : (14.01, 14.01),
    'HCO'       : ( 8.12,  9.31),
    'H2CO'      : (10.88, 10.88),
    'CH3OH'     : (10.84, 10.96),
    'N2'        : (15.58, 15.58),
    'N2H4'      : ( 8.10,  8.98),
    'NO'        : ( 9.26,  9.26),
    'O2'        : (12.07, 12.30),
    'H2O2'      : (10.58, 11.70),
    'F2'        : (15.70, 15.70),
    'CO2'       : (13.78, 13.78),
    'Na2'       : ( 4.89,  None),
    'Si2'       : ( 7.90,  None),
    'P2'        : (10.53, 10.62),
    'S2'        : ( 9.36,  9.55),
    'Cl2'       : (11.48, 11.49),
    'NaCl'      : ( 9.20,  9.80),
    'SiO'       : (11.49,  None),
    'CS'        : (11.33,  None),
    'SO'        : (11.29,  None),
    'ClO'       : (10.89, 11.01),
    'ClF'       : (12.66, 12.77),
    'Si2H6'     : ( 9.74, 10.53),
    'CH3Cl'     : (11.26, 11.29),
    'CH3SH'     : ( 9.44,  9.44),
    'HOCl'      : (11.12,  None),
    'SO2'       : (12.35, 12.50),
    }  

## Start of extra systems

# Diatomic Beryllium (Be2), D*h symm.
# MP2 energy = -29.204047 Hartree
# Charge = 0, multiplicity = 1
Be2 = Atoms([
    Atom('Be', [.000000, .000000, 1.010600]),
    Atom('Be', [.000000, .000000, -1.010600]),
    ])

## Start of G2-1 test set

# Lithium hydride (LiH), C*v symm.
# MP2 energy = -7.9965108 Hartree
# Charge = 0, multiplicity = 1
LiH = Atoms([
    Atom('Li', [.000000, .000000, .410000]),
    Atom('H', [.000000, .000000, -1.230000]),
    ])

# Beryllium hydride (BeH), D*h symm.
# MP2 energy = -15.171409 Hartree
# Charge = 0, multiplicity = 2
BeH = Atoms([
    Atom('Be', [.000000, .000000, .269654], magmom=1.),
    Atom('H', [.000000, .000000, -1.078616], magmom=0.),
    ])

# CH radical. Doublet, C*v symm.
# MP2 energy = -38.3423986 Hartree
# Charge = 0, multiplicity = 2
CH = Atoms([
    Atom('C', [.000000, .000000, .160074], magmom=1.),
    Atom('H', [.000000, .000000, -.960446], magmom=0.),
    ])

# Triplet methylene (CH2), C2v symm, 3-B1.
# MP2 energy = -39.0074352 Hartree
# Charge = 0, multiplicity = 3
CH2_s3B1d = Atoms([
    Atom('C', [.000000, .000000, .110381], magmom=2.),
    Atom('H', [.000000, .982622, -.331142], magmom=0.),
    Atom('H', [.000000, -.982622, -.331142], magmom=0.),
    ])

# Singlet methylene (CH2), C2v symm, 1-A1.
# MP2 energy = -38.9740078 Hartree
# Charge = 0, multiplicity = 1
CH2_s1A1d = Atoms([
    Atom('C', [.000000, .000000, .174343]),
    Atom('H', [.000000, .862232, -.523029]),
    Atom('H', [.000000, -.862232, -.523029]),
    ])

# Methyl radical (CH3), D3h symm.
# MP2 energy = -39.6730312 Hartree
# Charge = 0, multiplicity = 2
CH3 = Atoms([
    Atom('C', [.000000, .000000, .000000], magmom=1.),
    Atom('H', [.000000, 1.078410, .000000], magmom=0.),
    Atom('H', [.933930, -.539205, .000000], magmom=0.),
    Atom('H', [-.933930, -.539205, .000000], magmom=0.),
    ])

# Methane (CH4), Td symm.
# MP2 energy = -40.3370426 Hartree
# Charge = 0, multiplicity = 1
CH4 = Atoms([
    Atom('C', [.000000, .000000, .000000]),
    Atom('H', [.629118, .629118, .629118]),
    Atom('H', [-.629118, -.629118, .629118]),
    Atom('H', [.629118, -.629118, -.629118]),
    Atom('H', [-.629118, .629118, -.629118]),
    ])

# NH, triplet, C*v symm.
# MP2 energy = -55.0614242 Hartree
# Charge = 0, multiplicity = 3
NH = Atoms([
    Atom('N', [.000000, .000000, .129929], magmom=2.),
    Atom('H', [.000000, .000000, -.909501], magmom=0.),
    ])

# NH2 radical, C2v symm, 2-B1.
# MP2 energy = -55.6937452 Hartree
# Charge = 0, multiplicity = 2
NH2 = Atoms([
    Atom('N', [.000000, .000000, .141690], magmom=1.),
    Atom('H', [.000000, .806442, -.495913], magmom=0.),
    Atom('H', [.000000, -.806442, -.495913], magmom=0.),
    ])

# Ammonia (NH3), C3v symm.
# MP2 energy = -56.3573777 Hartree
# Charge = 0, multiplicity = 1
NH3 = Atoms([
    Atom('N', [.000000, .000000, .116489]),
    Atom('H', [.000000, .939731, -.271808]),
    Atom('H', [.813831, -.469865, -.271808]),
    Atom('H', [-.813831, -.469865, -.271808]),
    ])

# OH radical, C*v symm.
# MP2 energy = -75.5232063 Hartree
# Charge = 0, multiplicity = 2
OH = Atoms([
    Atom('O', [.000000, .000000, .108786], magmom=0.5),
    Atom('H', [.000000, .000000, -.870284], magmom=0.5),
    ])

# Water (H2O), C2v symm.
# MP2 energy = -76.1992442 Hartree
# Charge = 0, multiplicity = 1
H2O = Atoms([
    Atom('O', [.000000, .000000, .119262]),
    Atom('H', [.000000, .763239, -.477047]),
    Atom('H', [.000000, -.763239, -.477047]),
    ])

# Hydrogen fluoride (HF), C*v symm.
# MP2 energy = -100.1841614 Hartree
# Charge = 0, multiplicity = 1
HF = Atoms([
    Atom('F', [.000000, .000000, .093389]),
    Atom('H', [.000000, .000000, -.840502]),
    ])

# Singlet silylene (SiH2), C2v symm, 1-A1.
# MP2 energy = -290.0772034 Hartree
# Charge = 0, multiplicity = 1
SiH2_s1A1d = Atoms([
    Atom('Si', [.000000, .000000, .131272]),
    Atom('H', [.000000, 1.096938, -.918905]),
    Atom('H', [.000000, -1.096938, -.918905]),
    ])

# Triplet silylene (SiH2), C2v symm, 3-B1.
# MP2 energy = -290.0561783 Hartree
# Charge = 0, multiplicity = 3
SiH2_s3B1d = Atoms([
    Atom('Si', [.000000, .000000, .094869], magmom=2.),
    Atom('H', [.000000, 1.271862, -.664083], magmom=0.),
    Atom('H', [.000000, -1.271862, -.664083], magmom=0.),
    ])

# Silyl radical (SiH3), C3v symm.
# MP2 energy = -290.6841563 Hartree
# Charge = 0, multiplicity = 2
SiH3 = Atoms([
    Atom('Si', [.000000, .000000, .079299], magmom=1.),
    Atom('H', [.000000, 1.413280, -.370061], magmom=0.),
    Atom('H', [1.223937, -.706640, -.370061], magmom=0.),
    Atom('H', [-1.223937, -.706640, -.370061], magmom=0.),
    ])

# Silane (SiH4), Td symm.
# MP2 energy = -291.3168497 Hartree
# Charge = 0, multiplicity = 1
SiH4 = Atoms([
    Atom('Si', [.000000, .000000, .000000]),
    Atom('H', [.856135, .856135, .856135]),
    Atom('H', [-.856135, -.856135, .856135]),
    Atom('H', [-.856135, .856135, -.856135]),
    Atom('H', [.856135, -.856135, -.856135]),
    ])

# PH2 radical, C2v symm.
# MP2 energy = -341.9457892 Hartree
# Charge = 0, multiplicity = 2
PH2 = Atoms([
    Atom('P', [.000000, .000000, .115396], magmom=1.),
    Atom('H', [.000000, 1.025642, -.865468], magmom=0.),
    Atom('H', [.000000, -1.025642, -.865468], magmom=0.),
    ])

# Phosphine (PH3), C3v symm.
# MP2 energy = -342.562259 Hartree
# Charge = 0, multiplicity = 1
PH3 = Atoms([
    Atom('P', [.000000, .000000, .124619]),
    Atom('H', [.000000, 1.200647, -.623095]),
    Atom('H', [1.039791, -.600323, -.623095]),
    Atom('H', [-1.039791, -.600323, -.623095]),
    ])

# Hydrogen sulfide (H2S), C2v symm.
# MP2 energy = -398.7986975 Hartree
# Charge = 0, multiplicity = 1
SH2 = Atoms([
    Atom('S', [.000000, .000000, .102135]),
    Atom('H', [.000000, .974269, -.817083]),
    Atom('H', [.000000, -.974269, -.817083]),
    ])

# Hydrogen chloride (HCl), C*v symm.
# MP2 energy = -460.2021493 Hartree
# Charge = 0, multiplicity = 1
HCl = Atoms([
    Atom('Cl', [.000000, .000000, .071110]),
    Atom('H', [.000000, .000000, -1.208868]),
    ])

# Dilithium (Li2), D*h symm.
# MP2 energy = -14.8868485 Hartree
# Charge = 0, multiplicity = 1
Li2 = Atoms([
    Atom('Li', [.000000, .000000, 1.386530]),
    Atom('Li', [.000000, .000000, -1.386530]),
    ])

# Lithium Fluoride (LiF), C*v symm.
# MP2 energy = -107.1294652 Hartree
# Charge = 0, multiplicity = 1
LiF = Atoms([
    Atom('Li', [.000000, .000000, -1.174965]),
    Atom('F', [.000000, .000000, .391655]),
    ])

# Acetylene (C2H2), D*h symm.
# MP2 energy = -77.0762154 Hartree
# Charge = 0, multiplicity = 1
C2H2 = Atoms([
    Atom('C', [.000000, .000000, .608080]),
    Atom('C', [.000000, .000000, -.608080]),
    Atom('H', [.000000, .000000, -1.673990]),
    Atom('H', [.000000, .000000, 1.673990]),
    ])

# Ethylene (H2C=CH2), D2h symm.
# MP2 energy = -78.2942862 Hartree
# Charge = 0, multiplicity = 1
C2H4 = Atoms([
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
C2H6 = Atoms([
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
CN = Atoms([
    Atom('C', [.000000, .000000, -.611046], magmom=1.),
    Atom('N', [.000000, .000000, .523753], magmom=0.),
    ])

# Hydrogen cyanide (HCN), C*v symm.
# MP2 energy = -93.1669402 Hartree
# Charge = 0, multiplicity = 1
HCN = Atoms([
    Atom('C', [.000000, .000000, -.511747]),
    Atom('N', [.000000, .000000, .664461]),
    Atom('H', [.000000, .000000, -1.580746]),
    ])

# Carbon monoxide (CO), C*v symm.
# MP2 energy = -113.0281795 Hartree
# Charge = 0, multiplicity = 1
CO = Atoms([
    Atom('O', [.000000, .000000, .493003]),
    Atom('C', [.000000, .000000, -.657337]),
    ])

# HCO radical, Bent Cs symm.
# MP2 energy = -113.540332 Hartree
# Charge = 0, multiplicity = 2
HCO = Atoms([
    Atom('C', [.062560, .593926, .000000], magmom=1.),
    Atom('O', [.062560, -.596914, .000000], magmom=0.),
    Atom('H', [-.875835, 1.211755, .000000], magmom=0.),
    ])

# Formaldehyde (H2C=O), C2v symm.
# MP2 energy = -114.1749578 Hartree
# Charge = 0, multiplicity = 1
H2CO = Atoms([
    Atom('O', [.000000, .000000, .683501]),
    Atom('C', [.000000, .000000, -.536614]),
    Atom('H', [.000000, .934390, -1.124164]),
    Atom('H', [.000000, -.934390, -1.124164]),
    ])

# Methanol (CH3-OH), Cs symm.
# MP2 energy = -115.3532948 Hartree
# Charge = 0, multiplicity = 1
CH3OH = Atoms([
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
N2 = Atoms([
    Atom('N', [.000000, .000000, .564990]),
    Atom('N', [.000000, .000000, -.564990]),
    ])

# Hydrazine (H2N-NH2), C2 symm.
# MP2 energy = -111.5043953 Hartree
# Charge = 0, multiplicity = 1
N2H4 = Atoms([
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
NO = Atoms([
    Atom('N', [.000000, .000000, -.609442], magmom=0.6),
    Atom('O', [.000000, .000000, .533261], magmom=0.4),
    ])

# O2 molecule, D*h symm, Triplet.
# MP2 energy = -149.9543197 Hartree
# Charge = 0, multiplicity = 3
O2 = Atoms([
    Atom('O', [.000000, .000000, .622978], magmom=1.),
    Atom('O', [.000000, .000000, -.622978], magmom=1.),
    ])

# Hydrogen peroxide (HO-OH), C2 symm.
# MP2 energy = -151.1349184 Hartree
# Charge = 0, multiplicity = 1
H2O2 = Atoms([
    Atom('O', [.000000, .734058, -.052750]),
    Atom('O', [.000000, -.734058, -.052750]),
    Atom('H', [.839547, .880752, .422001]),
    Atom('H', [-.839547, -.880752, .422001]),
    ])

# F2 molecule, D*h symm.
# MP2 energy = -199.0388236 Hartree
# Charge = 0, multiplicity = 1
F2 = Atoms([
    Atom('F', [.000000, .000000, .710304]),
    Atom('F', [.000000, .000000, -.710304]),
    ])

# Carbon dioxide (CO2), D*h symm.
# MP2 energy = -188.1183633 Hartree
# Charge = 0, multiplicity = 1
CO2 = Atoms([
    Atom('C', [.000000, .000000, .000000]),
    Atom('O', [.000000, .000000, 1.178658]),
    Atom('O', [.000000, .000000, -1.178658]),
    ])

# Disodium (Na2), D*h symm.
# MP2 energy = -323.7039996 Hartree
# Charge = 0, multiplicity = 1
Na2 = Atoms([
    Atom('Na', [.000000, .000000, 1.576262]),
    Atom('Na', [.000000, .000000, -1.576262]),
    ])

# Si2 molecule, D*h symm, Triplet (3-Sigma-G-).
# MP2 energy = -577.8606556 Hartree
# Charge = 0, multiplicity = 3
Si2 = Atoms([
    Atom('Si', [.000000, .000000, 1.130054], magmom=1.),
    Atom('Si', [.000000, .000000, -1.130054], magmom=1.),
    ])

# P2 molecule, D*h symm.
# MP2 energy = -681.6646966 Hartree
# Charge = 0, multiplicity = 1
P2 = Atoms([
    Atom('P', [.000000, .000000, .966144]),
    Atom('P', [.000000, .000000, -.966144]),
    ])

# S2 molecule, D*h symm, triplet.
# MP2 energy = -795.2628131 Hartree
# Charge = 0, multiplicity = 3
S2 = Atoms([
    Atom('S', [.000000, .000000, .960113], magmom=1.),
    Atom('S', [.000000, .000000, -.960113], magmom=1.),
    ])

# Cl2 molecule, D*h symm.
# MP2 energy = -919.191224 Hartree
# Charge = 0, multiplicity = 1
Cl2 = Atoms([
    Atom('Cl', [.000000, .000000, 1.007541]),
    Atom('Cl', [.000000, .000000, -1.007541]),
    ])

# Sodium Chloride (NaCl), C*v symm.
# MP2 energy = -621.5463469 Hartree
# Charge = 0, multiplicity = 1
NaCl = Atoms([
    Atom('Na', [.000000, .000000, -1.451660]),
    Atom('Cl', [.000000, .000000, .939310]),
    ])

# Silicon monoxide (SiO), C*v symm.
# MP2 energy = -364.0594076 Hartree
# Charge = 0, multiplicity = 1
SiO = Atoms([
    Atom('Si', [.000000, .000000, .560846]),
    Atom('O', [.000000, .000000, -.981480]),
    ])

# Carbon monosulfide (CS), C*v symm.
# MP2 energy = -435.5576809 Hartree
# Charge = 0, multiplicity = 1
CS = Atoms([
    Atom('C', [.000000, .000000, -1.123382]),
    Atom('S', [.000000, .000000, .421268]),
    ])

# Sulfur monoxide (SO), C*v symm, triplet.
# MP2 energy = -472.6266876 Hartree
# Charge = 0, multiplicity = 3
SO = Atoms([
    Atom('O', [.000000, .000000, -1.015992], magmom=1.),
    Atom('S', [.000000, .000000, .507996], magmom=1.),
    ])

# ClO radical, C*v symm, 2-PI.
# MP2 energy = -534.5186484 Hartree
# Charge = 0, multiplicity = 2
ClO = Atoms([
    Atom('Cl', [.000000, .000000, .514172], magmom=1.),
    Atom('O', [.000000, .000000, -1.092615], magmom=0.),
    ])

# ClF molecule, C*v symm, 1-SG.
# MP2 energy = -559.1392996 Hartree
# Charge = 0, multiplicity = 1
ClF = Atoms([
    Atom('F', [.000000, .000000, -1.084794]),
    Atom('Cl', [.000000, .000000, .574302]),
    ])

# Disilane (H3Si-SiH3), D3d symm.
# MP2 energy = -581.4851067 Hartree
# Charge = 0, multiplicity = 1
Si2H6 = Atoms([
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
CH3Cl = Atoms([
    Atom('C', [.000000, .000000, -1.121389]),
    Atom('Cl', [.000000, .000000, .655951]),
    Atom('H', [.000000, 1.029318, -1.474280]),
    Atom('H', [.891415, -.514659, -1.474280]),
    Atom('H', [-.891415, -.514659, -1.474280]),
    ])

# Methanethiol (H3C-SH), Staggered, Cs symm.
# MP2 energy = -437.9678831 Hartree
# Charge = 0, multiplicity = 1
CH3SH = Atoms([
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
HOCl = Atoms([
    Atom('O', [.036702, 1.113517, .000000]),
    Atom('H', [-.917548, 1.328879, .000000]),
    Atom('Cl', [.036702, -.602177, .000000]),
    ])

# Sulfur dioxide (SO2), C2v symm.
# MP2 energy = -547.700099 Hartree
# Charge = 0, multiplicity = 1
SO2 = Atoms([
    Atom('S', [.000000, .000000, .370268]),
    Atom('O', [.000000, 1.277617, -.370268]),
    Atom('O', [.000000, -1.277617, -.370268]),
    ])

## Start of G2-2 test set

# BF3, Planar D3h symm.
# MP2 energy = -323.7915374. Hartree
# Charge = 0, multiplicity = 1
BF3 = Atoms([
    Atom('B', [.000000, .000000, .000000]),
    Atom('F', [.000000, 1.321760, .000000]),
    Atom('F', [1.144678, -.660880, .000000]),
    Atom('F', [-1.144678, -.660880, .000000]),
    ])

# BCl3, Planar D3h symm.
# MP2 energy = -1403.7595806 Hartree
# Charge = 0, multiplicity = 1
BCl3 = Atoms([
    Atom('B', [.000000, .000000, .000000]),
    Atom('Cl', [.000000, 1.735352, .000000]),
    Atom('Cl', [1.502859, -.867676, .000000]),
    Atom('Cl', [-1.502859, -.867676, .000000]),
    ])

# AlF3, Planar D3h symm.
# MP2 energy = -541.0397296 Hartree
# Charge = 0, multiplicity = 1
AlF3 = Atoms([
    Atom('Al', [.000000, .000000, .000000]),
    Atom('F', [.000000, 1.644720, .000000]),
    Atom('F', [1.424369, -.822360, .000000]),
    Atom('F', [-1.424369, -.822360, .000000]),
    ])

# AlCl3, Planar D3h symm.
# MP2 energy = -1621.0484142 Hartree
# Charge = 0, multiplicity = 1
AlCl3 = Atoms([
    Atom('Al', [.000000, .000000, .000000]),
    Atom('Cl', [.000000, 2.069041, .000000]),
    Atom('Cl', [1.791842, -1.034520, .000000]),
    Atom('Cl', [-1.791842, -1.034520, .000000]),
    ])

# CF4, Td symm.
# MP2 energy = -436.4622308 Hartree
# Charge = 0, multiplicity = 1
CF4 = Atoms([
    Atom('C', [.000000, .000000, .000000]),
    Atom('F', [.767436, .767436, .767436]),
    Atom('F', [-.767436, -.767436, .767436]),
    Atom('F', [-.767436, .767436, -.767436]),
    Atom('F', [.767436, -.767436, -.767436]),
    ])

# CCl4, Td symm.
# MP2 energy = -1876.4528012 Hartree
# Charge = 0, multiplicity = 1
CCl4 = Atoms([
    Atom('C', [.000000, .000000, .000000]),
    Atom('Cl', [1.021340, 1.021340, 1.021340]),
    Atom('Cl', [-1.021340, -1.021340, 1.021340]),
    Atom('Cl', [-1.021340, 1.021340, -1.021340]),
    Atom('Cl', [1.021340, -1.021340, -1.021340]),
    ])

# O=C=S, Linear, C*v symm.
# MP2 energy = -510.704382 Hartree
# Charge = 0, multiplicity = 1
OCS = Atoms([
    Atom('O', [.000000, .000000, -1.699243]),
    Atom('C', [.000000, .000000, -.520492]),
    Atom('S', [.000000, .000000, 1.044806]),
    ])

# CS2, Linear, D*h symm.
# MP2 energy = -833.2916974 Hartree
# Charge = 0, multiplicity = 1
CS2 = Atoms([
    Atom('S', [.000000, .000000, 1.561117]),
    Atom('C', [.000000, .000000, .000000]),
    Atom('S', [.000000, .000000, -1.561117]),
    ])

# COF2, C2v symm.
# MP2 energy = -312.2651646 Hartree
# Charge = 0, multiplicity = 1
COF2 = Atoms([
    Atom('O', [.000000, .000000, 1.330715]),
    Atom('C', [.000000, .000000, .144358]),
    Atom('F', [.000000, 1.069490, -.639548]),
    Atom('F', [.000000, -1.069490, -.639548]),
    ])

# SiF4, Td symm.
# MP2 energy = -687.7406597 Hartree
# Charge = 0, multiplicity = 1
SiF4 = Atoms([
    Atom('Si', [.000000, .000000, .000000]),
    Atom('F', [.912806, .912806, .912806]),
    Atom('F', [-.912806, -.912806, .912806]),
    Atom('F', [-.912806, .912806, -.912806]),
    Atom('F', [.912806, -.912806, -.912806]),
    ])

# SiCl4, Td symm.
# MP2 energy = -2127.6916411 Hartree
# Charge = 0, multiplicity = 1
SiCl4 = Atoms([
    Atom('Si', [.000000, .000000, .000000]),
    Atom('Cl', [1.169349, 1.169349, 1.169349]),
    Atom('Cl', [-1.169349, -1.169349, 1.169349]),
    Atom('Cl', [1.169349, -1.169349, -1.169349]),
    Atom('Cl', [-1.169349, 1.169349, -1.169349]),
    ])

# N2O, Cs symm.
# MP2 energy = -184.2136838 Hartree
# Charge = 0, multiplicity = 1
N2O = Atoms([
    Atom('N', [.000000, .000000, -1.231969]),
    Atom('N', [.000000, .000000, -.060851]),
    Atom('O', [.000000, .000000, 1.131218]),
    ])

# ClNO, Cs symm.
# MP2 energy = -589.1833856 Hartree
# Charge = 0, multiplicity = 1
ClNO = Atoms([
    Atom('Cl', [-.537724, -.961291, .000000]),
    Atom('N', [.000000, .997037, .000000]),
    Atom('O', [1.142664, 1.170335, .000000]),
    ])

# NF3, C3v symm.
# MP2 energy = -353.2366115 Hartree
# Charge = 0, multiplicity = 1
NF3 = Atoms([
    Atom('N', [.000000, .000000, .489672]),
    Atom('F', [.000000, 1.238218, -.126952]),
    Atom('F', [1.072328, -.619109, -.126952]),
    Atom('F', [-1.072328, -.619109, -.126952]),
    ])

# PF3, C3v symm.
# MP2 energy = -639.7725739 Hartree
# Charge = 0, multiplicity = 1
PF3 = Atoms([
    Atom('P', [.000000, .000000, .506767]),
    Atom('F', [.000000, 1.383861, -.281537]),
    Atom('F', [1.198459, -.691931, -.281537]),
    Atom('F', [-1.198459, -.691931, -.281537]),
    ])

# O3 (Ozone), C2v symm.
# MP2 energy = -224.8767539 Hartree
# Charge = 0, multiplicity = 1
O3 = Atoms([
    Atom('O', [.000000, 1.103810, -.228542]),
    Atom('O', [.000000, .000000, .457084]),
    Atom('O', [.000000, -1.103810, -.228542]),
    ])

# F2O, C2v symm.
# MP2 energy = -273.9997434 Hartree
# Charge = 0, multiplicity = 1
F2O = Atoms([
    Atom('F', [.000000, 1.110576, -.273729]),
    Atom('O', [.000000, .000000, .615890]),
    Atom('F', [.000000, -1.110576, -.273729]),
    ])

# ClF3, C2v symm.
# MP2 energy = .2017685 Hartree
# Charge = 0, multiplicity = 1
ClF3 = Atoms([
    Atom('Cl', [.000000, .000000, .376796]),
    Atom('F', [.000000, .000000, -1.258346]),
    Atom('F', [.000000, 1.714544, .273310]),
    Atom('F', [.000000, -1.714544, .273310]),
    ])

# C2F4 (F2C=CF2), D2H symm.
# MP2 energy = -474.3606919 Hartree
# Charge = 0, multiplicity = 1
C2F4 = Atoms([
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
C2Cl4 = Atoms([
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
CF3CN = Atoms([
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
C3H4_C3v = Atoms([
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
C3H4_D2d = Atoms([
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
C3H4_C2v = Atoms([
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
C3H6_Cs = Atoms([
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
C3H6_D3h = Atoms([
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
C3H8 = Atoms([
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
C4H6x = Atoms([
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
C4H6xx = Atoms([
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
C4H6xxx = Atoms([
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
C4H6xxxx = Atoms([
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
C4H6xxxxx = Atoms([
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
C4H8x = Atoms([
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
C4H8xx = Atoms([
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
C4H10x = Atoms([
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
C4H10xx = Atoms([
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
C5H8 = Atoms([
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
C6H6 = Atoms([
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
H2CF2 = Atoms([
    Atom('C', [.000000, .000000, .502903]),
    Atom('F', [.000000, 1.109716, -.290601]),
    Atom('F', [.000000, -1.109716, -.290601]),
    Atom('H', [-.908369, .000000, 1.106699]),
    Atom('H', [.908369, .000000, 1.106699]),
    ])

# Trifluoromethane (HCF3), C3v symm.
# MP2 energy = -337.4189848 Hartree
# Charge = 0, multiplicity = 1
HCF3 = Atoms([
    Atom('C', [.000000, .000000, .341023]),
    Atom('H', [.000000, .000000, 1.429485]),
    Atom('F', [.000000, 1.258200, -.128727]),
    Atom('F', [1.089633, -.629100, -.128727]),
    Atom('F', [-1.089633, -.629100, -.128727]),
    ])

# Dichloromethane (H2CCl2), C2v symm.
# MP2 energy = -958.4007187 Hartree
# Charge = 0, multiplicity = 1
H2CCl2 = Atoms([
    Atom('C', [.000000, .000000, .759945]),
    Atom('Cl', [.000000, 1.474200, -.215115]),
    Atom('Cl', [.000000, -1.474200, -.215115]),
    Atom('H', [-.894585, .000000, 1.377127]),
    Atom('H', [.894585, .000000, 1.377127]),
    ])

# Chloroform (HCCl3), C3v symm.
# MP2 energy = -1417.4294497 Hartree
# Charge = 0, multiplicity = 1
HCCl3 = Atoms([
    Atom('C', [.000000, .000000, .451679]),
    Atom('H', [.000000, .000000, 1.537586]),
    Atom('Cl', [.000000, 1.681723, -.083287]),
    Atom('Cl', [1.456415, -.840862, -.083287]),
    Atom('Cl', [-1.456415, -.840862, -.083287]),
    ])

# Methylamine (H3C-NH2), Cs symm.
# MP2 energy = -95.5144387 Hartree
# Charge = 0, multiplicity = 1
H3CNH2 = Atoms([
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
CH3CN = Atoms([
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
CH3NO2 = Atoms([
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
CH3ONO = Atoms([
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
CH3SiH3 = Atoms([
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
HCOOH = Atoms([
    Atom('O', [-1.040945, -.436432, .000000]),
    Atom('C', [.000000, .423949, .000000]),
    Atom('O', [1.169372, .103741, .000000]),
    Atom('H', [-.649570, -1.335134, .000000]),
    Atom('H', [-.377847, 1.452967, .000000]),
    ])

# Methyl formate (HCOOCH3), Cs symm.
# MP2 energy = -228.4116599 Hartree
# Charge = 0, multiplicity = 1
HCOOCH3 = Atoms([
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
CH3CONH2 = Atoms([
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
CH2NHCH2 = Atoms([
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
NCCN = Atoms([
    Atom('N', [.000000, .000000, 1.875875]),
    Atom('C', [.000000, .000000, .690573]),
    Atom('C', [.000000, .000000, -.690573]),
    Atom('N', [.000000, .000000, -1.875875]),
    ])

# Dimethylamine, (CH3)2NH, Cs symm.
# MP2 energy = -134.6781011 Hartree
# Charge = 0, multiplicity = 1
C2H6NH = Atoms([
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
CH3CH2NH2 = Atoms([
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
H2CCO = Atoms([
    Atom('C', [.000000, .000000, -1.219340]),
    Atom('C', [.000000, .000000, .098920]),
    Atom('H', [.000000, .938847, -1.753224]),
    Atom('H', [.000000, -.938847, -1.753224]),
    Atom('O', [.000000, .000000, 1.278620]),
    ])

# Oxirane (cyclic CH2-O-CH2 ring), C2v symm.
# MP2 energy = -153.3156907 Hartree
# Charge = 0, multiplicity = 1
CH2OCH2 = Atoms([
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
CH3CHO = Atoms([
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
OCHCHO = Atoms([
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
CH3CH2OH = Atoms([
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
CH3OCH3 = Atoms([
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
CH2SCH2 = Atoms([
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
C2H6SO = Atoms([
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
CH3CH2SH = Atoms([
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
CH3SCH3 = Atoms([
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
H2CCHF = Atoms([
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
CH3CH2Cl = Atoms([
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
H2CCHCl = Atoms([
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
H2CCHCN = Atoms([
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
CH3COCH3 = Atoms([
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
CH3COOH = Atoms([
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
CH3COF = Atoms([
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
CH3COCl = Atoms([
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
C3H7Cl = Atoms([
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
C2H6CHOH = Atoms([
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
CH3CH2OCH3 = Atoms([
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
C3H9N = Atoms([
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
C4H4O = Atoms([
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
C4H4S = Atoms([
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
C4H4NH = Atoms([
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
C5H5N = Atoms([
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
H2 = Atoms([
    Atom('H', [.000000, .000000, .368583]),
    Atom('H', [.000000, .000000, -.368583]),
    ])

# SH radical, C*v symm.
# MP2 energy = -398.1720853 Hartree
# Charge = 0, multiplicity = 2
SH = Atoms([
    Atom('S', [.000000, .000000, .079083], magmom=1.),
    Atom('H', [.000000, .000000, -1.265330], magmom=.0),
    ])

# CCH radical, C*v symm.
# MP2 energy = -76.3534702 Hartree
# Charge = 0, multiplicity = 2
CCH = Atoms([
    Atom('C', [.000000, .000000, -.462628], magmom=.0),
    Atom('C', [.000000, .000000, .717162], magmom=1.),
    Atom('H', [.000000, .000000, -1.527198], magmom=.0),
    ])

# C2H3 radical, Cs symm, 2-A'.
# MP2 energy = -77.613258 Hartree
# Charge = 0, multiplicity = 2
C2H3 = Atoms([
    Atom('C', [.049798, -.576272, .000000], magmom=.0),
    Atom('C', [.049798, .710988, .000000], magmom=1.),
    Atom('H', [-.876750, -1.151844, .000000], magmom=.0),
    Atom('H', [.969183, -1.154639, .000000], magmom=.0),
    Atom('H', [-.690013, 1.498185, .000000], magmom=.0),
    ])

# CH3CO radical, HCCO cis, Cs symm, 2-A'.
# MP2 energy = -152.7226518 Hartree
# Charge = 0, multiplicity = 2
CH3CO = Atoms([
    Atom('C', [-.978291, -.647814, .000000], magmom=.1),
    Atom('C', [.000000, .506283, .000000], magmom=.6),
    Atom('H', [-.455551, -1.607837, .000000], magmom=.0),
    Atom('H', [-1.617626, -.563271, .881061], magmom=.0),
    Atom('H', [-1.617626, -.563271, -.881061], magmom=.0),
    Atom('O', [1.195069, .447945, .000000], magmom=.3),
    ])

# H2COH radical, C1 symm.
# MP2 energy = -114.7033977 Hartree
# Charge = 0, multiplicity = 2
H2COH = Atoms([
    Atom('C', [.687448, .029626, -.082014], magmom=.7),
    Atom('O', [-.672094, -.125648, .030405], magmom=.3),
    Atom('H', [-1.091850, .740282, -.095167], magmom=.0),
    Atom('H', [1.122783, .975263, .225993], magmom=.0),
    Atom('H', [1.221131, -.888116, .118015], magmom=.0),
    ])

# CH3O radical, Cs symm, 2-A'.
# MP2 energy = -114.6930929 Hartree
# Charge = 0, multiplicity = 2
CH3O = Atoms([
    Atom('C', [-.008618, -.586475, .000000], magmom=.0),
    Atom('O', [-.008618, .799541, .000000], magmom=1.),
    Atom('H', [1.055363, -.868756, .000000], magmom=.0),
    Atom('H', [-.467358, -1.004363, .903279], magmom=.0),
    Atom('H', [-.467358, -1.004363, -.903279], magmom=.0),
    ])

# CH3CH2O radical, Cs symm, 2-A''.
# MP2 energy = -153.8670598 Hartree
# Charge = 0, multiplicity = 2
CH3CH2O = Atoms([
    Atom('C', [1.004757, -.568263, .000000], magmom=.0),
    Atom('C', [.000000, .588691, .000000], magmom=.0),
    Atom('O', [-1.260062, .000729, .000000], magmom=1.),
    Atom('H', [.146956, 1.204681, .896529], magmom=.0),
    Atom('H', [.146956, 1.204681, -.896529], magmom=.0),
    Atom('H', [2.019363, -.164100, .000000], magmom=.0),
    Atom('H', [.869340, -1.186832, .888071], magmom=.0),
    Atom('H', [.869340, -1.186832, -.888071], magmom=.0),
    ])

# CH3S radical, Cs symm, 2-A'.
# MP2 energy = -437.3459808 Hartree
# Charge = 0, multiplicity = 2
CH3S = Atoms([
    Atom('C', [-.003856, 1.106222, .000000], magmom=.0),
    Atom('S', [-.003856, -.692579, .000000], magmom=1.),
    Atom('H', [1.043269, 1.427057, .000000], magmom=.0),
    Atom('H', [-.479217, 1.508437, .895197], magmom=.0),
    Atom('H', [-.479217, 1.508437, -.895197], magmom=.0),
    ])

# C2H5 radical, Staggered, Cs symm, 2-A'.
# MP2 energy = -78.8446639 Hartree
# Charge = 0, multiplicity = 2
C2H5 = Atoms([
    Atom('C', [-.014359, -.694617, .000000], magmom=.0),
    Atom('C', [-.014359, .794473, .000000], magmom=1.),
    Atom('H', [1.006101, -1.104042, .000000], magmom=.0),
    Atom('H', [-.517037, -1.093613, .884839], magmom=.0),
    Atom('H', [-.517037, -1.093613, -.884839], magmom=.0),
    Atom('H', [.100137, 1.346065, .923705], magmom=.0),
    Atom('H', [.100137, 1.346065, -.923705], magmom=.0),
    ])

# (CH3)2CH radical, Cs symm, 2-A'.
# MP2 energy = -118.0192311 Hartree
# Charge = 0, multiplicity = 2
C3H7 = Atoms([
    Atom('C', [.014223, .543850, .000000], magmom=1.),
    Atom('C', [.014223, -.199742, 1.291572], magmom=.0),
    Atom('C', [.014223, -.199742, -1.291572], magmom=.0),
    Atom('H', [-.322890, 1.575329, .000000], magmom=.0),
    Atom('H', [.221417, .459174, 2.138477], magmom=.0),
    Atom('H', [.221417, .459174, -2.138477], magmom=.0),
    Atom('H', [-.955157, -.684629, 1.484633], magmom=.0),
    Atom('H', [.767181, -.995308, 1.286239], magmom=.0),
    Atom('H', [.767181, -.995308, -1.286239], magmom=.0),
    Atom('H', [-.955157, -.684629, -1.484633], magmom=.0),
    ])

# t-Butyl radical, (CH3)3C, C3v symm.
# MP2 energy = -157.1957937 Hartree
# Charge = 0, multiplicity = 2
C3H9C = Atoms([
    Atom('C', [.000000, .000000, .191929], magmom=1.),
    Atom('C', [.000000, 1.478187, -.020866], magmom=.0),
    Atom('C', [1.280147, -.739093, -.020866], magmom=.0),
    Atom('C', [-1.280147, -.739093, -.020866], magmom=.0),
    Atom('H', [.000000, 1.731496, -1.093792], magmom=.0),
    Atom('H', [-.887043, 1.945769, .417565], magmom=.0),
    Atom('H', [.887043, 1.945769, .417565], magmom=.0),
    Atom('H', [1.499520, -.865748, -1.093792], magmom=.0),
    Atom('H', [2.128607, -.204683, .417565], magmom=.0),
    Atom('H', [1.241564, -1.741086, .417565], magmom=.0),
    Atom('H', [-1.499520, -.865748, -1.093792], magmom=.0),
    Atom('H', [-1.241564, -1.741086, .417565], magmom=.0),
    Atom('H', [-2.128607, -.204683, .417565], magmom=.0),
    ])

# NO2 radical, C2v symm, 2-A1.
# MP2 energy = -204.5685941 Hartree
# Charge = 0, multiplicity = 2
NO2 = Atoms([
    Atom('N', [.000000, .000000, .332273], magmom=1.),
    Atom('O', [.000000, 1.118122, -.145370], magmom=.0),
    Atom('O', [.000000, -1.118122, -.145370], magmom=.0),
    ])

order = ['Be2','LiH','BeH','CH','CH2_s3B1d','CH2_s1A1d','CH3','CH4','NH','NH2','NH3','OH','H2O','HF','SiH2_s1A1d','SiH2_s3B1d','SiH3','SiH4','PH2','PH3','SH2','HCl','Li2','LiF','C2H2','C2H4','C2H6','CN','HCN','CO','HCO','H2CO','CH3OH','N2','N2H4','NO','O2','H2O2','F2','CO2','Na2','Si2','P2','S2','Cl2','NaCl','SiO','CS','SO','ClO','ClF','Si2H6','CH3Cl','CH3SH','HOCl','SO2','BF3','BCl3','AlF3','AlCl3','CF4','CCl4','OCS','CS2','COF2','SiF4','SiCl4','N2O','ClNO','NF3','PF3','O3','F2O','ClF3','C2F4','C2Cl4','CF3CN','C3H4_C3v','C3H4_D2d','C3H4_C2v','C3H6_Cs','C3H6_D3h','C3H8','C4H6x','C4H6xx','C4H6xxx','C4H6xxxx','C4H6xxxxx','C4H8x','C4H8xx','C4H10x','C4H10xx','C5H8','C6H6','H2CF2','HCF3','H2CCl2','HCCl3','H3CNH2','CH3CN','CH3NO2','CH3ONO','CH3SiH3','HCOOH','HCOOCH3','CH3CONH2','CH2NHCH2','NCCN','C2H6NH','CH3CH2NH2','H2CCO','CH2OCH2','CH3CHO','OCHCHO','CH3CH2OH','CH3OCH3','CH2SCH2','C2H6SO','CH3CH2SH','CH3SCH3','H2CCHF','CH3CH2Cl','H2CCHCl','H2CCHCN','CH3COCH3','CH3COOH','CH3COF','CH3COCl','C3H7Cl','C2H6CHOH','CH3CH2OCH3','C3H9N','C4H4O','C4H4S','C4H4NH','C5H5N','H2','SH','CCH','C2H3','CH3CO','H2COH','CH3O','CH3CH2O','CH3S','C2H5','C3H7','C3H9C','NO2','Butadiene_1_2','Isoprene','Cyclopentane','n_Pentane','Neopentane','Cyclohexadiene_1_3','Cyclohexadiene_1_4','Cyclohexane','n_Hexane','Methyl_pentane_3','Toluene','n_Heptane','Cyclooctatetraene','n_Octane','Naphthalene','Azulene','Methyl_acetate','t_Butanol','Aniline','Phenol','Divinyl_ether','Tetrahydrofuran','Cyclopentanone','Benzoquinone_1_4','Pyrimidine','Dimethyl_sulfone','Chlorobenzene','Succinonitrile','Pyrazine','Acetyl_acetylene','Crotonaldehyde','Acetic_anhydride','Dihydrothiophene_2_5','Methyl_propanenitrile_2','Methyl_ethyl_ketone','Isobutyraldehyde','dioxane_1_4','Tetrahydrothiophene','t_Butyl_chloride','n_Butyl_chloride','Tetrahydropyrrole','Nitrobutane_2','Diethyl_ether','Dimethoxy_ethane_1_1','t_Butanethiol','Diethyl_disulfide','t_Butylamine','Tetramethylsilane','Methyl_thiophene','N_methyl_pyrrole','Tetrahydropyran','Diethyl_ketone','Isopropyl_acetate','Tetrahydrothiopyran','Piperidine','t_Butyl_methyl_ether','Difluorobenzene_1_3','Difluorobenzene_1_4','Fluorobenzene','Diisopropyl_ether','PF5','SF6','P4_Td','SO3_D3h','SCl2','POCl3','PCl5','SO2Cl2','PCl3','S2Cl2','SiCl2_1A1','CF3Cl','C2F6','CF3_radical','Phenyl_radical']

elements = ['H','Li','Be','B','C','N','O','F','Na','Mg','Al','Si','P','S','Cl']
extra = order[:1] # The extra systems
g1 = order[1:56]  # The g1 molecules
g2 = order[1:149] # The g2 molecules
g3 = order[1:224] # The g3 molecules
del order

def get_ae(name):
    """Determine extrapolated experimental atomization energy.

    The atomization energy is extrapolated from experimental heats of
    formation at room temperature, using calculated zero-point energies
    and thermal corrections.
    """
    assert name in extra or name in g2
    descrip, longname, e, z, dh = molecules[name]
    ae = -e + z + dh
    for a in eval(name).get_chemical_symbols():
        mag, h, dh = atoms[a]
        ae += h - dh
    return ae

def get_g2(name, cell=(1.0, 1.0, 1.0)):
    if name in atoms:
        loa =  Atoms([Atom(name, magmom=atoms[name][0])], cell=cell, pbc=False)
    elif name in extra or name in g2:
        loa = eval(name).copy()
        loa.set_cell(cell)
        loa.set_pbc(False)
    else:
        raise NotImplementedError('System %s not in database.' % name)

    loa.center()
    return loa

if __name__ == '__main__':
    from gpaw.testing.atomization_data import atomization_vasp
    for name in g1:
        print '%-9s %6.1f %6.1f' % (name, get_ae(name),
                                    atomization_vasp[name][0])
