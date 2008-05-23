module libxc_funcs_m
  implicit none

  public

  integer, parameter :: XC_LDA_X             =   1  !  Exchange                     
  integer, parameter :: XC_LDA_C_WIGNER      =   2  !  Wigner parametrization       
  integer, parameter :: XC_LDA_C_RPA         =   3  !  Random Phase Approximation   
  integer, parameter :: XC_LDA_C_HL          =   4  !  Hedin & Lundqvist            
  integer, parameter :: XC_LDA_C_GL          =   5  !  Gunnarson & Lundqvist        
  integer, parameter :: XC_LDA_C_XALPHA      =   6  !  Slater's Xalpha              
  integer, parameter :: XC_LDA_C_VWN         =   7  !  Vosko, Wilk, & Nussair       
  integer, parameter :: XC_LDA_C_VWN_RPA     =   8  !  Vosko, Wilk, & Nussair (RPA) 
  integer, parameter :: XC_LDA_C_PZ          =   9  !  Perdew & Zunger              
  integer, parameter :: XC_LDA_C_PZ_MOD      =  10  !  Perdew & Zunger (Modified)   
  integer, parameter :: XC_LDA_C_OB_PZ       =  11  !  Ortiz & Ballone (PZ)         
  integer, parameter :: XC_LDA_C_PW          =  12  !  Perdew & Wang                
  integer, parameter :: XC_LDA_C_PW_MOD      =  13  !  Perdew & Wang (Modified)     
  integer, parameter :: XC_LDA_C_OB_PW       =  14  !  Ortiz & Ballone (PW)         
  integer, parameter :: XC_LDA_C_AMGB        =  15  !  Attacalite et al             
  integer, parameter :: XC_LDA_XC_TETER93    =  20  !  Teter 93 parametrization                
  integer, parameter :: XC_GGA_X_PBE         = 101  !  Perdew, Burke & Ernzerhof exchange             
  integer, parameter :: XC_GGA_X_PBE_R       = 102  !  Perdew, Burke & Ernzerhof exchange (revised)   
  integer, parameter :: XC_GGA_X_B86         = 103  !  Becke 86 Xalfa,beta,gamma                      
  integer, parameter :: XC_GGA_X_B86_R       = 104  !  Becke 86 Xalfa,beta,gamma (reoptimized)        
  integer, parameter :: XC_GGA_X_B86_MGC     = 105  !  Becke 86 Xalfa,beta,gamma (with mod. grad. correction) 
  integer, parameter :: XC_GGA_X_B88         = 106  !  Becke 88 
  integer, parameter :: XC_GGA_X_G96         = 107  !  Gill 96                                        
  integer, parameter :: XC_GGA_X_PW86        = 108  !  Perdew & Wang 86 
  integer, parameter :: XC_GGA_X_PW91        = 109  !  Perdew & Wang 91 
  integer, parameter :: XC_GGA_X_OPTX        = 110  !  Handy & Cohen OPTX 01                          
  integer, parameter :: XC_GGA_X_DK87_R1     = 111  !  dePristo & Kress 87 (version R1)               
  integer, parameter :: XC_GGA_X_DK87_R2     = 112  !  dePristo & Kress 87 (version R2)               
  integer, parameter :: XC_GGA_X_LG93        = 113  !  Lacks & Gordon 93 
  integer, parameter :: XC_GGA_X_FT97_A      = 114  !  Filatov & Thiel 97 (version A) 
  integer, parameter :: XC_GGA_X_FT97_B      = 115  !  Filatov & Thiel 97 (version B) 
  integer, parameter :: XC_GGA_X_PBE_SOL     = 116  !  Perdew, Burke & Ernzerhof exchange (solids)    
  integer, parameter :: XC_GGA_X_RPBE        = 117  !  Hammer, Hansen & Norskov (PBE-like) 
  integer, parameter :: XC_GGA_X_WC          = 118  !  Wu & Cohen 
  integer, parameter :: XC_GGA_X_mPW91       = 119  !  Modified form of PW91 by Adamo & Barone 
  integer, parameter :: XC_GGA_X_AM05        = 120  !  Armiento & Mattsson 05 exchange                
  integer, parameter :: XC_GGA_X_PBEA        = 121  !  Madsen (PBE-like) 
  integer, parameter :: XC_GGA_X_MPBE        = 122  !  Adamo & Barone modification to PBE             
  integer, parameter :: XC_GGA_X_XPBE        = 123  !  xPBE reparametrization by Xu & Goddard         
  integer, parameter :: XC_GGA_C_PBE         = 130  !  Perdew, Burke & Ernzerhof correlation          
  integer, parameter :: XC_GGA_C_LYP         = 131  !  Lee, Yang & Parr 
  integer, parameter :: XC_GGA_C_P86         = 132  !  Perdew 86 
  integer, parameter :: XC_GGA_C_PBE_SOL     = 133  !  Perdew, Burke & Ernzerhof correlation SOL      
  integer, parameter :: XC_GGA_C_PW91        = 134  !  Perdew & Wang 91 
  integer, parameter :: XC_GGA_C_AM05        = 135  !  Armiento & Mattsson 05 correlation             
  integer, parameter :: XC_GGA_C_XPBE        = 136  !  xPBE reparametrization by Xu & Goddard         
  integer, parameter :: XC_GGA_XC_LB         = 160  !  van Leeuwen & Baerends 
  integer, parameter :: XC_GGA_XC_HCTH_93    = 161  !  HCTH functional fitted to  93 molecules  
  integer, parameter :: XC_GGA_XC_HCTH_120   = 162  !  HCTH functional fitted to 120 molecules  
  integer, parameter :: XC_GGA_XC_HCTH_147   = 163  !  HCTH functional fitted to 147 molecules  
  integer, parameter :: XC_GGA_XC_HCTH_407   = 164  !  HCTH functional fitted to 147 molecules  
  integer, parameter :: XC_GGA_XC_EDF1       = 165  !  Empirical functionals from Adamson, Gill, and Pople 
  integer, parameter :: XC_GGA_XC_XLYP       = 166  !  XLYP functional 
  integer, parameter :: XC_HYB_GGA_XC_B3PW91 = 401  !  The original hybrid proposed by Becke 
  integer, parameter :: XC_HYB_GGA_XC_B3LYP  = 402  !  The (in)famous B3LYP 
  integer, parameter :: XC_HYB_GGA_XC_B3P86  = 403  !  Perdew 86 hybrid similar to B3PW91 
  integer, parameter :: XC_HYB_GGA_XC_O3LYP  = 404  !  hybrid using the optx functional 
  integer, parameter :: XC_HYB_GGA_XC_PBEH   = 406  !  aka PBE0 or PBE1PBE 
  integer, parameter :: XC_HYB_GGA_XC_X3LYP  = 411  !  maybe the best hybrid 
  integer, parameter :: XC_HYB_GGA_XC_B1WC   = 412  !  Becke 1-parameter mixture of WC and EXX 
  integer, parameter :: XC_MGGA_X_TPSS       = 201  !  Perdew, Tao, Staroverov & Scuseria exchange 
  integer, parameter :: XC_MGGA_C_TPSS       = 202  !  Perdew, Tao, Staroverov & Scuseria correlation 
  integer, parameter :: XC_LCA_OMC           = 301  !  Orestes, Marcasso & Capelle  
  integer, parameter :: XC_LCA_LCH           = 302  !  Lee, Colwell & Handy         

end module libxc_funcs_m
