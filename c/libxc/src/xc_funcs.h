#define  XC_LDA_X               1  /* Exchange                                                   */
#define  XC_LDA_C_WIGNER        2  /* Wigner parametrization                                     */
#define  XC_LDA_C_RPA           3  /* Random Phase Approximation                                 */
#define  XC_LDA_C_HL            4  /* Hedin & Lundqvist                                          */
#define  XC_LDA_C_GL            5  /* Gunnarson & Lundqvist                                      */
#define  XC_LDA_C_XALPHA        6  /* Slater's Xalpha                                            */
#define  XC_LDA_C_VWN           7  /* Vosko, Wilk, & Nussair                                     */
#define  XC_LDA_C_VWN_RPA       8  /* Vosko, Wilk, & Nussair (RPA)                               */
#define  XC_LDA_C_PZ            9  /* Perdew & Zunger                                            */
#define  XC_LDA_C_PZ_MOD       10  /* Perdew & Zunger (Modified)                                 */
#define  XC_LDA_C_OB_PZ        11  /* Ortiz & Ballone (PZ)                                       */
#define  XC_LDA_C_PW           12  /* Perdew & Wang                                              */
#define  XC_LDA_C_PW_MOD       13  /* Perdew & Wang (Modified)                                   */
#define  XC_LDA_C_OB_PW        14  /* Ortiz & Ballone (PW)                                       */
#define  XC_LDA_C_AMGB         15  /* Attacalite et al                                           */
#define  XC_LDA_XC_TETER93     20  /* Teter 93 parametrization                                   */
#define  XC_GGA_X_PBE         101  /* Perdew, Burke & Ernzerhof exchange                         */
#define  XC_GGA_X_PBE_R       102  /* Perdew, Burke & Ernzerhof exchange (revised)               */
#define  XC_GGA_X_B86         103  /* Becke 86 Xalfa,beta,gamma                                  */
#define  XC_GGA_X_B86_R       104  /* Becke 86 Xalfa,beta,gamma (reoptimized)                    */
#define  XC_GGA_X_B86_MGC     105  /* Becke 86 Xalfa,beta,gamma (with mod. grad. correction)     */
#define  XC_GGA_X_B88         106  /* Becke 88                                                   */
#define  XC_GGA_X_G96         107  /* Gill 96                                                    */
#define  XC_GGA_X_PW86        108  /* Perdew & Wang 86                                           */
#define  XC_GGA_X_PW91        109  /* Perdew & Wang 91                                           */
#define  XC_GGA_X_OPTX        110  /* Handy & Cohen OPTX 01                                      */
#define  XC_GGA_X_DK87_R1     111  /* dePristo & Kress 87 (version R1)                           */
#define  XC_GGA_X_DK87_R2     112  /* dePristo & Kress 87 (version R2)                           */
#define  XC_GGA_X_LG93        113  /* Lacks & Gordon 93                                          */
#define  XC_GGA_X_FT97_A      114  /* Filatov & Thiel 97 (version A)                             */
#define  XC_GGA_X_FT97_B      115  /* Filatov & Thiel 97 (version B)                             */
#define  XC_GGA_X_PBE_SOL     116  /* Perdew, Burke & Ernzerhof exchange (solids)                */
#define  XC_GGA_X_RPBE        117  /* Hammer, Hansen & Norskov (PBE-like)                        */
#define  XC_GGA_X_WC          118  /* Wu & Cohen                                                 */
#define  XC_GGA_X_mPW91       119  /* Modified form of PW91 by Adamo & Barone                    */
#define  XC_GGA_X_AM05        120  /* Armiento & Mattsson 05 exchange                            */
#define  XC_GGA_X_PBEA        121  /* Madsen (PBE-like)                                          */
#define  XC_GGA_X_MPBE        122  /* Adamo & Barone modification to PBE                         */
#define  XC_GGA_X_XPBE        123  /* xPBE reparametrization by Xu & Goddard                     */
#define  XC_GGA_C_PBE         130  /* Perdew, Burke & Ernzerhof correlation                      */
#define  XC_GGA_C_LYP         131  /* Lee, Yang & Parr                                           */
#define  XC_GGA_C_P86         132  /* Perdew 86                                                  */
#define  XC_GGA_C_PBE_SOL     133  /* Perdew, Burke & Ernzerhof correlation SOL                  */
#define  XC_GGA_C_PW91        134  /* Perdew & Wang 91                                           */
#define  XC_GGA_C_AM05        135  /* Armiento & Mattsson 05 correlation                         */
#define  XC_GGA_C_XPBE        136  /* xPBE reparametrization by Xu & Goddard                     */
#define  XC_GGA_XC_LB         160  /* van Leeuwen & Baerends                                     */
#define  XC_GGA_XC_HCTH_93    161  /* HCTH functional fitted to  93 molecules                    */
#define  XC_GGA_XC_HCTH_120   162  /* HCTH functional fitted to 120 molecules                    */
#define  XC_GGA_XC_HCTH_147   163  /* HCTH functional fitted to 147 molecules                    */
#define  XC_GGA_XC_HCTH_407   164  /* HCTH functional fitted to 147 molecules                    */
#define  XC_GGA_XC_EDF1       165  /* Empirical functionals from Adamson, Gill, and Pople        */
#define  XC_GGA_XC_XLYP       166  /* XLYP functional                                            */
#define  XC_HYB_GGA_XC_B3PW91 401  /* The original hybrid proposed by Becke                      */
#define  XC_HYB_GGA_XC_B3LYP  402  /* The (in)famous B3LYP                                       */
#define  XC_HYB_GGA_XC_B3P86  403  /* Perdew 86 hybrid similar to B3PW91                         */
#define  XC_HYB_GGA_XC_O3LYP  404  /* hybrid using the optx functional                           */
#define  XC_HYB_GGA_XC_PBEH   406  /* aka PBE0 or PBE1PBE                                        */
#define  XC_HYB_GGA_XC_X3LYP  411  /* maybe the best hybrid                                      */
#define  XC_HYB_GGA_XC_B1WC   412  /* Becke 1-parameter mixture of WC and EXX                    */
#define  XC_MGGA_X_TPSS       201  /* Perdew, Tao, Staroverov & Scuseria exchange                */
#define  XC_MGGA_C_TPSS       202  /* Perdew, Tao, Staroverov & Scuseria correlation             */
#define  XC_LCA_OMC           301  /* Orestes, Marcasso & Capelle                                */
#define  XC_LCA_LCH           302  /* Lee, Colwell & Handy                                       */
