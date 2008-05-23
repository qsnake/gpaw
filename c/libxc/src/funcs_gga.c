#include "util.h"

extern XC(func_info_type) XC(func_info_gga_x_pbe);
extern XC(func_info_type) XC(func_info_gga_x_pbe_r);
extern XC(func_info_type) XC(func_info_gga_x_b86);
extern XC(func_info_type) XC(func_info_gga_x_b86_r);
extern XC(func_info_type) XC(func_info_gga_x_b86_mgc);
extern XC(func_info_type) XC(func_info_gga_x_b88);
extern XC(func_info_type) XC(func_info_gga_x_g96);
extern XC(func_info_type) XC(func_info_gga_x_pw86);
extern XC(func_info_type) XC(func_info_gga_x_pw91);
extern XC(func_info_type) XC(func_info_gga_x_optx);
extern XC(func_info_type) XC(func_info_gga_x_dk87_r1);
extern XC(func_info_type) XC(func_info_gga_x_dk87_r2);
extern XC(func_info_type) XC(func_info_gga_x_lg93);
extern XC(func_info_type) XC(func_info_gga_x_ft97_a);
extern XC(func_info_type) XC(func_info_gga_x_ft97_b);
extern XC(func_info_type) XC(func_info_gga_x_pbe_sol);
extern XC(func_info_type) XC(func_info_gga_x_rpbe);
extern XC(func_info_type) XC(func_info_gga_x_wc);
extern XC(func_info_type) XC(func_info_gga_x_mpw91);
extern XC(func_info_type) XC(func_info_gga_x_am05);
extern XC(func_info_type) XC(func_info_gga_x_pbea);
extern XC(func_info_type) XC(func_info_gga_x_mpbe);
extern XC(func_info_type) XC(func_info_gga_x_xpbe);
extern XC(func_info_type) XC(func_info_gga_c_pbe);
extern XC(func_info_type) XC(func_info_gga_c_lyp);
extern XC(func_info_type) XC(func_info_gga_c_p86);
extern XC(func_info_type) XC(func_info_gga_c_pbe_sol);
extern XC(func_info_type) XC(func_info_gga_c_pw91);
extern XC(func_info_type) XC(func_info_gga_c_am05);
extern XC(func_info_type) XC(func_info_gga_c_xpbe);
extern XC(func_info_type) XC(func_info_gga_xc_lb);
extern XC(func_info_type) XC(func_info_gga_xc_hcth_93);
extern XC(func_info_type) XC(func_info_gga_xc_hcth_120);
extern XC(func_info_type) XC(func_info_gga_xc_hcth_147);
extern XC(func_info_type) XC(func_info_gga_xc_hcth_407);
extern XC(func_info_type) XC(func_info_gga_xc_edf1);
extern XC(func_info_type) XC(func_info_gga_xc_xlyp);


const XC(func_info_type) *XC(gga_known_funct)[] = {
  &XC(func_info_gga_x_pbe),
  &XC(func_info_gga_x_pbe_r),
  &XC(func_info_gga_x_b86),
  &XC(func_info_gga_x_b86_r),
  &XC(func_info_gga_x_b86_mgc),
  &XC(func_info_gga_x_b88),
  &XC(func_info_gga_x_g96),
  &XC(func_info_gga_x_pw86),
  &XC(func_info_gga_x_pw91),
  &XC(func_info_gga_x_optx),
  &XC(func_info_gga_x_dk87_r1),
  &XC(func_info_gga_x_dk87_r2),
  &XC(func_info_gga_x_lg93),
  &XC(func_info_gga_x_ft97_a),
  &XC(func_info_gga_x_ft97_b),
  &XC(func_info_gga_x_pbe_sol),
  &XC(func_info_gga_x_rpbe),
  &XC(func_info_gga_x_wc),
  &XC(func_info_gga_x_mpw91),
  &XC(func_info_gga_x_am05),
  &XC(func_info_gga_x_pbea),
  &XC(func_info_gga_x_mpbe),
  &XC(func_info_gga_x_xpbe),
  &XC(func_info_gga_c_pbe),
  &XC(func_info_gga_c_lyp),
  &XC(func_info_gga_c_p86),
  &XC(func_info_gga_c_pbe_sol),
  &XC(func_info_gga_c_pw91),
  &XC(func_info_gga_c_am05),
  &XC(func_info_gga_c_xpbe),
  &XC(func_info_gga_xc_lb),
  &XC(func_info_gga_xc_hcth_93),
  &XC(func_info_gga_xc_hcth_120),
  &XC(func_info_gga_xc_hcth_147),
  &XC(func_info_gga_xc_hcth_407),
  &XC(func_info_gga_xc_edf1),
  &XC(func_info_gga_xc_xlyp),
  NULL
};
