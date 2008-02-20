#include "util.h"

extern xc_func_info_type func_info_gga_x_pbe;
extern xc_func_info_type func_info_gga_x_pbe_r;
extern xc_func_info_type func_info_gga_x_b86;
extern xc_func_info_type func_info_gga_x_b86_r;
extern xc_func_info_type func_info_gga_x_b86_mgc;
extern xc_func_info_type func_info_gga_x_b88;
extern xc_func_info_type func_info_gga_x_g96;
extern xc_func_info_type func_info_gga_x_pw86;
extern xc_func_info_type func_info_gga_x_pw91;
extern xc_func_info_type func_info_gga_x_optx;
extern xc_func_info_type func_info_gga_x_dk87_r1;
extern xc_func_info_type func_info_gga_x_dk87_r2;
extern xc_func_info_type func_info_gga_x_lg93;
extern xc_func_info_type func_info_gga_x_ft97_a;
extern xc_func_info_type func_info_gga_x_ft97_b;
extern xc_func_info_type func_info_gga_x_pbe_sol;
extern xc_func_info_type func_info_gga_x_rpbe;
extern xc_func_info_type func_info_gga_x_wc;
extern xc_func_info_type func_info_gga_x_mpw91;
extern xc_func_info_type func_info_gga_c_pbe;
extern xc_func_info_type func_info_gga_c_lyp;
extern xc_func_info_type func_info_gga_c_p86;
extern xc_func_info_type func_info_gga_c_pbe_sol;
extern xc_func_info_type func_info_gga_c_pw91;
extern xc_func_info_type func_info_gga_xc_lb;
extern xc_func_info_type func_info_gga_xc_hcth_93;
extern xc_func_info_type func_info_gga_xc_hcth_120;
extern xc_func_info_type func_info_gga_xc_hcth_147;
extern xc_func_info_type func_info_gga_xc_hcth_407;
extern xc_func_info_type func_info_gga_xc_edf1;
extern xc_func_info_type func_info_gga_xc_xlyp;


const xc_func_info_type *gga_known_funct[] = {
  &func_info_gga_x_pbe,
  &func_info_gga_x_pbe_r,
  &func_info_gga_x_b86,
  &func_info_gga_x_b86_r,
  &func_info_gga_x_b86_mgc,
  &func_info_gga_x_b88,
  &func_info_gga_x_g96,
  &func_info_gga_x_pw86,
  &func_info_gga_x_pw91,
  &func_info_gga_x_optx,
  &func_info_gga_x_dk87_r1,
  &func_info_gga_x_dk87_r2,
  &func_info_gga_x_lg93,
  &func_info_gga_x_ft97_a,
  &func_info_gga_x_ft97_b,
  &func_info_gga_x_pbe_sol,
  &func_info_gga_x_rpbe,
  &func_info_gga_x_wc,
  &func_info_gga_x_mpw91,
  &func_info_gga_c_pbe,
  &func_info_gga_c_lyp,
  &func_info_gga_c_p86,
  &func_info_gga_c_pbe_sol,
  &func_info_gga_c_pw91,
  &func_info_gga_xc_lb,
  &func_info_gga_xc_hcth_93,
  &func_info_gga_xc_hcth_120,
  &func_info_gga_xc_hcth_147,
  &func_info_gga_xc_hcth_407,
  &func_info_gga_xc_edf1,
  &func_info_gga_xc_xlyp,
  NULL
};
