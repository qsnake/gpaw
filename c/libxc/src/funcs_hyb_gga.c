#include "util.h"

extern xc_func_info_type func_info_hyb_gga_xc_b3pw91;
extern xc_func_info_type func_info_hyb_gga_xc_b3lyp;
extern xc_func_info_type func_info_hyb_gga_xc_b3p86;
extern xc_func_info_type func_info_hyb_gga_xc_o3lyp;
extern xc_func_info_type func_info_hyb_gga_xc_pbeh;
extern xc_func_info_type func_info_hyb_gga_xc_x3lyp;


const xc_func_info_type *hyb_gga_known_funct[] = {
  &func_info_hyb_gga_xc_b3pw91,
  &func_info_hyb_gga_xc_b3lyp,
  &func_info_hyb_gga_xc_b3p86,
  &func_info_hyb_gga_xc_o3lyp,
  &func_info_hyb_gga_xc_pbeh,
  &func_info_hyb_gga_xc_x3lyp,
  NULL
};
