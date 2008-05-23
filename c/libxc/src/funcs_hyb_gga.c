#include "util.h"

extern XC(func_info_type) XC(func_info_hyb_gga_xc_b3pw91);
extern XC(func_info_type) XC(func_info_hyb_gga_xc_b3lyp);
extern XC(func_info_type) XC(func_info_hyb_gga_xc_b3p86);
extern XC(func_info_type) XC(func_info_hyb_gga_xc_o3lyp);
extern XC(func_info_type) XC(func_info_hyb_gga_xc_pbeh);
extern XC(func_info_type) XC(func_info_hyb_gga_xc_x3lyp);
extern XC(func_info_type) XC(func_info_hyb_gga_xc_b1wc);


const XC(func_info_type) *XC(hyb_gga_known_funct)[] = {
  &XC(func_info_hyb_gga_xc_b3pw91),
  &XC(func_info_hyb_gga_xc_b3lyp),
  &XC(func_info_hyb_gga_xc_b3p86),
  &XC(func_info_hyb_gga_xc_o3lyp),
  &XC(func_info_hyb_gga_xc_pbeh),
  &XC(func_info_hyb_gga_xc_x3lyp),
  &XC(func_info_hyb_gga_xc_b1wc),
  NULL
};
