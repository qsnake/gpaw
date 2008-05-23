#include "util.h"

extern XC(func_info_type) XC(func_info_mgga_x_tpss);
extern XC(func_info_type) XC(func_info_mgga_c_tpss);


const XC(func_info_type) *XC(mgga_known_funct)[] = {
  &XC(func_info_mgga_x_tpss),
  &XC(func_info_mgga_c_tpss),
  NULL
};
