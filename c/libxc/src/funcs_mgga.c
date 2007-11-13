#include "util.h"

extern xc_func_info_type func_info_mgga_x_tpss;
extern xc_func_info_type func_info_mgga_c_tpss;


const xc_func_info_type *mgga_known_funct[] = {
  &func_info_mgga_x_tpss,
  &func_info_mgga_c_tpss,
  NULL
};
