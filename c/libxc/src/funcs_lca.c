#include "util.h"

extern xc_func_info_type func_info_lca_omc;
extern xc_func_info_type func_info_lca_lch;


const xc_func_info_type *lca_known_funct[] = {
  &func_info_lca_omc,
  &func_info_lca_lch,
  NULL
};
