#include "util.h"

extern XC(func_info_type) XC(func_info_lda_x);
extern XC(func_info_type) XC(func_info_lda_c_wigner);
extern XC(func_info_type) XC(func_info_lda_c_rpa);
extern XC(func_info_type) XC(func_info_lda_c_hl);
extern XC(func_info_type) XC(func_info_lda_c_gl);
extern XC(func_info_type) XC(func_info_lda_c_xalpha);
extern XC(func_info_type) XC(func_info_lda_c_vwn);
extern XC(func_info_type) XC(func_info_lda_c_vwn_rpa);
extern XC(func_info_type) XC(func_info_lda_c_pz);
extern XC(func_info_type) XC(func_info_lda_c_pz_mod);
extern XC(func_info_type) XC(func_info_lda_c_ob_pz);
extern XC(func_info_type) XC(func_info_lda_c_pw);
extern XC(func_info_type) XC(func_info_lda_c_pw_mod);
extern XC(func_info_type) XC(func_info_lda_c_ob_pw);
extern XC(func_info_type) XC(func_info_lda_c_amgb);
extern XC(func_info_type) XC(func_info_lda_xc_teter93);


const XC(func_info_type) *XC(lda_known_funct)[] = {
  &XC(func_info_lda_x),
  &XC(func_info_lda_c_wigner),
  &XC(func_info_lda_c_rpa),
  &XC(func_info_lda_c_hl),
  &XC(func_info_lda_c_gl),
  &XC(func_info_lda_c_xalpha),
  &XC(func_info_lda_c_vwn),
  &XC(func_info_lda_c_vwn_rpa),
  &XC(func_info_lda_c_pz),
  &XC(func_info_lda_c_pz_mod),
  &XC(func_info_lda_c_ob_pz),
  &XC(func_info_lda_c_pw),
  &XC(func_info_lda_c_pw_mod),
  &XC(func_info_lda_c_ob_pw),
  &XC(func_info_lda_c_amgb),
  &XC(func_info_lda_xc_teter93),
  NULL
};
