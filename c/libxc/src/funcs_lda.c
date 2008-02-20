#include "util.h"

extern xc_func_info_type func_info_lda_x;
extern xc_func_info_type func_info_lda_c_wigner;
extern xc_func_info_type func_info_lda_c_rpa;
extern xc_func_info_type func_info_lda_c_hl;
extern xc_func_info_type func_info_lda_c_gl;
extern xc_func_info_type func_info_lda_c_xalpha;
extern xc_func_info_type func_info_lda_c_vwn;
extern xc_func_info_type func_info_lda_c_vwn_rpa;
extern xc_func_info_type func_info_lda_c_pz;
extern xc_func_info_type func_info_lda_c_pz_mod;
extern xc_func_info_type func_info_lda_c_ob_pz;
extern xc_func_info_type func_info_lda_c_pw;
extern xc_func_info_type func_info_lda_c_pw_mod;
extern xc_func_info_type func_info_lda_c_ob_pw;
extern xc_func_info_type func_info_lda_c_amgb;
extern xc_func_info_type func_info_lda_xc_teter93;


const xc_func_info_type *lda_known_funct[] = {
  &func_info_lda_x,
  &func_info_lda_c_wigner,
  &func_info_lda_c_rpa,
  &func_info_lda_c_hl,
  &func_info_lda_c_gl,
  &func_info_lda_c_xalpha,
  &func_info_lda_c_vwn,
  &func_info_lda_c_vwn_rpa,
  &func_info_lda_c_pz,
  &func_info_lda_c_pz_mod,
  &func_info_lda_c_ob_pz,
  &func_info_lda_c_pw,
  &func_info_lda_c_pw_mod,
  &func_info_lda_c_ob_pw,
  &func_info_lda_c_amgb,
  &func_info_lda_xc_teter93,
  NULL
};
