#include <stdlib.h>

#include "xc.h"

extern xc_func_info_type *lda_known_funct[], *gga_known_funct[], *mgga_known_funct[], *lca_known_funct[];

int xc_family_from_id(int id)
{
  int i;

  /* first let us check if it is an LDA */
  for(i=0; lda_known_funct[i]!=NULL; i++){
    if(lda_known_funct[i]->number == id) return XC_FAMILY_LDA;
  }

  /* or is it a GGA? */
  for(i=0; gga_known_funct[i]!=NULL; i++){
    if(gga_known_funct[i]->number == id) return XC_FAMILY_GGA;
  }

  /* or is it a MGGA? */
  for(i=0; mgga_known_funct[i]!=NULL; i++){
    if(mgga_known_funct[i]->number == id) return XC_FAMILY_MGGA;
  }

  /* or is it a LCA? */
  for(i=0; lca_known_funct[i]!=NULL; i++){
    if(lca_known_funct[i]->number == id) return XC_FAMILY_LCA;
  }

  return XC_FAMILY_UNKNOWN;
}
