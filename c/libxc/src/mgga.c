/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3 of the License, or
 (at your option) any later version.
  
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
  
 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <stdlib.h>
#include <assert.h>

#include "util.h"
#include "funcs_mgga.c"

/* initialization */
void XC(mgga_init)(XC(mgga_type) *p, int functional, int nspin)
{
  int i;

  assert(p != NULL);

  for(i=0; XC(mgga_known_funct)[i]!=NULL; i++){
    if(XC(mgga_known_funct)[i]->number == functional) break;
  }
  assert(XC(mgga_known_funct)[i] != NULL);
  
  /* initialize structure */
  p->info = XC(mgga_known_funct)[i];

  assert(nspin==XC_UNPOLARIZED || nspin==XC_POLARIZED);
  p->nspin = nspin;
  
  /* initialize the functionals that need it */
  switch(functional){
  case(XC_MGGA_X_TPSS):
    XC(mgga_x_tpss_init)(p);
    break;
  case(XC_MGGA_C_TPSS):
    XC(mgga_c_tpss_init)(p);
    break;
  case(XC_MGGA_X_M06L): 
    XC(mgga_x_m06l_init)(p);
    break;
  case(XC_MGGA_C_M06L): 
    XC(mgga_c_m06l_init)(p);
    break;
  }
}


void XC(mgga_end)(XC(mgga_type) *p)
{
  switch(p->info->number){
  case(XC_MGGA_X_TPSS) :
    XC(mgga_x_tpss_end)(p);
    break;
  case(XC_MGGA_C_TPSS) :
    XC(mgga_c_tpss_end)(p);
    break;
  case(XC_MGGA_X_M06L) : 
    XC(mgga_x_m06l_end)(p);
    break;
  case(XC_MGGA_C_M06L) : 
    XC(mgga_c_m06l_end)(p);
    break;
  }
}


void XC(mgga)(XC(mgga_type) *p, FLOAT *rho, FLOAT *sigma, FLOAT *tau,
	  FLOAT *e, FLOAT *dedd, FLOAT *vsigma, FLOAT *dedtau)

{
  FLOAT dens;

  assert(p!=NULL);
  
  dens = rho[0];
  if(p->nspin == XC_POLARIZED) dens += rho[1];
  
  if(dens <= MIN_DENS){
    int i, n;
    *e = 0.0;
    for(i=0; i<  p->nspin; i++){
      dedd  [i] = 0.0;
      dedtau[i] = 0.0;
    }
    n = (p->nspin == XC_UNPOLARIZED) ? 1 : 3;
    for(i=0; i<n; i++)
      vsigma[i] = 0.0;
    return;
  }
  
  switch(p->info->number){
  case(XC_MGGA_X_TPSS):
    XC(mgga_x_tpss)(p, rho, sigma, tau, e, dedd, vsigma, dedtau);
    break;
  case(XC_MGGA_C_TPSS):
    XC(mgga_c_tpss)(p, rho, sigma, tau, e, dedd, vsigma, dedtau);
    break;
  case(XC_MGGA_X_M06L):
	XC(mgga_x_m06l)(p, rho, sigma, tau, e, dedd, vsigma, dedtau);
    break;
  case(XC_MGGA_C_M06L):
	XC(mgga_c_m06l)(p, rho, sigma, tau, e, dedd, vsigma, dedtau);
    break;
  }

}

