
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "util.h"

#define XC_MGGA_C_TPSS          202 /* Perdew, Tao, Staroverov & Scuseria correlation */
#define NMIN   1.0E-10



/************************************************************************
 Implements Perdew, Tao, Staroverov & Scuseria 
   meta-Generalized Gradient Approximation.
   J. Chem. Phys. 120, 6898 (2004)
   http://dx.doi.org/10.1063/1.1665298

  Correlation part
************************************************************************/

/*changes static with const*/
const XC(func_info_type) XC(func_info_mgga_c_tpss) = {
  XC_MGGA_C_TPSS,
  XC_CORRELATION,
  "Perdew, Tao, Staroverov & Scuseria",
  XC_FAMILY_MGGA,
  "J.P.Perdew, Tao, Staroverov, and Scuseria, Phys. Rev. Lett. 91, 146401 (2003)",
  XC_PROVIDES_EXC | XC_PROVIDES_VXC
};


void XC(mgga_c_tpss_init)(XC(mgga_type) *p)
{
  p->info = &XC(func_info_mgga_c_tpss);

  p->gga_aux1 = (XC(gga_type) *) malloc(sizeof(XC(gga_type)));
  XC(gga_init)(p->gga_aux1, XC_GGA_C_PBE, p->nspin);

  if(p->nspin == XC_UNPOLARIZED){
    p->gga_aux2 = (XC(gga_type) *) malloc(sizeof(XC(gga_type)));
    XC(gga_init)(p->gga_aux2, XC_GGA_C_PBE, XC_POLARIZED);
  }
}


void XC(mgga_c_tpss_end)(XC(mgga_type) *p)
{
  XC(gga_end)(p->gga_aux1);
  free(p->gga_aux1);

  if(p->nspin == XC_UNPOLARIZED) {
    XC(gga_end)(p->gga_aux2);
    free(p->gga_aux2);
  }
}


/* some parameters */
static FLOAT d = 2.8;


/* Equation (14) */
static void
c_tpss_14(FLOAT csi, FLOAT zeta, FLOAT *C, FLOAT *dCdcsi, FLOAT *dCdzeta)
{
  FLOAT fz, C0, dC0dz, dfzdz;
  FLOAT z2 = zeta*zeta;
    
  /* Equation (13) */
  C0    = 0.53 + z2*(0.87 + z2*(0.50 + z2*2.26));
  dC0dz = zeta*(2.0*0.87 + z2*(4.0*0.5 + z2*6.0*2.26));  /*OK*/
  
  fz    = 0.5*(POW(1.0 + zeta, -4.0/3.0) + POW(1.0 - zeta, -4.0/3.0));
  dfzdz = 0.5*(-4.0/3.0)*(POW(1.0 + zeta, -7.0/3.0) - POW(1.0 - zeta, -7.0/3.0)); /*OK*/
  
  { /* Equation (14) */
    FLOAT csi2 = csi*csi;
    FLOAT a = 1.0 + csi2*fz, a4 = POW(a, 4);
    
    *C      =  C0 / a4;
    *dCdcsi = -8.0*C0*csi*fz/(a*a4);  /*added C OK*/
    *dCdzeta = (dC0dz*a - C0*4.0*csi2*dfzdz)/(a*a4);  /*OK*/
  }
}


/* Equation (12) */
static void c_tpss_12(XC(mgga_type) *p, FLOAT *rho, FLOAT *sigma, 
		 FLOAT dens, FLOAT zeta, FLOAT z,
		 FLOAT *e_PKZB, FLOAT *de_PKZBdd, FLOAT *de_PKZBdsigma, FLOAT *de_PKZBdz)
{
  /*some incoming variables:  
   dens = rho[0] + rho[1]
   z = tau_w/tau
   zeta = (rho[0] - rho[1])/dens*/

  FLOAT e_PBE, e_PBEup, e_PBEdn;
  FLOAT de_PBEdd[2], de_PBEdsigma[3], de_PBEddup[2], de_PBEdsigmaup[3], de_PBEdddn[2], de_PBEdsigmadn[3] ;
  FLOAT aux, zsq;
  FLOAT dzetadd[2], dcsidd[2], dcsidsigma[3];  

  FLOAT C, dCdcsi, dCdzeta;
  FLOAT densp[2], densp2[2], sigmatot[3], sigmaup[3], sigmadn[3];
  int i;
  /*initialize dCdcsi and dCdzeta and the energy*/
  dCdcsi = dCdzeta = 0.0;  
  e_PBE = 0.0;
  e_PBEup = 0.0;
  e_PBEdn = 0.0;

  /* get the PBE stuff */
  if(p->nspin== XC_UNPOLARIZED)
    { densp[0]=rho[0]/2.;
      densp[1]=rho[0]/2.;
      sigmatot[0] = sigma[0]/4.;
      sigmatot[1] = sigma[0]/4.;
      sigmatot[2] = sigma[0]/4.;
    }else{
	  densp[0] = rho[0];
	  densp[1] = rho[1];
      sigmatot[0] = sigma[0];
      sigmatot[1] = sigma[1];
      sigmatot[2] = sigma[2];
    }

  /* e_PBE */
  XC(gga_type) *aux2 = (p->nspin == XC_UNPOLARIZED) ? p->gga_aux2 : p->gga_aux1;
  XC(gga_vxc)(aux2, densp, sigmatot, &e_PBE, de_PBEdd, de_PBEdsigma); 

  densp2[0]=densp[0];
  densp2[1]=0.0;

  if(p->nspin== XC_UNPOLARIZED)
  {
      sigmaup[0] = sigma[0]/4.;
      sigmaup[1] = 0.;
      sigmaup[2] = 0.;
  }else{
      sigmaup[0] = sigma[0];
      sigmaup[1] = 0.;
      sigmaup[2] = 0.;
  }
  /* e_PBE spin up */
  XC(gga_vxc)(aux2, densp2, sigmaup, &e_PBEup, de_PBEddup, de_PBEdsigmaup); 
  
  densp2[0]=densp[1];
  densp2[1]=0.0;

  if(p->nspin== XC_UNPOLARIZED)
  {
      sigmadn[0] = sigma[0]/4.;
      sigmadn[1] = 0.;
      sigmadn[2] = 0.;
  }else{
      sigmadn[0] = sigma[2];
      sigmadn[1] = 0.;
      sigmadn[2] = 0.;
  }

  /* e_PBE spin down */
  XC(gga_vxc)(aux2,  densp2, sigmadn, &e_PBEdn, de_PBEdddn, de_PBEdsigmadn); 
  
  /*get Eq. (13) and (14) for the polarized case*/
  if(p->nspin == XC_UNPOLARIZED){   
    C          = 0.53;
    dzetadd[0] = 0.0;
    dcsidd [0] = 0.0;
    for(i=0; i<3; i++) dcsidsigma[i] = 0.0;
  }else{
    // initialize derivatives
    for(i=0; i<2; i++){
    dzetadd[i] = 0.0;
    dcsidd [i] = 0.0;}

    for(i=0; i<3; i++) dcsidsigma[i] = 0.0;



    FLOAT num, gzeta, csi, a;

	  /*numerator of csi: derive as grho all components and then square the 3 parts
	  [2 (grho_a[0]n_b - grho_b[0]n_a) +2 (grho_a[1]n_b - grho_b[1]n_a) + 2 (grho_a[2]n_b - grho_b[2]n_a)]/(n_a+n_b)^2   
	   -> 4 (sigma_aa n_b^2 - 2 sigma_ab n_a n_b + sigma_bb n_b^2)/(n_a+n_b)^2 */

    num = sigma[0] * POW(rho[1],2) - 2.* sigma[1]*rho[0]*rho[1]+ sigma[2]*POW(rho[0],2);
	num = max(num,0);
	gzeta = sqrt(4*(num))/(dens*dens);
	gzeta = max(gzeta, MIN_GRAD);
	  /*denominator of csi*/
	a = 2*POW(3.0*M_PI*M_PI*dens, 1.0/3.0);

	csi = gzeta/a;

	c_tpss_14(csi, zeta, &C, &dCdcsi, &dCdzeta);

	dzetadd[0] =  (1.0 - zeta)/dens; /*OK*/
    dzetadd[1] = -(1.0 + zeta)/dens; /*OK*/


    dcsidd [0] = 0.5*csi*(-2*sigma[1]*rho[1]+2*sigma[2]*rho[0])/num - 7./3.*csi/dens; /*OK*/
    dcsidd [1] = 0.5*csi*(-2*sigma[1]*rho[0]+2*sigma[0]*rho[1])/num - 7./3.*csi/dens; /*OK*/

    dcsidsigma[0]=  csi*POW(rho[1],2)/(2*num);   /*OK*/
    dcsidsigma[1]= -csi*rho[0]*rho[1]/num;  /*OK*/
    dcsidsigma[2]=  csi*POW(rho[0],2)/(2*num);   /*OK*/

    }

  aux = (densp[0] * max(e_PBEup, e_PBE) + densp[1] * max(e_PBEdn, e_PBE)) / dens;

  FLOAT dauxdd[2], dauxdsigma[3];
      
      if(e_PBEup > e_PBE)
       {
	   //case densp[0] * e_PBEup
	   dauxdd[0] = de_PBEddup[0];
	   dauxdd[1] = 0.0;
	   dauxdsigma[0] = de_PBEdsigmaup[0];
	   dauxdsigma[1] = 0.0;
	   dauxdsigma[2] = 0.0;
       }else{
	   //case densp[0] * e_PBE
	   dauxdd[0] = densp[0] / dens * (de_PBEdd[0] - e_PBE) + e_PBE;
	   dauxdd[1] = densp[0] / dens * (de_PBEdd[1] - e_PBE);
	   dauxdsigma[0] = densp[0] / dens * de_PBEdsigma[0];
	   dauxdsigma[1] = densp[0] / dens * de_PBEdsigma[1];
	   dauxdsigma[2] = densp[0] / dens * de_PBEdsigma[2];
       }

      if(e_PBEdn > e_PBE)
       {//case densp[1] * e_PBEdn
	   dauxdd[0] += 0.0;
	   dauxdd[1] += de_PBEdddn[0];
	   dauxdsigma[0] += 0.0;
	   dauxdsigma[1] += 0.0;
	   dauxdsigma[2] += de_PBEdsigmadn[0];
       }else{//case densp[1] * e_PBE
       dauxdd[0] += densp[1] / dens * (de_PBEdd[0] - e_PBE);
       dauxdd[1] += densp[1] / dens * (de_PBEdd[1] - e_PBE) + e_PBE;
       dauxdsigma[0] += densp[1] / dens * de_PBEdsigma[0];
       dauxdsigma[1] += densp[1] / dens * de_PBEdsigma[1];
       dauxdsigma[2] += densp[1] / dens * de_PBEdsigma[2];
       }
 
    zsq=z*z;
    *e_PKZB    = (e_PBE*(1.0 + C * zsq) - (1.0 + C) * zsq * aux);
    *de_PKZBdz = dens * e_PBE * C * 2*z - dens * (1.0 + C) * 2*z * aux;  /*? think ok*/

      
      FLOAT dCdd[2];
      
      dCdd[0] = dCdzeta*dzetadd[0] + dCdcsi*dcsidd[0]; /*OK*/
      dCdd[1] = dCdzeta*dzetadd[1] + dCdcsi*dcsidd[1]; /*OK*/
      
      /* partial derivatives*/
	  de_PKZBdd[0] = de_PBEdd[0] * (1.0 + C*zsq) + dens * e_PBE * dCdd[0] * zsq
		           - zsq * (dens*dCdd[0] * aux + (1.0 + C) * dauxdd[0]);
	  de_PKZBdd[1] = de_PBEdd[1] * (1.0 + C*zsq) + dens * e_PBE * dCdd[1] * zsq
		           - zsq * (dens*dCdd[1] * aux + (1.0 + C) * dauxdd[1]);
			  
	  int nder = (p->nspin==XC_UNPOLARIZED) ? 1 : 3;
      for(i=0; i<nder; i++){
	  if(p->nspin==XC_UNPOLARIZED) dauxdsigma[i] /= 2.;
      FLOAT dCdsigma[i]; 
	  dCdsigma[i]=  dCdcsi*dcsidsigma[i];
	
      /* partial derivatives*/
	  de_PKZBdsigma[i] = de_PBEdsigma[i] * (1.0 + C * zsq) + dens * e_PBE * dCdsigma[i] * zsq
		               - zsq * (dens * dCdsigma[i] * aux + (1.0 + C) * dauxdsigma[i]);

      }
} 


void 
XC(mgga_c_tpss)(XC(mgga_type) *p, FLOAT *rho, FLOAT *sigma, FLOAT *tau,
	    FLOAT *energy, FLOAT *dedd, FLOAT *vsigma, FLOAT *dedtau)
{
  FLOAT dens, zeta;
  FLOAT taut, tauw, z;
  FLOAT e_PKZB, de_PKZBdd[2], de_PKZBdsigma[3], de_PKZBdz;
  int i, is;

  zeta = (rho[0]-rho[1])/(rho[0]+rho[1]);

  dens = rho[0];
  if(p->nspin == XC_POLARIZED) dens += rho[1];

  sigma[0] = max(MIN_GRAD*MIN_GRAD, sigma[0]);
  if(p->nspin == XC_POLARIZED) sigma[2] = max(MIN_GRAD*MIN_GRAD, sigma[2]);

  tauw = max(sigma[0]/(8.0*rho[0]), 1.0e-12);
  if(p->nspin == XC_POLARIZED) tauw += max(sigma[2]/(8.0*rho[1]),1.0e-12);

  /* GMadsen: tau lower bound by tauw*/ 
  taut = max(tau[0]+tau[1], tauw);
  z = tauw/taut;
  /* Equation (12) */
  c_tpss_12(p, rho, sigma, dens, zeta, z,
	    &e_PKZB, de_PKZBdd, de_PKZBdsigma, &de_PKZBdz);

  /* Equation (11) */
  {
    FLOAT z2 = z*z, z3 = z2*z;
    FLOAT dedz;
    FLOAT dzdd[2], dzdsigma[3], dzdtau;

	if(tauw >= tau[0]+tau[1]){
		dzdtau = 0.0;        /*OK*/
	    dzdd[0] = 0.0;          /*OK*/
	    dzdd[1] = 0.0;          /*OK*/
		dzdsigma[0] = 0.0;
		dzdsigma[1] = 0.0;
		dzdsigma[2] = 0.0;
	}else{
		dzdtau = -z/taut;        /*OK*/
		dzdd[0] = - sigma[0]/(8*rho[0]*rho[0]*taut);          /*OK*/
		dzdd[1] = - sigma[2]/(8*rho[1]*rho[1]*taut);          /*OK*/
		dzdsigma[0] = 1.0/(8*rho[0]*taut);    /*OK*/
		dzdsigma[1] = 0.0;  /*OK*/
		dzdsigma[2] = 1.0/(8*rho[1]*taut);    /*OK*/
	}
    
    *energy = e_PKZB * (1.0 + d*e_PKZB*z3);
    /* due to the definition of na and nb in libxc.c we need to divide by (na+nb) to recover the 
	 * same energy for polarized and unpolarized calculation with the same total density */
	if(p->nspin == XC_UNPOLARIZED) *energy *= dens/(rho[0]+rho[1]);
	
     dedz = de_PKZBdz*(1.0 + 2.0*d*e_PKZB*z3) +  dens*e_PKZB * e_PKZB * d * 3.0*z2;  

    for(is=0; is<p->nspin; is++){
      dedd[is]   = de_PKZBdd[is] * (1.0 + 2.0*d*e_PKZB*z3) + dedz*dzdd[is] - e_PKZB*e_PKZB * d * z3; /*OK*/
      dedtau[is] = dedz * dzdtau; /*OK*/
      }
	int nder = (p->nspin==XC_UNPOLARIZED) ? 1 : 3;
    for(i=0; i<nder; i++){  
      vsigma[i] = de_PKZBdsigma[i] * (1.0 + 2.0*d*e_PKZB*z3) + dedz*dzdsigma[i];
      }
  }
}
