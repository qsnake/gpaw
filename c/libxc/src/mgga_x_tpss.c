
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "util.h"

/************************************************************************
 Implements Perdew, Tao, Staroverov & Scuseria 
   meta-Generalized Gradient Approximation.

  Exchange part
************************************************************************/

#define XC_MGGA_X_TPSS          201 /* Perdew, Tao, Staroverov & Scuseria exchange */
#define NMIN   1.0E-10

#define   _(is, x)   [3*is + x]

/*changes static with const*/
const XC(func_info_type) XC(func_info_mgga_x_tpss) = {
  XC_MGGA_X_TPSS,
  XC_EXCHANGE,
  "Perdew, Tao, Staroverov & Scuseria",
  XC_FAMILY_MGGA,
  "J.P.Perdew, Tao, Staroverov, and Scuseria, Phys. Rev. Lett. 91, 146401 (2003)",
  XC_PROVIDES_EXC | XC_PROVIDES_VXC
};


void XC(mgga_x_tpss_init)(XC(mgga_type) *p)
{
  p->info = &XC(func_info_mgga_x_tpss);

  p->lda_aux = (XC(lda_type) *) malloc(sizeof(XC(lda_type)));
  XC(lda_x_init)(p->lda_aux, XC_UNPOLARIZED, 3, XC_NON_RELATIVISTIC);

}


void XC(mgga_x_tpss_end)(XC(mgga_type) *p)
{
  free(p->lda_aux);
}


/* some parameters */
static FLOAT b=0.40, c=1.59096, e=1.537, kappa=0.804, mu=0.21951;


/* This is Equation (7) from the paper and its derivatives */
static void 
x_tpss_7(FLOAT p, FLOAT z, 
	 FLOAT *qb, FLOAT *dqbdp, FLOAT *dqbdz)
{
  FLOAT alpha, dalphadp, dalphadz;

  { /* Eq. (8) */
    FLOAT a = (1.0/z - 1.0), h = 5.0/3.0;
    alpha    = h*a*p;
    dalphadp = h*a;
    dalphadz = -h*p/(z*z);
  }

  { /* Eq. (7) */
    FLOAT dqbda;
    FLOAT a = sqrt(1.0 + b*alpha*(alpha-1.0)), h = 9.0/20.0;
    dqbda = h*(1.0 + 0.5*b*(alpha-1.0))/POW(a, 3);

    *qb    = h*(alpha - 1.0)/a + 2.0*p/3.0;
    *dqbdp = dqbda*dalphadp + 2.0/3.0;
    *dqbdz = dqbda*dalphadz;
  }

}

/* Equation (10) in all it's glory */
static 
void x_tpss_10(FLOAT p, FLOAT z, 
	       FLOAT *x, FLOAT *dxdp, FLOAT *dxdz)
{
  FLOAT x1, dxdp1, dxdz1;
  FLOAT aux1, z2, p2;
  FLOAT qb, dqbdp, dqbdz;
  
  /* Equation 7 */
  x_tpss_7(p, z, &qb, &dqbdp, &dqbdz);

  z2   = z*z;
  p2   = p*p; 
  aux1 = 10.0/81.0;
  
  /* first we handle the numerator */
  x1    = 0.0;
  dxdp1 = 0.0;
  dxdz1 = 0.0;

  { /* first term */
    FLOAT a = 1.0+z2, a2 = a*a;
    x1    += (aux1 + c*z2/a2)*p;
    dxdp1 += (aux1 + c*z2/a2);
    dxdz1 += c*2.0*z*(1.0-z2)*p/(a*a2);
  }
  
  { /* second term */
    FLOAT a = 146.0/2025.0*qb;
    x1    += a*qb;
    dxdp1 += 2.0*a*dqbdp;
    dxdz1 += 2.0*a*dqbdz;
  }
  
  { /* third term */
    FLOAT a = sqrt(0.5*(9.0*z2/25.0 + p2));
    FLOAT h = 73.0/405;
    x1    += -h*qb*a;
    dxdp1 += -h*(a*dqbdp + 0.5*qb*p/a);
    dxdz1 += -h*(a*dqbdz + 0.5*qb*(9.0/25.0)*z/a);
  }
  
  { /* forth term */
    FLOAT a = aux1*aux1/kappa;
    x1    += a*p2;
    dxdp1 += a*2.0*p;
  }
  
  { /* fifth term */
    FLOAT a = 2.0*sqrt(e)*aux1*9.0/25.0;
    x1    += a*z2;
    dxdz1 += a*2.0*z;
  }
  
  { /* sixth term */
    FLOAT a = e*mu;
    x1    += a*p*p2;
    dxdp1 += a*3.0*p2;
  }
  
  /* and now the denominator */
  {
    FLOAT a = 1.0+sqrt(e)*p, a2 = a*a;
    *x    = x1/a2;
    *dxdp = (dxdp1*a - 2.0*sqrt(e)*x1)/(a2*a);
    *dxdz = dxdz1/a2;
  }
}

static void 
x_tpss_para(XC(mgga_type) *pt, FLOAT *rho, FLOAT sigma, FLOAT tau_,
	    FLOAT *energy, FLOAT *dedd, FLOAT *vsigma, FLOAT *dedtau)
{

  FLOAT gdms, p, tau, tauw, z;
  FLOAT x, dxdp, dxdz, Fx, dFxdx;
  FLOAT exunif, vxunif;
  FLOAT dpdd, dpdsigma, dzdtau, dzdd, dzdsigma;


  /* get the uniform gas energy and potential */
  XC(lda_vxc)(pt->lda_aux, rho, &exunif, &vxunif);

  /* calculate |nabla rho|^2 */
  gdms = sigma;
  gdms = max(MIN_GRAD*MIN_GRAD, gdms);
  
  /* Eq. (4) */
  p = gdms/(4.0*POW(3*M_PI*M_PI, 2.0/3.0)*POW(rho[0], 8.0/3.0));
  dpdd = -(8.0/3.0)*p/rho[0];
  dpdsigma= 1/(4.0*POW(3*M_PI*M_PI, 2.0/3.0)*POW(rho[0], 8.0/3.0));

  /* von Weisaecker kinetic energy density */
  tauw = max(gdms/(8.0*rho[0]), 1.0e-12);
  /* GMadsen: tau lower bound by tauw */
  tau = max(tau_, tauw);
  z  = tauw/tau;
  if(tauw >= tau_){
	  dzdtau = 0.0;
	  dzdd = 0.0;
	  dzdsigma = 0.0;
  }else{
	  dzdtau= -z/tau;
	  dzdd = -z/rho[0];
	  dzdsigma = 1/(8*rho[0]*tau);
  }

  /* get Eq. (10) */
  x_tpss_10(p, z, &x, &dxdp, &dxdz);

  { /* Eq. (5) */
    FLOAT a = kappa/(kappa + x);
    Fx    = 1.0 + kappa*(1.0 - a);
    dFxdx = a*a;
  }
  
  { /* Eq. (3) */

    *energy = exunif*Fx*rho[0];
	//printf("Ex %.9e\n", *energy);

    /* exunif is en per particle already so we multiply by n the terms with exunif*/

    *dedd   = vxunif*Fx + exunif*dFxdx*(dpdd*dxdp + dzdd*dxdz)*rho[0];

    *vsigma = exunif*dFxdx*rho[0]*(dxdp*dpdsigma + dxdz*dzdsigma);

    *dedtau = exunif*dFxdx*rho[0]*(dzdtau*dxdz);

  }
}


void 
XC(mgga_x_tpss)(XC(mgga_type) *p, FLOAT *rho, FLOAT *sigma, FLOAT *tau,
	    FLOAT *e, FLOAT *dedd, FLOAT *vsigma, FLOAT *dedtau)
{
  if(p->nspin == XC_UNPOLARIZED){
	  FLOAT en;
    x_tpss_para(p, rho, sigma[0], tau[0], &en, dedd, vsigma, dedtau);
    *e = en/(rho[0]+rho[1]);
  }else{ 
    /* The spin polarized version is handle using the exact spin scaling
          Ex[n1, n2] = (Ex[2*n1] + Ex[2*n2])/2
    */

	  *e = 0.0;

      FLOAT e2na, e2nb, rhoa[2], rhob[2];

      FLOAT vsigmapart[3]; 
	  
	  rhoa[0]=2*rho[0];
	  rhoa[1]=0.0;
	  rhob[0]=2*rho[1];
	  rhob[1]=0.0;


		  
      x_tpss_para(p, rhoa, 4*sigma[0], 2.0*tau[0], &e2na, &(dedd[0]), &(vsigmapart[0]), &(dedtau[0]));

      x_tpss_para(p, rhob, 4*sigma[2], 2.0*tau[1], &e2nb, &(dedd[1]), &(vsigmapart[2]), &(dedtau[1]));
		 
	  *e = (e2na + e2nb )/(2.*(rho[0]+rho[1]));
	  vsigma[0] = 2*vsigmapart[0];
	  vsigma[2] = 2*vsigmapart[2];
  }
}
