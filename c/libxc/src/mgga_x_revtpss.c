#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "util.h"

/************************************************************************
 Implements John P. Perdew, Adrienn Ruzsinszky, Gabor I. Csonka, Lucian A. Constantin, and Jianwei Sun
   meta-Generalized Gradient Approximation.

  Exchange part
************************************************************************/

#define XC_MGGA_X_REVTPSS          205 /* Perdew, Ruzsinszky, Csonka, Constantin and Sun Exchange*/
#define NMIN   1.0E-10

#define   _(is, x)   [3*is + x]

/*changes static with const*/
const XC(func_info_type) XC(func_info_mgga_x_revtpss) = {
  XC_MGGA_X_REVTPSS,
  XC_EXCHANGE,
  "Perdew, Ruzsinszky, Csonka, Constantin and Sun",
  XC_FAMILY_MGGA,
  "Perdew, Ruzsinszky, Csonka, Constantin and Sun PRL 103 026403 (2009)", 
  XC_PROVIDES_EXC | XC_PROVIDES_VXC
};


void XC(mgga_x_revtpss_init)(XC(mgga_type) *p)
{
  p->info = &XC(func_info_mgga_x_revtpss);

  p->lda_aux = (XC(lda_type) *) malloc(sizeof(XC(lda_type)));
  XC(lda_x_init)(p->lda_aux, XC_UNPOLARIZED, 3, XC_NON_RELATIVISTIC);

}


void XC(mgga_x_revtpss_end)(XC(mgga_type) *p)
{
  free(p->lda_aux);
}


/* some parameters */
static FLOAT b=0.40, c=2.35204, e=2.1677, kappa=0.804, mu=0.14;


/* This is Equation (7) from the paper and its derivatives */
static void 
x_revtpss_7(FLOAT p, FLOAT alpha, 
	 FLOAT *qb, FLOAT *dqbdp, FLOAT *dqbdalpha)
{

   /* Eq. (7) */
    FLOAT a = sqrt(1.0 + b*alpha*(alpha-1.0)), h = 9.0/20.0;

    *qb    = h*(alpha - 1.0)/a + 2.0*p/3.0;
    *dqbdp = 2.0/3.0;
    *dqbdalpha = h*(1.0 + 0.5*b*(alpha-1.0))/POW(a, 3);
  

}

/* Equation (10) in all it's glory */
static 
void x_revtpss_10(FLOAT p, FLOAT alpha,
	       FLOAT *x, FLOAT *dxdp, FLOAT *dxdalpha)
{
  FLOAT x1, dxdp1, dxdalpha1;
  FLOAT aux1, ap, apsr, p2;
  FLOAT qb, dqbdp, dqbdalpha;
  
  /* Equation 7 */
  x_revtpss_7(p, alpha, &qb, &dqbdp, &dqbdalpha);

  p2   = p*p; 
  aux1 = 10.0/81.0;
  ap = (3*alpha + 5*p)*(3*alpha + 5*p);
  apsr = (3*alpha + 5*p);
  
  /* first we handle the numerator */
  x1    = 0.0;
  dxdp1 = 0.0;
  dxdalpha1 = 0.0;

  { /* first term */
    FLOAT a = (9*alpha*alpha+30*alpha*p+50*p2), a2 = a*a;
    x1    += aux1*p + 125*c*p2*p2*apsr/a2;
    dxdp1 += aux1 + ((4*c*p2*p*125*3*alpha+ 125*25*c*p2*p2)*a2 - 125*c*p2*p2*apsr*2*a*(30*alpha+50*2*p))/(a2*a2);
    dxdalpha1 += ((125*3*c*p2*p2)*a2 - 125*c*p2*p2*apsr*2*a*(9*2*alpha+30*p))/(a2*a2);
  }
  
  { /* second term */
    FLOAT a = 146.0/2025.0*qb;
    x1    += a*qb;
	dxdp1 += 2.0*a*dqbdp;
	dxdalpha1 += 2.0*a*dqbdalpha;
  }
  
  { /* third term */
    FLOAT h = 73.0/(405*sqrt(2.0));
    x1    += -h*qb*p/apsr * sqrt(ap+9);
    dxdp1 += -h * qb *((3*alpha)/ap * sqrt(ap+9) + p/apsr * 1./2. * POW(ap+9,-1./2.)* 2*apsr*5) - h*p/apsr*sqrt(ap+9)*dqbdp; 
	dxdalpha1 += -h*qb*( (-1)*p*3/ap * sqrt(ap+9) + p/apsr * 1./2. * POW(ap+9,-1./2.)* 2*apsr*3) - h*p/apsr*sqrt(ap+9)*dqbdalpha;
  }
  
  { /* forth term */
    FLOAT a = aux1*aux1/kappa;
    x1    += a*p2;
    dxdp1 += a*2.0*p;
	dxdalpha1 += 0.0;
  }
  
  { /* fifth term */
    x1    += 20*sqrt(e)*p2/(9*ap);
    dxdp1 += 20*sqrt(e)/9*(2*p*ap-p2*2*(3*alpha + 5*p)*5)/(ap*ap);
	dxdalpha1 +=-20*2*sqrt(e)/3*p2/(ap*(3*alpha + 5*p));
  }
  
  { /* sixth term */
    FLOAT a = e*mu;
    x1    += a*p*p2;
    dxdp1 += a*3.0*p2;
	dxdalpha1 += 0.0;
  }
  
  /* and now the denominator */
  {
    FLOAT a = 1.0+sqrt(e)*p, a2 = a*a;
    *x    = x1/a2;
    *dxdp = (dxdp1*a - 2.0*sqrt(e)*x1)/(a2*a);
	*dxdalpha = dxdalpha1/a2;
  }
}

static void 
x_revtpss_para(XC(mgga_type) *pt, FLOAT *rho, FLOAT sigma, FLOAT tau_,
	    FLOAT *energy, FLOAT *dedd, FLOAT *vsigma, FLOAT *dedtau)
{

  FLOAT gdms, p, tau, tauw;
  FLOAT x, dxdp, dxdalpha, Fx, dFxdx;
  FLOAT tau_lsda, exunif, vxunif, dtau_lsdadd;
  FLOAT dpdd, dpdsigma;
  FLOAT alpha, dalphadtau_lsda, dalphadd, dalphadsigma, dalphadtau; 
  FLOAT aux =  (3./10.) * pow((3*M_PI*M_PI),2./3.); 


  /* get the uniform gas energy and potential */
  XC(lda_vxc)(pt->lda_aux, rho, &exunif, &vxunif);

  /* calculate |nabla rho|^2 */
  gdms = max(MIN_GRAD*MIN_GRAD, sigma);
  
  /* Eq. (4) */
  p = gdms/(4.0*POW(3*M_PI*M_PI, 2.0/3.0)*POW(rho[0], 8.0/3.0));
  dpdd = -(8.0/3.0)*p/rho[0];
  dpdsigma= 1/(4.0*POW(3*M_PI*M_PI, 2.0/3.0)*POW(rho[0], 8.0/3.0));

  /* von Weisaecker kinetic energy density */
  tauw = max(gdms/(8.0*rho[0]), 1.0e-12);
  tau = max(tau_, tauw);

  tau_lsda = aux * pow(rho[0],5./3.); 
  dtau_lsdadd = aux * 5./3.* pow(rho[0],2./3.);
  
  alpha = (tau - tauw)/tau_lsda;
  dalphadtau_lsda = -1./POW(tau_lsda,2.);
  

  if(ABS(tauw-tau_)< 1.0e-10){
	  dalphadsigma = 0.0;
	  dalphadtau = 0.0;
	  dalphadd = 0.0; 
  }else{
	  dalphadtau = 1./tau_lsda;
	  dalphadsigma = -1./(tau_lsda*8.0*rho[0]);
	  dalphadd = (tauw/rho[0]* tau_lsda - (tau - tauw) * dtau_lsdadd)/ POW(tau_lsda,2.); 
  }

  /* get Eq. (10) */
  x_revtpss_10(p, alpha, &x, &dxdp, &dxdalpha);

  { /* Eq. (5) */
    FLOAT a = kappa/(kappa + x);
    Fx    = 1.0 + kappa*(1.0 - a);
    dFxdx = a*a;
  }
  
  { /* Eq. (3) */

    *energy = exunif*Fx*rho[0];
	//printf("Ex %.9e\n", *energy);

    /* exunif is en per particle already so we multiply by n the terms with exunif*/

    *dedd   = vxunif*Fx + exunif*dFxdx*rho[0]*(dxdp*dpdd + dxdalpha*dalphadd);

    *vsigma = exunif*dFxdx*rho[0]*(dxdp*dpdsigma + dxdalpha*dalphadsigma);

    *dedtau = exunif*dFxdx*rho[0]*(dxdalpha*dalphadtau);


  }
}


void 
XC(mgga_x_revtpss)(XC(mgga_type) *p, FLOAT *rho, FLOAT *sigma, FLOAT *tau,
	    FLOAT *e, FLOAT *dedd, FLOAT *vsigma, FLOAT *dedtau)
{
  if(p->nspin == XC_UNPOLARIZED){
	  FLOAT en;
    x_revtpss_para(p, rho, sigma[0], tau[0], &en, dedd, vsigma, dedtau);
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


		  
      x_revtpss_para(p, rhoa, 4*sigma[0], 2.0*tau[0], &e2na, &(dedd[0]), &(vsigmapart[0]), &(dedtau[0]));

      x_revtpss_para(p, rhob, 4*sigma[2], 2.0*tau[1], &e2nb, &(dedd[1]), &(vsigmapart[2]), &(dedtau[1]));
		 
	  *e = (e2na + e2nb )/(2.*(rho[0]+rho[1]));
	  vsigma[0] = 2*vsigmapart[0];
	  vsigma[2] = 2*vsigmapart[2];
  }
}
