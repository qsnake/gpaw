
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "util.h"

/************************************************************************
 Implements Zhao, Truhlar
   Meta-gga M06-Local

  Exchange part
************************************************************************/

#define XC_MGGA_X_M06L          203 /* Zhao, Truhlar exchange */

const XC(func_info_type) XC(func_info_mgga_x_m06l) = {
  XC_MGGA_X_M06L,
  XC_EXCHANGE,
  "Zhao, Truhlar",
  XC_FAMILY_MGGA,
  "Zhao, Truhlar JCP 125, 194101 (2006)",
  XC_PROVIDES_EXC | XC_PROVIDES_VXC
};


void XC(mgga_x_m06l_init)(xc_mgga_type *p)
{
  p->info = &XC(func_info_mgga_x_m06l);

  p->gga_aux1 = (XC(gga_type) *) malloc(sizeof(XC(gga_type)));
  XC(gga_init)(p->gga_aux1, XC_GGA_X_PBE, XC_POLARIZED);
}


void XC(mgga_x_m06l_end)(XC(mgga_type) *p)
{

  XC(gga_end)(p->gga_aux1);
  free(p->gga_aux1);

}

/* derivatives of x and z with respect to rho, grho and tau: Eq.(1) and Eq.(3)*/
static void 
x_m06l_zx(FLOAT x, FLOAT z, FLOAT rho, FLOAT tau, FLOAT *dxdd, FLOAT *dxdgd, FLOAT *dzdd, FLOAT *dzdtau)
{
    *dxdd = -8./3. * x * 1/rho;
	*dxdgd = 1./pow(rho,8./3.);

	*dzdd = -5./3. * 2* tau/pow(rho, 8./3.);
	*dzdtau = 2./pow(rho, 5./3.);
}

/* Build gamma and its derivatives with respect to rho, grho and tau: Eq. (4)*/
static void 
x_m06l_gamma(FLOAT x, FLOAT z, FLOAT rho, FLOAT tau, FLOAT *gamma, FLOAT *dgammadd, FLOAT *dgammadgd, FLOAT *dgammadtau)
{
  static FLOAT alpha = 0.00186726;   /*set alpha of Eq. (4)*/
  FLOAT dgammadx, dgammadz;
  FLOAT dxdd, dxdgd, dzdd, dzdtau;

  *gamma = 1 + alpha*(x + z);
  /*printf("gamma %19.12f\n", *gamma);*/

  { /* derivatives */ 
    dgammadx = alpha;        
    dgammadz = alpha;
  }

  x_m06l_zx(x, z, rho, tau, &dxdd, &dxdgd, &dzdd, &dzdtau);

  {
	  *dgammadd = dgammadx*dxdd + dgammadz*dzdd;
	  *dgammadgd = dgammadx*dxdgd;
	  *dgammadtau = dgammadz*dzdtau;
  }

}

/* calculate h and h derivatives with respect to rho, grho and tau: Equation (5) */
static 
void x_m06l_h(FLOAT x, FLOAT z, FLOAT rho, FLOAT tau, FLOAT *h, FLOAT *dhdd, FLOAT *dhdgd, FLOAT *dhdtau)
{
	/* parameters for h(x_sigma,z_sigma) of Eq. (5)*/
	static FLOAT d0=0.6012244, d1=0.004748822, d2=-0.008635108, d3=-0.000009308062, d4=0.00004482811;

	FLOAT h1, dhdd1, dhdgd1, dhdtau1;
	FLOAT gamma, dgammadd, dgammadgd, dgammadtau;
	FLOAT xgamma, zgamma;
  	FLOAT dxdd, dxdgd, dzdd, dzdtau;
  
	x_m06l_gamma(x, z, rho, tau, &gamma, &dgammadd, &dgammadgd, &dgammadtau);

	xgamma = x/gamma;
	zgamma = z/gamma;

	/* we initialize h and its derivatives and collect the terms*/
  	h1    = 0.0;
  	dhdd1 = 0.0;
 	dhdgd1 = 0.0;
  	dhdtau1 = 0.0;


  { /* first term */
	FLOAT g2=pow(gamma,2.);

    h1      += d0/gamma; 
    dhdd1   += -d0*dgammadd/g2;
    dhdgd1  += -d0*dgammadgd/g2;
    dhdtau1 += -d0*dgammadtau/g2 ;
  }

	x_m06l_zx(x, z, rho, tau, &dxdd, &dxdgd, &dzdd, &dzdtau);
  
  { /* second term */
	FLOAT g3=pow(gamma,3.);

    h1      += (d1*xgamma + d2*zgamma)/gamma;
    dhdd1   += (gamma*(d1*dxdd+d2*dzdd)-2*dgammadd*(d1*x+d2*z))/g3;
    dhdgd1  += (d1*dxdgd*gamma -2*(d1*x+d2*z)*dgammadgd)/g3;
    dhdtau1 += (d2*dzdtau*gamma -2*(d1*x+d2*z)*dgammadtau)/g3;
  }
  
  { /* third term */
	FLOAT g4= pow(gamma,4);

    h1      += (d3*xgamma*xgamma+d4*xgamma*zgamma)/gamma;
    dhdd1   += (-3*dgammadd*(d3*pow(x,2.)+d4*x*z)+dxdd*gamma*(2*d3*x+d4*z)+d4*x*dzdd*gamma)/g4;
    dhdgd1  += (-3*x*(d3*x+d4*z)*dgammadgd+gamma*(2*d3*x+d4*z)*dxdgd)/g4;
    dhdtau1 += (d4*x*dzdtau*gamma-3*x*(d3*x+d4*z)*dgammadtau)/g4;
  }
  	*h = h1;
    /*printf(" h %19.12f\n", *h);*/
	*dhdd = dhdd1;
	*dhdgd =dhdgd1;
	*dhdtau = dhdtau1;

}

/* f(w) and its derivatives with respect to rho and tau*/
static void 
x_m06l_fw(FLOAT rho, FLOAT tau, FLOAT *fw, FLOAT *dfwdd, FLOAT *dfwdtau)
{
	/*define the parameters for fw of Eq. (8) as in the reference paper*/
	static FLOAT a0= 0.3987756, a1= 0.2548219, a2= 0.3923994, a3= -2.103655, a4= -6.302147, a5= 10.97615,
				  a6= 30.97273,  a7=-23.18489,  a8=-56.73480,  a9=21.60364,  a10= 34.21814, a11= -9.049762;

	FLOAT tau_lsda, t, w;
	FLOAT dtdd, dtdtau; 
	FLOAT dfwdw, dwdt, dtau_lsdadd;
	FLOAT aux =  (3./10.) * pow((6*M_PI*M_PI),2./3.); /*3->6 for nspin=2 */
	

	tau_lsda = aux * pow(rho,5./3.); 
	t = tau_lsda/tau;
	dtdtau = -t/tau; 
	w = (t - 1)/(t + 1);

	*fw = a0*pow(w,0.)+a1*pow(w,1.)+a2*pow(w,2.)+a3*pow(w,3.)+a4*pow(w,4.)+
		+ a5*pow(w,5.)+a6*pow(w,6.)+a7*pow(w,7.)+a8*pow(w,8.)+a9*pow(w,9.)+a10*pow(w,10.)+a11*pow(w,11.);

	dfwdw = 0.0*a0*pow(w,-1)+1.0*a1*pow(w,0.)+2.0*a2*pow(w,1.)+3.0*a3*pow(w,2.)+4.0*a4*pow(w,3.)+
			    + 5.0*a5*pow(w,4.)+6.0*a6*pow(w,5.)+7.0*a7*pow(w,6.)+8.0*a8*pow(w,7.)+9.0*a9*pow(w,8.)+
                   + 10*a10*pow(w,9.)+11*a11*pow(w,10.);

	dwdt = 2/pow((t + 1),2.);

	dtau_lsdadd = aux * 5./3.* pow(rho,2./3.);
	dtdd = dtau_lsdadd/tau;

	*dfwdd =   dfwdw * dwdt * dtdd;
	*dfwdtau = dfwdw * dwdt * dtdtau;
}

static void 
x_m06l_para(XC(mgga_type) *pt, FLOAT rho, FLOAT sigma, FLOAT tau, FLOAT *energy, FLOAT *dedd, FLOAT *vsigma, FLOAT *dedtau)
{
	/*Build Eq. (6) collecting the terms Fx_PBE,  fw, e_lsda and h*/
  FLOAT grad, tauw, tau2, x, z;
  FLOAT rho2[2],sigmatot[3];
  FLOAT F_PBE, de_PBEdd[2], de_PBEdgd[3];
  FLOAT h, dhdd, dhdgd, dhdtau;
  FLOAT fw, dfwdd, dfwdtau;
  FLOAT epsx_lsda, depsx_lsdadd;
  const FLOAT Cfermi = (3./5.) * pow(6*M_PI*M_PI,2./3.);


  /* calculate |nabla rho|^2 */
  grad = sigma;
  grad = max(MIN_GRAD*MIN_GRAD, grad);
  tauw = max(grad/(8.0*rho),1.0e-12); /* tau^W = |nabla rho|^2/ 8rho */
  tau = max(tau, tauw);

  rho2[0]=rho/2.;
  rho2[1]=0.0;
  sigmatot[0] = grad/4.;
  sigmatot[1] = 0.0;
  sigmatot[2] = 0.0;
  tau2 =tau/2.;
  

  /* get the uniform gas energy and potential a MINUS was missing in the paper*/

  epsx_lsda = -(3./2.)*pow(3./(4*M_PI),1./3.)*pow(rho2[0],4./3.);

  depsx_lsdadd = -2*pow(3./(4*M_PI),1./3.)*pow(rho2[0],1./3.);

  /*get Fx for PBE*/
  XC(gga_vxc)(pt->gga_aux1, rho2, sigmatot, &F_PBE, de_PBEdd, de_PBEdgd);


  /* define x and z from Eq. (1) and Eq. (3) NOTE: we build directly x^2 */
  x = grad/(4*pow(rho2[0], 8./3.));
  
  z = 2*tau2/pow(rho2[0],5./3.) - Cfermi;  /*THERE IS A 2 IN FRONT AS IN THEOR. CHEM. ACCOUNT 120 215 (2008)*/
  
  /*get  h and fw*/
  x_m06l_h(x, z, rho2[0], tau2, &h, &dhdd, &dhdgd, &dhdtau);
  x_m06l_fw(rho2[0], tau2, &fw, &dfwdd, &dfwdtau);


  
  { /* Eq. (6)  E_x = Int F_PBE*fw + exunif*h, the factor 2 accounts for spin. */

    *energy = 2*(F_PBE*rho2[0] *fw + epsx_lsda *h);

    *dedd   = (de_PBEdd[0] *fw + F_PBE*rho2[0] * dfwdd+ depsx_lsdadd *h + epsx_lsda * dhdd);
    *dedtau = (F_PBE * dfwdtau *rho2[0] + epsx_lsda * dhdtau);

    *vsigma = (de_PBEdgd[0] *fw +  epsx_lsda*dhdgd)/2.;
  }
}


void 
XC(mgga_x_m06l)(XC(mgga_type) *p, FLOAT *rho, FLOAT *sigma, FLOAT *tau,
	    FLOAT *e, FLOAT *dedd, FLOAT *vsigma, FLOAT *dedtau)

{
  if(p->nspin == XC_UNPOLARIZED){
	  FLOAT en;
    x_m06l_para(p, rho[0], sigma[0], tau[0], &en, dedd, vsigma, dedtau);
	*e = en/(rho[0]+rho[1]);

  }else{
  
  
	  *e = 0.0;

      FLOAT e2na, e2nb, rhoa[2], rhob[2];

      FLOAT vsigmapart[3]; 
	  

	  rhoa[0]=2*rho[0];
	  rhoa[1]=0.0;
	  rhob[0]=2*rho[1];
	  rhob[1]=0.0;


		  
      x_m06l_para(p, rhoa[0], 4*sigma[0], 2.0*tau[0], &e2na, &(dedd[0]), &(vsigmapart[0]), &(dedtau[0]));

      x_m06l_para(p, rhob[0], 4*sigma[2], 2.0*tau[1], &e2nb, &(dedd[1]), &(vsigmapart[2]), &(dedtau[1]));
		 
	  *e = (e2na + e2nb )/(2.*(rho[0]+rho[1]));
	  vsigma[0] = 2*vsigmapart[0];
	  vsigma[2] = 2*vsigmapart[2];
  }
}
