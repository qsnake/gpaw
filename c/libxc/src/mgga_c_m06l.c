
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "util.h"


/************************************************************************
 Implements Zhao, Truhlar
   Meta-gga M06-Local

  Correlation part
************************************************************************/

#define XC_MGGA_C_M06L          204 /* Zhao, Truhlar correlation */

const XC(func_info_type) XC(func_info_mgga_c_m06l) = {
  XC_MGGA_C_M06L,
  XC_CORRELATION,
  "Zhao, Truhlar",
  XC_FAMILY_MGGA,
  "Zhao, Truhlar JCP 125, 194101 (2006)",
  XC_PROVIDES_EXC | XC_PROVIDES_VXC
};


void XC(mgga_c_m06l_init)(XC(mgga_type) *p)
{
  p->info = &XC(func_info_mgga_c_m06l);

    p->lda_aux2 = (XC(lda_type) *) malloc(sizeof(XC(lda_type)));
    XC(lda_init)(p->lda_aux2, XC_LDA_C_PW, XC_POLARIZED);

}


void XC(mgga_c_m06l_end)(XC(mgga_type) *p)
{

    free(p->lda_aux2);
}


/* derivatives of x and z with respect to rho, grho and tau*/
static void 
c_m06l_zx(FLOAT x, FLOAT z, FLOAT rho, FLOAT tau, FLOAT *dxdd, FLOAT *dxdgd, FLOAT *dzdd, FLOAT *dzdtau)
{
    *dxdd = -8./3. * x * 1/rho;
	*dxdgd = 1./pow(rho,8./3.);

	*dzdd = -5./3. * 2 * tau/pow(rho, 8./3.);
	*dzdtau = 2./pow(rho, 5./3.);
}

/* Get g for Eq. (13)*/
static void
c_m06_13(FLOAT *x, FLOAT *rho, FLOAT *g_ab, FLOAT *dg_abdd, FLOAT *dg_abdgd)
{
	/*define the C_ab,i */
  static FLOAT c_ab0= 0.6042374, c_ab1= 177.6783, c_ab2= -251.3252, c_ab3=76.35173, c_ab4=-12.55699;
  FLOAT gammaCab = 0.0031 ;
  FLOAT x_ab, a; 
  FLOAT dg_abdx, dxdd_a, dxdgd_a, dzdd_a, dzdtau_a;
  FLOAT dxdd_b, dxdgd_b, dzdd_b, dzdtau_b;

  /*x = x_ba^2 = x_a^2+x_b^2*/
  x_ab = x[0] + x[1];

  a= (gammaCab*x_ab/(1+gammaCab*x_ab));

  *g_ab = c_ab0*pow(a,0)+ c_ab1*pow(a,1)+ c_ab2*pow(a,2)+c_ab3*pow(a,3)+c_ab4*pow(a,4);

  FLOAT dadx = gammaCab/pow(1+gammaCab*x_ab, 2.);
  dg_abdx = (0.0*c_ab0*pow(a,-1)+ 1.*c_ab1*pow(a,0)+ 2.*c_ab2*pow(a,1)+3.*c_ab3*pow(a,2)+4.*c_ab4*pow(a,3))*dadx;
    
  c_m06l_zx(x[0], 0.0, rho[0], 0.0, &dxdd_a, &dxdgd_a, &dzdd_a, &dzdtau_a);
  c_m06l_zx(x[1], 0.0, rho[1], 0.0, &dxdd_b, &dxdgd_b, &dzdd_b, &dzdtau_b);

  dg_abdd[0] = dg_abdx*dxdd_a; 
  dg_abdd[1] = dg_abdx*dxdd_b; 
  dg_abdgd[0] = dg_abdx*dxdgd_a; 
  dg_abdgd[1] = 0.0;
  dg_abdgd[2] = dg_abdx*dxdgd_b; 
}

/* Get g for Eq. (15)*/
static void
c_m06_15(FLOAT x, FLOAT rho, FLOAT *g_ss, FLOAT *dg_ssdd, FLOAT *dg_ssdgd)
{
	/*define the C_ss,i */
  static FLOAT c_ss0=0.5349466, c_ss1=0.5396620, c_ss2=-31.61217, c_ss3= 51.49592, c_ss4=-29.19613;
  FLOAT gammaCss = 0.06 ;
  FLOAT a; 
  FLOAT dg_ssdx, dxdd, dxdgd, dzdd, dzdtau;

  /*x = x_a^2 */

  a= (gammaCss*x/(1+gammaCss*x));

  *g_ss = c_ss0*pow(a,0)+ c_ss1*pow(a,1)+ c_ss2*pow(a,2)+c_ss3*pow(a,3)+c_ss4*pow(a,4);

  FLOAT dadx = gammaCss/pow(1+gammaCss*x, 2.);
  dg_ssdx = (0.0*c_ss0*pow(a,-1)+ 1.*c_ss1*pow(a,0)+ 2.*c_ss2*pow(a,1)+3.*c_ss3*pow(a,2)+4.*c_ss4*pow(a,3))*dadx;

  c_m06l_zx(x, 0.0, rho, 0.0, &dxdd, &dxdgd, &dzdd, &dzdtau);

  *dg_ssdd = dg_ssdx*dxdd; 
  *dg_ssdgd = dg_ssdx*dxdgd; 
  /*printf("g_ss %19.12f\n", *g_ss);*/
    
}

/* Get h_ab for Eq. (12)*/
static 
void c_m06l_hab(FLOAT *x, FLOAT *z, FLOAT *rho, FLOAT *tau, FLOAT *h_ab, FLOAT *dh_abdd, FLOAT *dh_abdgd, FLOAT *dh_abdtau)
{
	/* define the d_ab,i for Eq. (12)*/
	static FLOAT d_ab0= 0.3957626, d_ab1= -0.5614546, d_ab2= 0.01403963, d_ab3= 0.0009831442, d_ab4= -0.003577176;
	FLOAT alpha_ab = 0.00304966; 
	FLOAT hab1, dhabdd1[2], dhabdgd1[3], dhabdtau1[2];
	FLOAT x_ab, z_ab, gamma, xgamma, zgamma;
	FLOAT dgammadx, dgammadz;
	FLOAT dgammadd_a, dgammadgd_a, dgammadtau_a;
	FLOAT dgammadd_b, dgammadgd_b, dgammadtau_b;
  	FLOAT dxdd_a, dxdgd_a, dzdd_a, dzdtau_a;
  	FLOAT dxdd_b, dxdgd_b, dzdd_b, dzdtau_b;
  
    x_ab = x[0] + x[1];
	z_ab = z[0] + z[1];
	gamma = 1 + alpha_ab*(x_ab + z_ab);
	{ /* derivatives of gamma with respect to x and z*/ 
		dgammadx = alpha_ab;        
        dgammadz = alpha_ab;
    }

	c_m06l_zx(x[0], z[0], rho[0], tau[0], &dxdd_a, &dxdgd_a, &dzdd_a, &dzdtau_a);
	c_m06l_zx(x[1], z[1], rho[1], tau[1], &dxdd_b, &dxdgd_b, &dzdd_b, &dzdtau_b);

	{ /*derivatives of gamma with respect to density, gradient and kietic energy*/
		dgammadd_a   = dgammadx * dxdd_a + dgammadz * dzdd_a;
	    dgammadd_b   = dgammadx * dxdd_b + dgammadz * dzdd_b;
	    dgammadgd_a  = dgammadx * dxdgd_a;
	    dgammadgd_b  = dgammadx * dxdgd_b;
	    dgammadtau_a = dgammadz * dzdtau_a;
	    dgammadtau_b = dgammadz * dzdtau_b;
    }

	xgamma = x_ab/gamma;
	zgamma = z_ab/gamma;

	/* we initialize h and collect the terms*/
  	hab1    = 0.0;
  	dhabdd1[0]   = dhabdd1[1]   = 0.0;
 	dhabdgd1[0]  = dhabdgd1[1]  = dhabdgd1[2] = 0.0;
  	dhabdtau1[0] = dhabdtau1[1] = 0.0;


  { /* first term */
	FLOAT g2=pow(gamma,2.);

    hab1         +=  d_ab0/gamma; 
    dhabdd1[0]   += -d_ab0*dgammadd_a/g2;
    dhabdd1[1]   += -d_ab0*dgammadd_b/g2;
    dhabdgd1[0]  += -d_ab0*dgammadgd_a/g2;
    dhabdgd1[1]  +=  0.0;
    dhabdgd1[2]  += -d_ab0*dgammadgd_b/g2;
    dhabdtau1[0] += -d_ab0*dgammadtau_a/g2 ;
    dhabdtau1[1] += -d_ab0*dgammadtau_b/g2 ;
  }

  { /* second term */
	FLOAT g3=pow(gamma,3.);

    hab1         += (d_ab1*xgamma + d_ab2*zgamma)/gamma;
    dhabdd1[0]   += (gamma*(d_ab1*dxdd_a+d_ab2*dzdd_a)-2*dgammadd_a*(d_ab1*x_ab+d_ab2*z_ab))/g3;
    dhabdd1[1]   += (gamma*(d_ab1*dxdd_b+d_ab2*dzdd_b)-2*dgammadd_b*(d_ab1*x_ab+d_ab2*z_ab))/g3;
    dhabdgd1[0]  += (d_ab1*dxdgd_a*gamma -2*(d_ab1*x_ab+d_ab2*z_ab)*dgammadgd_a)/g3;
    dhabdgd1[1]  += 0.0;
    dhabdgd1[2]  += (d_ab1*dxdgd_b*gamma -2*(d_ab1*x_ab+d_ab2*z_ab)*dgammadgd_b)/g3;
    dhabdtau1[0] += (d_ab2*dzdtau_a*gamma -2*(d_ab1*x_ab+d_ab2*z_ab)*dgammadtau_a)/g3;
    dhabdtau1[1] += (d_ab2*dzdtau_b*gamma -2*(d_ab1*x_ab+d_ab2*z_ab)*dgammadtau_b)/g3;
  }
  
  { /* third term */
	FLOAT g4= pow(gamma,4);

    hab1      += (d_ab3*xgamma*xgamma+d_ab4*xgamma*zgamma)/gamma;
    dhabdd1[0]   += (-3*dgammadd_a*(d_ab3*pow(x_ab,2.)+d_ab4*x_ab*z_ab)+dxdd_a*gamma*(2*d_ab3*x_ab+d_ab4*z_ab)+d_ab4*x_ab*dzdd_a*gamma)/g4;
    dhabdd1[1]   += (-3*dgammadd_b*(d_ab3*pow(x_ab,2.)+d_ab4*x_ab*z_ab)+dxdd_b*gamma*(2*d_ab3*x_ab+d_ab4*z_ab)+d_ab4*x_ab*dzdd_b*gamma)/g4;
    dhabdgd1[0]  += (-3*x_ab*(d_ab3*x_ab+d_ab4*z_ab)*dgammadgd_a+gamma*(2*d_ab3*x_ab+d_ab4*z_ab)*dxdgd_a)/g4;
    dhabdgd1[1]  += 0.0;
    dhabdgd1[2]  += (-3*x_ab*(d_ab3*x_ab+d_ab4*z_ab)*dgammadgd_b+gamma*(2*d_ab3*x_ab+d_ab4*z_ab)*dxdgd_b)/g4;
    dhabdtau1[0] += (d_ab4*x_ab*dzdtau_a*gamma-3*x_ab*(d_ab3*x_ab+d_ab4*z_ab)*dgammadtau_a)/g4;
    dhabdtau1[1] += (d_ab4*x_ab*dzdtau_b*gamma-3*x_ab*(d_ab3*x_ab+d_ab4*z_ab)*dgammadtau_b)/g4;
  }
  	*h_ab = hab1;
	//derivatives
	dh_abdd[0]   = dhabdd1[0];
	dh_abdd[1]   = dhabdd1[1];
	dh_abdgd[0]  = dhabdgd1[0];
	dh_abdgd[1]  = dhabdgd1[1];
	dh_abdgd[2]  = dhabdgd1[2];
	dh_abdtau[0] = dhabdtau1[0];
	dh_abdtau[1] = dhabdtau1[1];

}

/* Get h_ss for Eq. (14)*/
static 
void c_m06l_hss(FLOAT x, FLOAT z, FLOAT rho, FLOAT tau, FLOAT *h_ss, FLOAT *dh_ssdd, FLOAT *dh_ssdgd, FLOAT *dh_ssdtau)
{
	/* define the d_ab,i for Eq. (12)*/
	static FLOAT d_ss0= 0.4650534, d_ss1= 0.1617589, d_ss2= 0.1833657, d_ss3= 0.0004692100, d_ss4= -0.004990573;
	FLOAT alpha_ss = 0.00515088; 
	FLOAT hss1, dhssdd1, dhssdgd1, dhssdtau1;
	FLOAT gamma, xgamma, zgamma;
	FLOAT dgammadx, dgammadz;
	FLOAT dgammadd, dgammadgd, dgammadtau;
  	FLOAT dxdd, dxdgd, dzdd, dzdtau;
  

	gamma = 1 + alpha_ss*(x + z);
	{ /* derivatives of gamma with respect to x and z*/ 
		dgammadx = alpha_ss;        
        dgammadz = alpha_ss;
	}

    c_m06l_zx(x, z, rho, tau, &dxdd, &dxdgd, &dzdd, &dzdtau);

	{ /* derivatives of gamma with respect to density, gradient and kinetic energy */ 
		dgammadd   = dgammadx * dxdd + dgammadz * dzdd;
	    dgammadgd  = dgammadx * dxdgd;
	    dgammadtau = dgammadz * dzdtau;
	}

	xgamma = x/gamma;
	zgamma = z/gamma;

	/* we initialize h and collect the terms*/
  	hss1    = 0.0;
  	dhssdd1 = 0.0;
 	dhssdgd1 = 0.0;
  	dhssdtau1 = 0.0;


  { /* first term */
	FLOAT g2=pow(gamma,2.);

    hss1    +=  d_ss0/gamma; 
    dhssdd1   += -d_ss0*dgammadd/g2;
    dhssdgd1  += -d_ss0*dgammadgd/g2;
    dhssdtau1 += -d_ss0*dgammadtau/g2 ;
  }


  { /* second term */
	FLOAT g3=pow(gamma,3.);

    hss1      += (d_ss1*xgamma + d_ss2*zgamma)/gamma;
    dhssdd1   += (gamma*(d_ss1*dxdd+d_ss2*dzdd)-2*dgammadd*(d_ss1*x+d_ss2*z))/g3;
    dhssdgd1  += (d_ss1*dxdgd*gamma -2*(d_ss1*x+d_ss2*z)*dgammadgd)/g3;
    dhssdtau1 += (d_ss2*dzdtau*gamma -2*(d_ss1*x+d_ss2*z)*dgammadtau)/g3;
  }
  
  { /* third term */
	FLOAT g4= pow(gamma,4);

    hss1    += (d_ss3*xgamma*xgamma+d_ss4*xgamma*zgamma)/gamma;
    dhssdd1   += (-3*dgammadd*(d_ss3*pow(x,2.)+d_ss4*x*z)+dxdd*gamma*(2*d_ss3*x+d_ss4*z)+d_ss4*x*dzdd*gamma)/g4;
    dhssdgd1  += (-3*x*(d_ss3*x+d_ss4*z)*dgammadgd+gamma*(2*d_ss3*x+d_ss4*z)*dxdgd)/g4;
    dhssdtau1 += (d_ss4*x*dzdtau*gamma-3*x*(d_ss3*x+d_ss4*z)*dgammadtau)/g4;
  }
  	*h_ss = hss1;
	//derivatives
	*dh_ssdd   = dhssdd1;
	*dh_ssdgd  = dhssdgd1;
	*dh_ssdtau = dhssdtau1;


}


void 
c_m06l_para(xc_mgga_type *p, FLOAT *rho, FLOAT *sigma_, FLOAT *tau,
	    FLOAT *energy, FLOAT *dedd, FLOAT *vsigma, FLOAT *dedtau)
{
  FLOAT rho2[2], rho2s[2], x[2], z[2], zc_ss[2];
  FLOAT tau2[2], tauw[2], dens, dens1, sigma[3];
  FLOAT g_ss[2], h_ss[2], Ec_ss[2], D_ss[2]; 
  FLOAT g_ab=0.0, h_ab=0.0, Ec_ab=0.0; 
  FLOAT exunif_ss[2], vxunif_up[2], vxunif_dn[2], vxunif_ss[2];
  FLOAT exunif =0.0, exunif_ab=0.0, vxunif[2];
  //derivatives
  FLOAT dh_ssdd[2], dh_ssdgd[3], dh_ssdtau[2];
  FLOAT dg_ssdd[2], dg_ssdgd[3] ;
  FLOAT dh_abdd[2], dh_abdgd[3], dh_abdtau[2];
  FLOAT dg_abdd[2], dg_abdgd[3];
  FLOAT dEc_ssdd[2], dEc_ssdgd[3], dEc_ssdtau[2];
  FLOAT dEc_abdd[2], dEc_abdgd[3], dEc_abdtau[2];
  FLOAT dD_ssdd[2], dD_ssdgd[3], dD_ssdtau[2], dD_ssdx[2], dD_ssdz[2];
  FLOAT dxdd[2], dxdgd[2], dzdd[2], dzdtau[2];

  const FLOAT Cfermi= (3./5.)*pow(6*M_PI*M_PI,2./3.); 
  

  /*calculate |nabla rho|^2 */
  sigma_[0] = max(MIN_GRAD*MIN_GRAD, sigma_[0]);
  tauw[0] = sigma_[0]/(8.0*rho[0]);
  tau[0] = max(tauw[0], tau[0]);


  dens1 = rho[0]+rho[1];

  if(p->nspin== XC_UNPOLARIZED)
    {
	  tau[1]  = 0.0; 

      rho2[0] = rho[0]/2.;
	  rho2[1] = rho[0]/2.;	
      sigma[0] = sigma_[0]/4.;
      sigma[1] = sigma_[0]/4.;
      sigma[2] = sigma_[0]/4.;
      dens = rho[0];

	  tau2[0] = tau[0]/2.;
	  tau2[1] = tau[0]/2.;

    }else{
      sigma_[2] = max(MIN_GRAD*MIN_GRAD, sigma_[2]);
	  tauw[1] = sigma_[2]/(8.0*rho[1]);
      tau[1] = max(tauw[1], tau[1]);

      rho2[0]=rho[0];
	  rho2[1]=rho[1];	
      sigma[0] = sigma_[0];
      sigma[1] = sigma_[1];
      sigma[2] = sigma_[2];
      dens = rho[0]+rho[1];

	  tau2[0] =tau[0];
	  tau2[1] =tau[1];

    }
	  //get the e_LDA(rho_a,b)
      XC(lda_vxc)(p->lda_aux2, rho2, &exunif, vxunif);
	  exunif = exunif*dens;

	  /*==============get the E_sigma part================*/
             /*============ spin up =============*/

      rho2s[0]=rho2[0];
	  rho2s[1]=0.;	

	  //get the e_LDA(rho_up,0)
      XC(lda_vxc)(p->lda_aux2, rho2s, &(exunif_ss[0]), vxunif_up);
	  exunif_ss[0] = exunif_ss[0] * rho2s[0];
	  vxunif_ss[0] = vxunif_up[0];

	  /*define variables for rho_up and zc in order to avoid x/0 -> D_ss = -inf */
	  x[0] = sigma[0]/(pow(rho2s[0], 8./3.)); 
      z[0] = 2*tau2[0]/pow(rho2s[0],5./3.) - Cfermi;
	  zc_ss[0] = 2*tau2[0]/pow(rho2s[0],5./3.);

	  /*D_ss = 1 -x/4*(z + Cf), z+Cf = 2*tau2/pow(rho2s[0],5./3.) = zc */
	  D_ss[0] = 1 - x[0]/(4. * zc_ss[0]);
	  //derivatives for D_up
	  dD_ssdx[0] = -1/(4 * zc_ss[0]);
	  dD_ssdz[0] =  4 * x[0]/pow(4.*zc_ss[0],2.);

      c_m06l_zx(x[0], z[0], rho2s[0], tau2[0], &(dxdd[0]), &(dxdgd[0]), &(dzdd[0]), &(dzdtau[0]));
	  
	  dD_ssdd[0]   = dD_ssdx[0] * dxdd[0] + dD_ssdz[0] * dzdd[0];
	  dD_ssdgd[0]  = dD_ssdx[0] * dxdgd[0];
	  dD_ssdtau[0] = dD_ssdz[0] * dzdtau[0];

	  /*build up Eq. (14): Ec_sigmasigma*/
	  c_m06_15(x[0], rho2s[0], &(g_ss[0]), &(dg_ssdd[0]), &(dg_ssdgd[0]));
	  c_m06l_hss(x[0], z[0], rho2s[0], tau2[0], &(h_ss[0]), &(dh_ssdd[0]), &(dh_ssdgd[0]), &(dh_ssdtau[0]));

	  Ec_ss[0] = (exunif_ss[0] * (g_ss[0]+h_ss[0]) * D_ss[0]);
	  //printf("Ec_up %.9e\n", Ec_ss[0]);

               /*============== spin down =============*/

      rho2s[0]=rho2[1];
	  rho2s[1]=0.;	
	  
	  //get the e_LDA(0,rho_dn)
	  XC(lda_vxc)(p->lda_aux2, rho2s, &(exunif_ss[1]), vxunif_dn);
	  exunif_ss[1] = exunif_ss[1] * rho2s[0];
	  vxunif_ss[1] = vxunif_dn[0];

	  /*define variables for rho_beta*/
	  x[1] = sigma[2]/(pow(rho2s[0], 8./3.)); 
      z[1] = 2*tau2[1]/pow(rho2s[0],5./3.) - Cfermi;
	  zc_ss[1] = 2*tau2[1]/pow(rho2s[0],5./3.);

	  //printf("x1 %.9e, zc_ss%.9e\n", x[1], zc_ss[1]);
	  D_ss[1] = 1 - x[1]/(4.*zc_ss[1]);
	  //derivatives for D_dn
	  dD_ssdx[1] = - 1/(4*zc_ss[1]);
	  dD_ssdz[1] = 4*x[1]/pow(4.*zc_ss[1],2.);

      c_m06l_zx(x[1], z[1], rho2s[0], tau2[1], &(dxdd[1]), &(dxdgd[1]), &(dzdd[1]), &(dzdtau[1]));
	  
	  dD_ssdd[1]   = dD_ssdx[1] * dxdd[1] + dD_ssdz[1] * dzdd[1];
	  dD_ssdgd[2]  = dD_ssdx[1] * dxdgd[1];
	  dD_ssdtau[1] = dD_ssdz[1] * dzdtau[1];

	  c_m06_15(x[1], rho2s[0], &(g_ss[1]), &(dg_ssdd[1]), &(dg_ssdgd[2]));
	  c_m06l_hss(x[1], z[1], rho2s[0], tau2[1], &(h_ss[1]), &(dh_ssdd[1]), &(dh_ssdgd[2]), &(dh_ssdtau[1]));


	  //printf("exunif_ss %.9e, (g_ss[1]+h_ss[1])%.9e, D_ss %.9e\n", exunif_ss[1],(g_ss[1]+h_ss[1]),D_ss[1]);
	  Ec_ss[1] = (exunif_ss[1] * (g_ss[1]+h_ss[1]) * D_ss[1]);
	  //printf("Ec_dn %.9e\n", Ec_ss[1]);
	  
	  // Derivatives for Ec_up and Ec_dn with respect to density and kinetic energy
	  int i;
	  for(i=0; i<2; i++){

	  dEc_ssdd[i]   = exunif_ss[i] * dh_ssdd[i] * D_ss[i] + vxunif_ss[i] * h_ss[i] * D_ss[i] + exunif_ss[i] * h_ss[i] * dD_ssdd[i] +
	                  exunif_ss[i] * dg_ssdd[i] * D_ss[i] + vxunif_ss[i] * g_ss[i] * D_ss[i] + exunif_ss[i] * g_ss[i] * dD_ssdd[i];

	  dEc_ssdtau[i] = exunif_ss[i] * dh_ssdtau[i] * D_ss[i] + exunif_ss[i] * h_ss[i] * dD_ssdtau[i] + exunif_ss[i] * g_ss[i] * dD_ssdtau[i];

	  }
	  // Derivatives for Ec_up and Ec_dn with respect to gradient
	  dEc_ssdgd[0]  = exunif_ss[0] * dh_ssdgd[0] * D_ss[0] + exunif_ss[0] * h_ss[0] * dD_ssdgd[0] +
	                  exunif_ss[0] * dg_ssdgd[0] * D_ss[0] + exunif_ss[0] * g_ss[0] * dD_ssdgd[0];
	  dEc_ssdgd[2]  = exunif_ss[1] * dh_ssdgd[2] * D_ss[1] + exunif_ss[1] * h_ss[1] * dD_ssdgd[2] + 
	                  exunif_ss[1] * dg_ssdgd[2] * D_ss[1] + exunif_ss[1] * g_ss[1] * dD_ssdgd[2];

	  
	  /*==============get the E_ab part========================*/

	  exunif_ab = exunif - exunif_ss[0] - exunif_ss[1];

	  //x_ab = sigmatot[0] /(pow(rho2[0], 8./3.)) + sigmatot[2] /(pow(rho2[1], 8./3.));
      //z_ab = 2*tau2[0]/pow(rho2[0],5./3.) + 2*tau2[1]/pow(rho2[1],5./3.) - 2*Cfermi;

	  /*build up Eq. (12): Ec_alphabeta*/
	  c_m06_13(x, rho2, &g_ab, dg_abdd, dg_abdgd);
	  c_m06l_hab(x, z, rho2, tau2, &h_ab, dh_abdd, dh_abdgd, dh_abdtau);

	  Ec_ab = exunif_ab * (g_ab+h_ab);

	  // Derivatives for Ec_ab with respect to density and kinetic energy
	  for(i=0; i<2; i++){

	  dEc_abdd[i]   = exunif_ab * (dh_abdd[i]+ dg_abdd[i]) + (vxunif[i]- vxunif_ss[i]) * (g_ab+h_ab);
	  dEc_abdtau[i] = exunif_ab * dh_abdtau[i];

	  }
	  // Derivatives for Ec_ab with respect to gradient
	  for(i=0; i<3; i++){
	  dEc_abdgd[i] = exunif_ab * (dh_abdgd[i] + dg_abdgd[i]);
	  }

	  /*==============get the total energy E_c= E_up + E_dn + E_ab========================*/
	  /*==============================and derivatives=====================================*/

	  *energy = (Ec_ss[0] + Ec_ss[1] + Ec_ab)/dens1;
	  //printf("Ec_ss %.9e, Ec_ss %.9e, Ec_ab %.9e\n", Ec_ss[0], Ec_ss[1], Ec_ab);

	  
	  //derivative for the total correlation energy
	  if(p->nspin== XC_UNPOLARIZED)
	  {
      dedd[0]=dEc_ssdd[0] + dEc_abdd[0];
      dedd[1]=0.0;
	  
	  vsigma[0]= (dEc_ssdgd[0] + dEc_abdgd[0])/2.;
	  vsigma[1]= 0.0;
	  vsigma[2]= 0.0;

      dedtau[0]= dEc_ssdtau[0] + dEc_abdtau[0];
      dedtau[1]= 0.0;
	  }else{
      dedd[0]=dEc_ssdd[0] + dEc_abdd[0];
      dedd[1]=dEc_ssdd[1] + dEc_abdd[1];
	  
	  vsigma[0]= dEc_ssdgd[0] + dEc_abdgd[0];
	  vsigma[1]= 0.0;
	  vsigma[2]= dEc_ssdgd[2] + dEc_abdgd[2];

      dedtau[0]= dEc_ssdtau[0] + dEc_abdtau[0];
      dedtau[1]= dEc_ssdtau[1] + dEc_abdtau[1];
	  }


}
void 
XC(mgga_c_m06l)(XC(mgga_type) *p, FLOAT *rho, FLOAT *sigma, FLOAT *tau,
	    FLOAT *e, FLOAT *dedd, FLOAT *vsigma, FLOAT *dedtau)
{
    c_m06l_para(p, rho, sigma, tau, e, dedd, vsigma, dedtau);
}
