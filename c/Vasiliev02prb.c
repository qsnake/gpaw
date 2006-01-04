#include <stdio.h>
#include <math.h>

#define PI 3.1415927

double sqr(double x) { return x*x; }

double d2exdn2_(double n) {
  if(n<=0) return 0;
  return - pow(PI/sqr(3*PI*n),1./3.);
}

/* vasiliev.f -- translated by f2c (version 20000817).
*/

/* Table of constant values */

static double c_b2 = .33333333333333331;
static double c_b5 = 1.3333333333333333;
static double c_b7 = 2.;
static double c_b8 = -.33333333333333331;
static double c_b24 = -.66666666666666663;

/*     $Id: vasiliev.F,v 1.4 2005/10/31 18:37:04 miwalter Exp $ */
/*     derivates for d^2 E_xc^{LDA}/d\rho^2 after */
/*     Vasiliev, Phys ReV B, 65, pp 115416 (2002) */
/*     second derivative of the exchange energy, eq. (34) */
/*<       function d2Exdnsdnt(nu,nd,ssp,tsp) >*/
double d2exdnsdnt_(nu, nd, ssp, tsp)
double *nu, *nd;
const int *ssp, *tsp;
{
    /* System generated locals */
    double ret_val, d__1;

    /* Local variables */
    static double n2, pi;

/*<       implicit none >*/
/*<       real*8 d2Exdnsdnt,nu,nd   ! nu,nd = up,down desnisty >*/
/*<       integer ssp,tsp           ! spin indicees (1/2) >*/
/*<       real*8 n2,pi >*/
/*<       d2Exdnsdnt=0. >*/
    ret_val = 0.;
/*<       if(ssp.ne.tsp) return >*/
    if (*ssp != *tsp) {
	return ret_val;
    }
/*<       if(ssp.eq.1) then >*/
    if (*ssp == 1) {
/*<          n2=nu*nu >*/
	n2 = *nu * *nu;
/*<       else  >*/
    } else {
/*<          n2=nd*nd >*/
	n2 = *nd * *nd;
/*<       endif >*/
    }
/* ccc      write(*,*) '<d2Exdnsdnt> n2=',n2 */
/*<       if(n2.eq.0) return >*/
    if (n2 == 0.) {
	return ret_val;
    }
/*<       pi=4.*atan(1.) >*/
    pi = PI;
/*<       d2Exdnsdnt=-(2./(9.*pi*n2))**(1./3.) >*/
    d__1 = 2. / (pi * 9. * n2);
    ret_val = -pow(d__1, 1./3.);
/*<       return >*/
    return ret_val;
/*<       end >*/
} /* d2exdnsdnt_ */

/*     second derivative of the correlation energy, eq. (42) */
/*<       function d2Ecdnsdnt(nu,nd,ssp,tsp) >*/
double d2ecdnsdnt_(nu, nd, ssp, tsp)
double *nu, *nd;
const int *ssp, *tsp;
{
    /* System generated locals */
    double ret_val;

    /* Local variables */
    static double n;
    extern double decdrho_p__(), decdrho_u__();
    static double dxins, dxint;
    extern double ecorltn_p__(), ecorltn_u__();
    static double dxidnsdnt;
    extern double d2xidnudnd_();
    static double xd;
    extern double dxidnd_(), d2ecdrho2_p__();
    static double xu;
    extern double d2ecdrho2_u__(), dxidnu_(), d2xidnd2_(), d2xidnu2_(), 
	    xi_vasi__();

/*<       implicit none >*/
/*<       real*8 d2Ecdnsdnt,nu,nd   ! nu,nd = up,down desnisty >*/
/*<       integer ssp,tsp           ! spin indicees (1/2) >*/
/*<       real*8 n,xu,xd >*/
/*<       real*8 dxins,dxint,dxidnsdnt >*/
/*<       real*8 dxidnu,dxidnd,d2xidnu2,d2xidnd2,d2xidnudnd >*/
/*<    >*/
/*     avoid negative or zero densities */
/*<       if(nu.le.0) nu=1.d-99 >*/
    if (*nu <= 0.) {
	*nu = 1e-99;
    }
/*<       if(nd.le.0) nd=1.d-99 >*/
    if (*nd <= 0.) {
	*nd = 1e-99;
    }
/*<       d2Ecdnsdnt=0. >*/
    ret_val = (float)0.;
/*<       n=nu+nd >*/
    n = *nu + *nd;
/*<       if(n.eq.0) return >*/
    if (n == 0.) {
	return ret_val;
    }
/*<       xu=nu/n >*/
    xu = *nu / n;
/*<       xd=nd/n >*/
    xd = *nd / n;
/*<       if(ssp.eq.1) then >*/
    if (*ssp == 1) {
/*<          dxins=dxidnu(nu,nd) >*/
	dxins = dxidnu_(nu, nd);
/*<          if(tsp.eq.1) then      ! u u >*/
	if (*tsp == 1) {
/*<             dxint=dxidnu(nu,nd) >*/
	    dxint = dxidnu_(nu, nd);
/*<             dxidnsdnt=d2xidnu2(nu,nd) >*/
	    dxidnsdnt = d2xidnu2_(nu, nd);
/*<          else                   ! u d >*/
	} else {
/*<             dxint=dxidnd(nu,nd) >*/
	    dxint = dxidnd_(nu, nd);
/*<             dxidnsdnt=d2xidnudnd(nu,nd) >*/
	    dxidnsdnt = d2xidnudnd_(nu, nd);
/*<          endif >*/
	}
/*<       else >*/
    } else {
/*<          dxins=dxidnd(nu,nd) >*/
	dxins = dxidnd_(nu, nd);
/*<          if(tsp.eq.1) then      ! d u >*/
	if (*tsp == 1) {
/*<             dxint=dxidnu(nu,nd) >*/
	    dxint = dxidnu_(nu, nd);
/*<             dxidnsdnt=d2xidnudnd(nu,nd) >*/
	    dxidnsdnt = d2xidnudnd_(nu, nd);
/*<          else                   ! d d >*/
	} else {
/*<             dxint=dxidnd(nu,nd) >*/
	    dxint = dxidnd_(nu, nd);
/*<             dxidnsdnt=d2xidnd2(nu,nd) >*/
	    dxidnsdnt = d2xidnd2_(nu, nd);
/*<          endif >*/
	}
/*<       endif >*/
    }
/*<    >*/
    ret_val = d2ecdrho2_u__(&n) + xi_vasi__(nu, nd) * (d2ecdrho2_p__(&n) - 
	    d2ecdrho2_u__(&n)) + (dxins + dxint) * (decdrho_p__(&n) - 
	    decdrho_u__(&n)) + dxidnsdnt * n * (ecorltn_p__(&n) - ecorltn_u__(
	    &n));
/*<       end >*/
    return ret_val;
} /* d2ecdnsdnt_ */

/*     up/down density interpolation and derivatives */
/*     eq. (41) */
/*<       function xi_vasi(nu,nd) >*/
double xi_vasi__(nu, nd)
double *nu, *nd;
{
    /* System generated locals */
    double ret_val;

    /* Local variables */
    static double n, xd, xu;

/*<       implicit none >*/
/*<       real*8 xi_vasi,nu,nd      ! nu,nd = up,down desnisty  >*/
/*<       real*8 n,xu,xd >*/
/*<       n=nu+nd >*/
    n = *nu + *nd;
/*<       xu=nu/n >*/
    xu = *nu / n;
/*<       xd=nd/n >*/
    xd = *nd / n;
/*<    >*/
    ret_val = (pow(xu, c_b5) + pow(xd, c_b5) - pow(c_b7, c_b8))
	     / ((float)1. - pow(c_b7,c_b8));
/*<       end >*/
    return ret_val;
} /* xi_vasi__ */

/*     eq. (43) */
/*<       function dxidnu(nu,nd) >*/
double dxidnu_(nu, nd)
double *nu, *nd;
{
    /* System generated locals */
    double ret_val;

    /* Local variables */
    static double n, xd, xu;

/*<       implicit none >*/
/*<       real*8 dxidnu,nu,nd       ! nu,nd = up,down desnisty  >*/
/*<       real*8 n,xu,xd >*/
/*<       n=nu+nd >*/
    n = *nu + *nd;
/*<       xu=nu/n >*/
    xu = *nu / n;
/*<       xd=nd/n >*/
    xd = *nd / n;
/*<    >*/
    ret_val = (pow(xu, c_b2) - pow(xu, c_b5) - pow(xd, c_b5)) *
	     (float)4. / (n * (float)3. * ((float)1. - pow(c_b7, c_b8)));
/*<       end >*/
    return ret_val;
} /* dxidnu_ */

/*     eq. (43) */
/*<       function dxidnd(nu,nd) >*/
double dxidnd_(nu, nd)
double *nu, *nd;
{
    /* System generated locals */
    double ret_val;

    /* Local variables */
    static double n, xd, xu;

/*<       implicit none >*/
/*<       real*8 dxidnd,nu,nd       ! nu,nd = up,down desnisty  >*/
/*<       real*8 n,xu,xd >*/
/*<       n=nu+nd >*/
    n = *nu + *nd;
/*<       xu=nu/n >*/
    xu = *nu / n;
/*<       xd=nd/n >*/
    xd = *nd / n;
/*<    >*/
    ret_val = (pow(xd, c_b2) - pow(xd, c_b5) - pow(xu, c_b5)) *
	     (float)4. / (n * (float)3. * ((float)1. - pow(c_b7, c_b8)));
/*<       end >*/
    return ret_val;
} /* dxidnd_ */

/*     eq. (44) */
/*<       function d2xidnu2(nu,nd) >*/
double d2xidnu2_(nu, nd)
double *nu, *nd;
{
    /* System generated locals */
    double ret_val;

    /* Local variables */
    static double n, xd, xu;

/*<       implicit none >*/
/*<       real*8 d2xidnu2,nu,nd       ! nu,nd = up,down desnisty  >*/
/*<       real*8 n,xu,xd >*/
/*<       n=nu+nd >*/
    n = *nu + *nd;
/*<       xu=nu/n >*/
    xu = *nu / n;
/*<       xd=nd/n >*/
    xd = *nd / n;
/*<    >*/
    ret_val = (pow(xu, c_b24) - pow(xu, c_b2) * (float)8. + (pow(
	    xu, c_b5) + pow(xd, c_b5)) * (float)7.) * (float)4. / (n * 
	    (float)9. * n * ((float)1. - pow(c_b7, c_b8)));
/*<       end >*/
    return ret_val;
} /* d2xidnu2_ */

/*     eq. (44) */
/*<       function d2xidnd2(nu,nd) >*/
double d2xidnd2_(nu, nd)
double *nu, *nd;
{
    /* System generated locals */
    double ret_val;

    /* Local variables */
    static double n, xd, xu;

/*<       implicit none >*/
/*<       real*8 d2xidnd2,nu,nd       ! nu,nd = up,down desnisty  >*/
/*<       real*8 n,xu,xd >*/
/*<       n=nu+nd >*/
    n = *nu + *nd;
/*<       xu=nu/n >*/
    xu = *nu / n;
/*<       xd=nd/n >*/
    xd = *nd / n;
/*<    >*/
    ret_val = (pow(xd, c_b24) - pow(xd, c_b2) * (float)8. + (pow(
	    xd, c_b5) + pow(xu, c_b5)) * (float)7.) * (float)4. / (n * 
	    (float)9. * n * ((float)1. - pow(c_b7, c_b8)));
/*<       end >*/
    return ret_val;
} /* d2xidnd2_ */

/*     eq. (45) */
/*<       function d2xidnudnd(nu,nd) >*/
double d2xidnudnd_(nu, nd)
double *nu, *nd;
{
    /* System generated locals */
    double ret_val;

    /* Local variables */
    static double n, xd, xu;

/*<       implicit none >*/
/*<       real*8 d2xidnudnd,nu,nd       ! nu,nd = up,down desnisty  >*/
/*<       real*8 n,xu,xd >*/
/*<       n=nu+nd >*/
    n = *nu + *nd;
/*<       xu=nu/n >*/
    xu = *nu / n;
/*<       xd=nd/n >*/
    xd = *nd / n;
/*<    >*/
    ret_val = ((pow(xu, c_b5) + pow(xd, c_b5)) * (float)7. - (
	    pow(xu, c_b2) + pow(xd, c_b2)) * (float)4.) * (float)4. 
	    / (n * (float)9. * n * ((float)1. - pow(c_b7, c_b8)));
/*<       end >*/
    return ret_val;
} /* d2xidnudnd_ */

/*     correlation energy completely polarised, eq. (35) */
/*<       function ecorltn_P(n)  >*/
double ecorltn_p__(n)
double *n;
{
    /* System generated locals */
    double ret_val;

    /* Local variables */
    static double cvas;
    extern /* Subroutine */ int ldavasip_();
    static double beta1, beta2, a, b, d__, gamma, x;
    extern double ecorltn_();

/*<       implicit none >*/
/*<       real*8  ecorltn_P, n >*/
/*<       real*8 A,B,Cvas,D,X,gamma,beta1,beta2 >*/
/*<       real*8 ecorltn >*/
/*<       call LDAvasiP(A,B,Cvas,D,X,gamma,beta1,beta2) >*/
    ldavasip_(&a, &b, &cvas, &d__, &x, &gamma, &beta1, &beta2);
/*<       ecorltn_P = ecorltn(n,A,B,Cvas,D,X,gamma,beta1,beta2) >*/
    ret_val = ecorltn_(n, &a, &b, &cvas, &d__, &x, &gamma, &beta1, &beta2);
/*<       end >*/
    return ret_val;
} /* ecorltn_p__ */

/*     correlation energy completely unpolarised, eq. (35) */
/*<       function ecorltn_U(n)  >*/
double ecorltn_u__(n)
double *n;
{
    /* System generated locals */
    double ret_val;

    /* Local variables */
    static double cvas;
    extern /* Subroutine */ int ldavasiu_();
    static double beta1, beta2, a, b, d__, gamma, x;
    extern double ecorltn_();

/*<       implicit none >*/
/*<       real*8  ecorltn_U, n >*/
/*<       real*8 A,B,Cvas,D,X,gamma,beta1,beta2 >*/
/*<       real*8 ecorltn >*/
/*<       call LDAvasiU(A,B,Cvas,D,X,gamma,beta1,beta2) >*/
    ldavasiu_(&a, &b, &cvas, &d__, &x, &gamma, &beta1, &beta2);
/*<       ecorltn_U = ecorltn(n,A,B,Cvas,D,X,gamma,beta1,beta2) >*/
    ret_val = ecorltn_(n, &a, &b, &cvas, &d__, &x, &gamma, &beta1, &beta2);
/*<       end >*/
    return ret_val;
} /* ecorltn_u__ */

/*     function used by ecorltn_P and ecorltn_U, eq. (35) */
/*<       function ecorltn(n,A,B,Cvas,D,X,gamma,beta1,beta2) >*/
double ecorltn_(n, a, b, cvas, d__, x, gamma, beta1, beta2)
double *n, *a, *b, *cvas, *d__, *x, *gamma, *beta1, *beta2;
{
    /* System generated locals */
    double ret_val, d__1;

    /* Builtin functions */
    double sqrt(), log();

    /* Local variables */
    static double lnrs, rsrt, pi, rs;

/*<       implicit none >*/
/*<       real*8 ecorltn >*/
/*<       real*8 n,A,B,Cvas,D,X,gamma,beta1,beta2 >*/
/*<       real*8 rs, pi, rsrt, lnrs >*/
/*<       pi=4.*atan(1.) >*/
    pi = PI;
/*<       rs=(3./(4.*pi*n))**(1/3.) >*/
    d__1 = (float)3. / (pi * (float)4. * *n);
    rs = pow(d__1, c_b2);
/*<       rsrt=sqrt(rs) >*/
    rsrt = sqrt(rs);
/*<       lnrs=log(rs) >*/
    lnrs = log(rs);
/*<       if(rs.lt.1) then >*/
    if (rs < 1.) {
/*<          ecorltn=A*lnrs + B + Cvas*rs*lnrs + D*rs + X*rs*rs*lnrs >*/
	ret_val = *a * lnrs + *b + *cvas * rs * lnrs + *d__ * rs + *x * rs * 
		rs * lnrs;
/*<       else >*/
    } else {
/*<          ecorltn=gamma/(1.+beta1*rsrt+beta2*rs) >*/
	ret_val = *gamma / (*beta1 * rsrt + (float)1. + *beta2 * rs);
/*<       endif >*/
    }
/*<       end >*/
    return ret_val;
} /* ecorltn_ */

/*     first derivative completely polarised, eq. (36) */
/*<       function dEcdrho_P(n) >*/
double decdrho_p__(n)
double *n;
{
    /* System generated locals */
    double ret_val;

    /* Local variables */
    static double cvas;
    extern /* Subroutine */ int ldavasip_();
    static double beta1, beta2, a, b, d__, gamma, x;
    extern double decdrho_();

/*<       implicit none >*/
/*<       real*8  dEcdrho_P, n >*/
/*<       real*8 A,B,Cvas,D,X,gamma,beta1,beta2 >*/
/*<       real*8 dEcdrho >*/
/*<       if(n.le.0) then >*/
    if (*n <= 0.) {
/*<          dEcdrho_P=0 >*/
	ret_val = 0.;
/*<          return >*/
	return ret_val;
/*<       endif >*/
    }
/*<       call LDAvasiP(A,B,Cvas,D,X,gamma,beta1,beta2) >*/
    ldavasip_(&a, &b, &cvas, &d__, &x, &gamma, &beta1, &beta2);
/*<       dEcdrho_P = dEcdrho(n,A,B,Cvas,D,X,gamma,beta1,beta2) >*/
    ret_val = decdrho_(n, &a, &b, &cvas, &d__, &x, &gamma, &beta1, &beta2);
/*<       end >*/
    return ret_val;
} /* decdrho_p__ */

/*     first derivative completely unpolarised, eq. (36) */
/*<       function dEcdrho_U(n) >*/
double decdrho_u__(n)
double *n;
{
    /* System generated locals */
    double ret_val;

    /* Local variables */
    static double cvas;
    extern /* Subroutine */ int ldavasiu_();
    static double beta1, beta2, a, b, d__, gamma, x;
    extern double decdrho_();

/*<       implicit none >*/
/*<       real*8  dEcdrho_U, n >*/
/*<       real*8 A,B,Cvas,D,X,gamma,beta1,beta2 >*/
/*<       real*8 dEcdrho >*/
/*<       if(n.le.0) then >*/
    if (*n <= 0.) {
/*<          dEcdrho_U=0 >*/
	ret_val = 0.;
/*<          return >*/
	return ret_val;
/*<       endif >*/
    }
/*<       call LDAvasiU(A,B,Cvas,D,X,gamma,beta1,beta2) >*/
    ldavasiu_(&a, &b, &cvas, &d__, &x, &gamma, &beta1, &beta2);
/*<       dEcdrho_U = dEcdrho(n,A,B,Cvas,D,X,gamma,beta1,beta2) >*/
    ret_val = decdrho_(n, &a, &b, &cvas, &d__, &x, &gamma, &beta1, &beta2);
/*<       end >*/
    return ret_val;
} /* decdrho_u__ */

/*     used by dEcdrho_P and dEcdrho_U, eq. (36) */
/*<       function dEcdrho(n,A,B,Cvas,D,X,gamma,beta1,beta2) >*/
double decdrho_(n, a, b, cvas, d__, x, gamma, beta1, beta2)
double *n, *a, *b, *cvas, *d__, *x, *gamma, *beta1, *beta2;
{
    /* System generated locals */
    double ret_val, d__1;

    /* Builtin functions */
    double sqrt(), log();

    /* Local variables */
    static double lnrs, rsrt, pi, rs;

/*<       implicit none >*/
/*<       real*8 dEcdrho >*/
/*<       real*8 n,A,B,Cvas,D,X,gamma,beta1,beta2 >*/
/*<       real*8 rs, pi, rsrt, lnrs >*/
/*<       pi=4.*atan(1.) >*/
    pi = PI;
/*<       rs=(3./(4.*pi*n))**(1/3.) >*/
    d__1 = (float)3. / (pi * (float)4. * *n);
    rs = pow(d__1, c_b2);
/*<       rsrt=sqrt(rs) >*/
    rsrt = sqrt(rs);
/*<       lnrs=log(rs) >*/
    lnrs = log(rs);
/*<       if(rs.lt.1) then >*/
    if (rs < 1.) {
/*<    >*/
	ret_val = (*a * (float)3. * lnrs + *b * (float)3. - *a + *cvas * (
		float)2. * rs * lnrs + (*d__ * (float)2. - *cvas) * rs + *x * 
		rs * rs * (lnrs - (float)1.)) / (float)3.;
/*<       else >*/
    } else {
/*<    >*/
/* Computing 2nd power */
	d__1 = *beta1 * rsrt + (float)1. + *beta2 * rs;
	ret_val = *gamma * (*beta1 * (float)7. * rsrt + (float)6. + *beta2 * 
		8 * rs) / (d__1 * d__1 * 6);
/*<       endif >*/
    }
/*<       end >*/
    return ret_val;
} /* decdrho_ */

/*     second derivative completely polarised, eq. (37) */
/*<       function d2Ecdrho2_P(n) >*/
double d2ecdrho2_p__(n)
double *n;
{
    /* System generated locals */
    double ret_val;

    /* Local variables */
    extern double d2ecdrho2_();
    static double cvas;
    extern /* Subroutine */ int ldavasip_();
    static double beta1, beta2, a, b, d__, gamma, x;

/*<       implicit none >*/
/*<       real*8  d2Ecdrho2_P, n >*/
/*<       real*8 A,B,Cvas,D,X,gamma,beta1,beta2 >*/
/*<       real*8 d2Ecdrho2 >*/
/*<       if(n.le.0) then >*/
    if (*n <= 0.) {
/*<          d2Ecdrho2_P=0 >*/
	ret_val = 0.;
/*<          return >*/
	return ret_val;
/*<       endif >*/
    }
/*<       call LDAvasiP(A,B,Cvas,D,X,gamma,beta1,beta2) >*/
    ldavasip_(&a, &b, &cvas, &d__, &x, &gamma, &beta1, &beta2);
/*<       d2Ecdrho2_P = d2Ecdrho2(n,A,B,Cvas,D,X,gamma,beta1,beta2) >*/
    ret_val = d2ecdrho2_(n, &a, &b, &cvas, &d__, &x, &gamma, &beta1, &beta2);
/*<       end >*/
    return ret_val;
} /* d2ecdrho2_p__ */

/*     second derivative completely unpolarised, eq. (37) */
/*<       function d2Ecdrho2_U(n) >*/
double d2ecdrho2_u__(n)
double *n;
{
    /* System generated locals */
    double ret_val;

    /* Local variables */
    extern double d2ecdrho2_();
    static double cvas;
    extern /* Subroutine */ int ldavasiu_();
    static double beta1, beta2, a, b, d__, gamma, x;

/*<       implicit none >*/
/*<       real*8  d2Ecdrho2_U, n >*/
/*<       real*8 A,B,Cvas,D,X,gamma,beta1,beta2 >*/
/*<       real*8 d2Ecdrho2 >*/
/*<       if(n.le.0) then >*/
    if (*n <= 0.) {
/*<          d2Ecdrho2_U=0 >*/
	ret_val = 0.;
/*<          return >*/
	return ret_val;
/*<       endif >*/
    }
/*<       call LDAvasiU(A,B,Cvas,D,X,gamma,beta1,beta2) >*/
    ldavasiu_(&a, &b, &cvas, &d__, &x, &gamma, &beta1, &beta2);
/*<       d2Ecdrho2_U = d2Ecdrho2(n,A,B,Cvas,D,X,gamma,beta1,beta2) >*/
    ret_val = d2ecdrho2_(n, &a, &b, &cvas, &d__, &x, &gamma, &beta1, &beta2);
/*<       end >*/
    return ret_val;
} /* d2ecdrho2_u__ */

/*     function used by d2Ecdrho2_U and d2Ecdrho2_U, eq. (37) */
/*<       function d2Ecdrho2(n,A,B,Cvas,D,X,gamma,beta1,beta2) >*/
double d2ecdrho2_(n, a, b, cvas, d__, x, gamma, beta1, beta2)
double *n, *a, *b, *cvas, *d__, *x, *gamma, *beta1, *beta2;
{
    /* System generated locals */
    double ret_val, d__1, d__2;

    /* Builtin functions */
    double sqrt(), log();

    /* Local variables */
    static double lnrs, rsrt, pi, rs;

/*<       implicit none >*/
/*<       real*8 d2Ecdrho2 >*/
/*<       real*8 n,A,B,Cvas,D,X,gamma,beta1,beta2 >*/
/*<       real*8 rs, pi, rsrt, lnrs >*/
/*<       pi=4.*atan(1.) >*/
    pi = PI;
/*<       rs=(3./(4.*pi*n))**(1/3.) >*/
    d__1 = (float)3. / (pi * (float)4. * *n);
    rs = pow(d__1, c_b2);
/*<       rsrt=sqrt(rs) >*/
    rsrt = sqrt(rs);
/*<       lnrs=log(rs) >*/
    lnrs = log(rs);
/*<       if(rs.lt.1) then >*/
    if (rs < 1.) {
/*<    >*/
	ret_val = -(*a * (float)3. + (*d__ * (float)2. + *cvas) * rs + *cvas *
		 (float)2. * rs * lnrs + *x * rs * rs * (lnrs * (float)2. - (
		float)1.)) / (*n * (float)9.);
/*<       else >*/
    } else {
/*<    >*/
/* Computing 2nd power */
	d__1 = *beta1;
/* Computing 3rd power */
	d__2 = *beta1 * rsrt + (float)1. + *beta2 * rs;
	ret_val = *gamma * (*beta1 * (float)5. * rsrt + (d__1 * d__1 * (float)
		7. + *beta2 * 8) * rs + *beta1 * (float)21. * *beta2 * rsrt * 
		rs + *beta2 * (float)16. * *beta2 * rs * rs) / (*n * 36 * (
		d__2 * (d__2 * d__2)));
/*<       endif >*/
    }
/*<       end >*/
    return ret_val;
} /* d2ecdrho2_ */

/*     LDA parametrisation polarized, Table I */
/*<       subroutine LDAvasiP(A,B,Cvas,D,X,gamma,beta1,beta2) >*/
/* Subroutine */ int ldavasip_(a, b, cvas, d__, x, gamma, beta1, beta2)
double *a, *b, *cvas, *d__, *x, *gamma, *beta1, *beta2;
{
/*<       real*8 A,B,Cvas,D,X,gamma,beta1,beta2 >*/
/*<       A=0.01555 >*/
    *a = (float).01555;
/*<       B=-0.0269 >*/
    *b = (float)-.0269;
/*<       Cvas=-0.0005 >*/
    *cvas = (float)-5e-4;
/*<       D=-0.0048 >*/
    *d__ = (float)-.0048;
/*<       X=0.0012 >*/
    *x = (float).0012;
/*<       gamma=-0.0843 >*/
    *gamma = (float)-.0843;
/*<       beta1=1.3981 >*/
    *beta1 = (float)1.3981;
/*<       beta2=0.2611 >*/
    *beta2 = (float).2611;
/*<       return >*/
    return 0;
/*<       end >*/
} /* ldavasip_ */

/*     LDA parametrisation unpolarized, Table I */
/*<       subroutine LDAvasiU(A,B,Cvas,D,X,gamma,beta1,beta2) >*/
/* Subroutine */ int ldavasiu_(a, b, cvas, d__, x, gamma, beta1, beta2)
double *a, *b, *cvas, *d__, *x, *gamma, *beta1, *beta2;
{
/*<       real*8 A,B,Cvas,D,X,gamma,beta1,beta2 >*/
/*<       A=0.0311 >*/
    *a = (float).0311;
/*<       B=-0.048 >*/
    *b = (float)-.048;
/*<       Cvas=-0.0015 >*/
    *cvas = (float)-.0015;
/*<       D=-0.0116 >*/
    *d__ = (float)-.0116;
/*<       X=0.0036 >*/
    *x = (float).0036;
/*<       gamma=-0.1423 >*/
    *gamma = (float)-.1423;
/*<       beta1=1.0529 >*/
    *beta1 = (float)1.0529;
/*<       beta2=0.3334 >*/
    *beta2 = (float).3334;
/*<       return >*/
    return 0;
/*<       end >*/
} /* ldavasiu_ */

