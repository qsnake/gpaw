/* $Id: tpss_ec.c,v 1.2 2006/12/09 09:37:40 miwalter Exp $ */
#define Pi 3.1415926535897932384626433832795
#define E 2.7182818284590452353602874713527
#define Power(a,b) ((b==2)?(a)*(a) : ((b==3)?(a)*(a)*(a):pow(a,b) ) )
#define Sqrt(a) sqrt(fabs(a))
#define Log(a) log(a)
#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */ 

void tpssfc(double *nu, double *nd,    // I: density
	      double *guu, double *gdd,double *gud,  // I: g=gradient squared=gamma 
	      double *tu, double *td,  // I: tau
	      double *fc   // O: local Ex
/* 	      double *dfcdnu,   // O: local derivative after n */
/* 	      double *dfcdguu, // O: local derivative after g */
/* 	      double *dfxudtau, // O: local derivative after tau */
/* 	      double *d2fxudndg,   // O: local second derivative after n and g */
/* 	      double *d2fxudg2, // O: local second derivative after g */
/* 	      double *d2fxudtaudg, // O: local derivative after tau and g */
/* 	      double *d2fxudndt, // O: local derivative after n and tau */
/*               double *d2fxud2t // O: local derivative after n and tau */
	      ) {

  double n,rtau,zeta,xi,c,rs,g1,g2,g3,fz,EcU,phi,nut,tH,AH,H,EcG,rsup,g1up,g3up;
  double EcUup,nutup,tHup,AHup,ectildeup,rsdn,g1dn,g3dn,EcUdn,nutdn,tHdn,AHdn,ectildedn;
  double Hup,Hdn,EcR,ecterm;
/* Correlation Functional Ec[n,p,t]= Int fc[nu,nd,...] dr*/

  n =  *nd  +  *nu  ;
  rtau = ( *gdd  + 2* *gud  +  *guu )/(8.*n*( *td  +  *tu )) ;
  zeta = (- *nd  +  *nu )/( *nd  +  *nu ) ;
  xi = Sqrt( *guu * Power(*nd,2) - 2.* *gud * *nd * *nu + *gdd*Power(*nu,2))/
   (Power(3,0.3333333333333333)*Power(*nd + *nu,2.3333333333333335)*
    Power(Pi,0.6666666666666666));
  c = (0.53 + 0.87*Power(zeta,2) + 0.5*Power(zeta,4) + 2.26*Power(zeta,6))/
   Power(1 + (Power(xi,2)*(Power(1 - zeta,-1.3333333333333333) + Power(1 + zeta,-1.3333333333333333)))/2.,4) ;
  rs =(Power(1/n,0.3333333333333333)*Power(3/Pi,0.3333333333333333))/Power(2,0.6666666666666666)  ; 
  g1 = -0.0621814*(1 + 0.2137*rs)*Log(1 +  16.081979498692537/(7.5957*Sqrt(rs) + 3.5876*rs + 1.6382*Power(rs,1.5) + 0.49294*Power(rs,2)));
  g2 = 0.0337738*(1 + 0.11125*rs)*Log(1 +   29.608749977793437/ (10.357*Sqrt(rs) + 3.6231*rs + 0.88026*Power(rs,1.5) +  0.49671*Power(rs,2))) ; 
  g3 =0.0621814*(1 + 0.2137*rs)*Log(1 +   16.081979498692537/ (7.5957*Sqrt(rs) + 3.5876*rs + 1.6382*Power(rs,1.5) +  0.49294*Power(rs,2))) -  0.0310907*(1 + 0.20548*rs)*Log(1 +  32.16395899738507/   (14.1189*Sqrt(rs) + 6.1977*rs + 3.3662*Power(rs,1.5) +   0.62517*Power(rs,2))) ; 
  fz = (-2 + Power(1 - zeta,1.3333333333333333) + Power(1 + zeta,1.3333333333333333))/(-2 + 2*Power(2,0.3333333333333333)) ;
  EcU = g1 + fz*g3*Power(zeta,4) + 0.5848223397455204*fz*g2*(1 - Power(zeta,4)) ;
  phi = (Power(1 - zeta,0.6666666666666666) + Power(1 + zeta,0.6666666666666666))/2. ;
  nut = Sqrt( *gdd  + 2* *gud  +  *guu )/n ; 
  tH = ( nut*Power(Pi/3.,0.3333333333333333)*Sqrt(rs))/(2.*Power(2,0.6666666666666666)*phi) ; 
  AH =2.1461263399673642/(-1 + Power(E,-(EcU*Power(Pi,2))/(Power(phi,3)*(1 - Log(2))))) ;
  H = (Power(phi,3)*(1 - Log(2))*Log(1 + 
       (2.1461263399673642*Power(tH,2)*(1 + AH*Power(tH,2)))/
        (1 + AH*Power(tH,2) + Power(AH,2)*Power(tH,4))))/Power(Pi,2) ; 
  EcG =EcU + H  ;
  rsup =  (Power(1/ *nu ,0.3333333333333333)*Power(3/Pi,0.3333333333333333))/Power(2,0.6666666666666666); 
  g1up = -0.0621814*(1 + 0.2137*rsup)*Log(1 + 16.081979498692537/
      (7.5957*Sqrt(rsup) + 3.5876*rsup + 1.6382*Power(rsup,1.5) + 0.49294*Power(rsup,2))) ; 
  g3up = 0.0621814*(1 + 0.2137*rsup)*Log(1 + 16.081979498692537/
       (7.5957*Sqrt(rsup) + 3.5876*rsup + 1.6382*Power(rsup,1.5) + 0.49294*Power(rsup,2))) - 
   0.0310907*(1 + 0.20548*rsup)*Log(1 + 32.16395899738507/
       (14.1189*Sqrt(rsup) + 6.1977*rsup + 3.3662*Power(rsup,1.5) + 0.62517*Power(rsup,2))) ;
  EcUup =g1up + g3up  ;
  nutup =Sqrt( *guu )/ *nu   ; 
  tHup = (nutup*Power(Pi/6.,0.3333333333333333)*Sqrt(rsup))/2. ; 
  AHup = 2.1461263399673642/(-1 + Power(E,(-2*EcUup*Power(Pi,2))/(1 - Log(2)))) ; 
  Hup =  ((1 - Log(2))*Log(1 + (2.1461263399673642*Power(tHup,2)*(1 + AHup*Power(tHup,2)))/(1 + AHup*Power(tHup,2) + Power(AHup,2)*Power(tHup,4))))/
   (2.*Power(Pi,2)) ;
  ectildeup = EcUup + Hup ;
  rsdn = (Power(1/ *nd ,0.3333333333333333)*Power(3/Pi,0.3333333333333333))/Power(2,0.6666666666666666) ; 
  g1dn = -0.0621814*(1 + 0.2137*rsdn)*Log(1 + 16.081979498692537/
      (7.5957*Sqrt(rsdn) + 3.5876*rsdn + 1.6382*Power(rsdn,1.5) + 0.49294*Power(rsdn,2))) ;
  g3dn = 0.0621814*(1 + 0.2137*rsdn)*Log(1 + 16.081979498692537/
       (7.5957*Sqrt(rsdn) + 3.5876*rsdn + 1.6382*Power(rsdn,1.5) + 0.49294*Power(rsdn,2))) - 0.0310907*(1 + 0.20548*rsdn)*Log(1 + 32.16395899738507/  (14.1189*Sqrt(rsdn) + 6.1977*rsdn + 3.3662*Power(rsdn,1.5) + 0.62517*Power(rsdn,2))) ; 
  EcUdn = g1dn + g3dn ; 
  nutdn =Sqrt( *gdd )/ *nd   ; 
  tHdn = (nutdn*Power(Pi/6.,0.3333333333333333)*Sqrt(rsdn))/2. ; 
  AHdn =2.1461263399673642/(-1 + Power(E,(-2*EcUdn*Power(Pi,2))/(1 - Log(2))))  ;

  Hdn = ((1 - Log(2))*Log(1 + (2.1461263399673642*Power(tHdn,2)*(1 + AHdn*Power(tHdn,2)))/(1 + AHdn*Power(tHdn,2) + Power(AHdn,2)*Power(tHdn,4))))/
   (2.*Power(Pi,2)) ;
  ectildedn =EcUdn + Hdn ;
  ecterm = (ectildedn* *nd + ectildeup* *nu)/(*nd + *nu) ;
  EcR = -((1 + c)*ecterm*Power(rtau,2)) + EcG*(1 + c*Power(rtau,2));
  *fc = EcR*n*(1 + 2.8*EcR*Power(rtau,3));

/* Internal variables */


/* Total derivatives */






 // /*Particular values n to 0 and t to 0 */

 /*  if(*n < 1.e-20)) {
    *fxu=0.; 
    *dfxudn=0.;
    *dfxudg=0.;
    *dfxudtau=0.;
    *d2fxudndg=0.;
    *d2fxudg2=0.;
    *d2fxudtaudg=0.;  
  }
  else if(*t < Power(10,-39)){
    *dfxudg=0.;
    *dfxudtau=0.;
    *d2fxudndg=0.;
    *d2fxudg2=0.;
    *d2fxudtaudg=0.;
  }
 */
}
#ifdef __cplusplus
}                               /* extern "C" */
#endif                          /* __cplusplus */

/* $Log: tpss_ec.c,v $
 * Revision 1.2  2006/12/09 09:37:40  miwalter
 * *** empty log message ***
 *
 * Revision 1.1  2005/02/21 14:42:44  miwalter
 * Initial revision
 *
 */
