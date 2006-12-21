/* $Id: tpss_ex.c,v 1.1 2006/11/30 14:37:59 miwalter Exp miwalter $ */
#define Pi 3.1415926535897932384626433832795
#define Power(a,b) ((b==2)?(a)*(a) : ((b==3)?(a)*(a)*(a):pow(a,b)))
#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */ 

  void tpssfxu(double *n,    // I: density
	       double *g,  // I: g=gradient squared=gamma 
	       double *t,  // I: tau
	      double *fxu,   // O: local Ex
	      double *dfxudn,   // O: local derivative after n
	      double *dfxudg, // O: local derivative after g
	      double *dfxudtau, // O: local derivative after tau
	      double *d2fxudndg,   // O: local second derivative after n and g
	      double *d2fxudg2, // O: local second derivative after g
	      double *d2fxudtaudg, // O: local derivative after tau and g
	      double *d2fxudndt, // O: local derivative after n and tau
              double *d2fxud2t // O: local derivative after n and tau
	      ) {
  double exunif,p,z,alpha,q,x,Fx,fxunp;
  double dfxunpdn,dfxunpdexunif,dfxunpdFx,d2fxunpdndFx,d2fxunpdexunifdFx,dexunifdn,dFxdx,d2Fxd2x; 
  double dxdp,dxdz,dxdq,d2xd2p,d2xdpdz,d2xdpdq,d2xd2z,d2xdzdq,d2xd2q;
  double dqdp,d2qd2alpha,dqdalpha,dalphadn,dalphadg,dalphadt,d2alphadndg;
  double dpdn,dpdg,d2pdndg,dzdn,dzdg,dzdt,d2zdndg,d2zdgdt;
  double d2alphadndt,d2zdndt,d2zd2t;

/* Exchange Functional Ex[n,p,t]= Int fxunp[n,g,t] dr*/
  exunif = -3. * Power(3. * Pi * Pi* *n,0.3333333333333333)/(4.*Pi);
//  exunif = (-3 * Power(*n,0.3333333333333333) * Power(3./Pi,0.3333333333333333)) / 4.;
  p = *g / (4. * Power(3,0.6666666666666666) * Power(*n,2.6666666666666665) * Power(Pi,1.3333333333333333));
  z = *g / (8. * *n * *t);
  alpha = (10 * (-1 * *g / (8. * *n) + *t))/(3. * Power(3,0.6666666666666666) * Power(*n,1.6666666666666667) * Power(Pi,1.3333333333333333));
  q = (9. * (-1. + alpha)) / (20. * sqrt(1 + 0.4 * (-1. + alpha) * alpha)) + (2. * p)/3.;  x = (0.018957187845257784 * Power(p,2) + 0.33738687 * Power(p,3) + (146 * Power(q,2))/2025. + 0.11020071474751965 * Power(z,2) - (73 * q * sqrt(Power(p,2) / 2. + (9 * Power(z,2)) / 50.)) / 405. + p * (0.12345679012345678 + (1.59096 * Power(z,2))/(1 + 2 * Power(z,2) + Power(z,4))))/(1 + 2.479516081819192 * p + 1.537 * Power(p,2));
  Fx = 1.804 - 0.804 / (1 + 1.243781094527363 * x);
  *fxu = exunif * Fx * *n;
	
	      
/* Internal variables */

  dfxunpdn=exunif*Fx; /*partial derivative*/
  dfxunpdexunif=Fx* *n ;
  dfxunpdFx=exunif* *n ;
  d2fxunpdndFx=exunif;
  d2fxunpdexunifdFx= *n ;
  dexunifdn=-Power(3/Pi,0.3333333333333333)/(4.*Power( *n ,0.6666666666666666));
  dFxdx=1./Power(1 + 1.243781094527363*x,2);
  d2Fxd2x=-2.487562189054726/Power(1 + 1.243781094527363*x,3);
  dxdp=(0.12345679012345678 + 0.03791437569051557*p + 1.01216061*Power(p,2) - (73*p*q)/(810.*sqrt(Power(p,2)/2. + (9*Power(z,2))/50.)) + 
      (1.59096*Power(z,2))/(1 + 2*Power(z,2) + Power(z,4)))/(1 + 2.479516081819192*p + 1.537*Power(p,2)) - 
   ((2.479516081819192 + 3.074*p)*(0.018957187845257784*Power(p,2) + 0.33738687*Power(p,3) + (146*Power(q,2))/2025. + 0.11020071474751965*Power(z,2) - 
        (73*q*sqrt(Power(p,2)/2. + (9*Power(z,2))/50.))/405. + p*(0.12345679012345678 + (1.59096*Power(z,2))/(1 + 2*Power(z,2) + Power(z,4)))))/
    Power(1 + 2.479516081819192*p + 1.537*Power(p,2),2);
  dxdz=(0.2204014294950393*z - (73*q*z)/(2250.*sqrt(Power(p,2)/2. + (9*Power(z,2))/50.)) + 
     p*((-1.59096*Power(z,2)*(4*z + 4*Power(z,3)))/Power(1 + 2*Power(z,2) + Power(z,4),2) + (3.18192*z)/(1 + 2*Power(z,2) + Power(z,4))))/
   (1 + 2.479516081819192*p + 1.537*Power(p,2));
 dxdq=((292*q)/2025. - (73*sqrt(Power(p,2)/2. + (9*Power(z,2))/50.))/405.)/(1 + 2.479516081819192*p + 1.537*Power(p,2));
 d2xd2p=(0.03791437569051557 + 2.02432122*p + (73*Power(p,2)*q)/(1620.*Power(Power(p,2)/2. + (9*Power(z,2))/50.,1.5)) - 
      (73*q)/(810.*sqrt(Power(p,2)/2. + (9*Power(z,2))/50.)))/(1 + 2.479516081819192*p + 1.537*Power(p,2)) - 
   (2*(2.479516081819192 + 3.074*p)*(0.12345679012345678 + 0.03791437569051557*p + 1.01216061*Power(p,2) - 
        (73*p*q)/(810.*sqrt(Power(p,2)/2. + (9*Power(z,2))/50.)) + (1.59096*Power(z,2))/(1 + 2*Power(z,2) + Power(z,4))))/
    Power(1 + 2.479516081819192*p + 1.537*Power(p,2),2) + (2*Power(2.479516081819192 + 3.074*p,2)*
      (0.018957187845257784*Power(p,2) + 0.33738687*Power(p,3) + (146*Power(q,2))/2025. + 0.11020071474751965*Power(z,2) - 
        (73*q*sqrt(Power(p,2)/2. + (9*Power(z,2))/50.))/405. + p*(0.12345679012345678 + (1.59096*Power(z,2))/(1 + 2*Power(z,2) + Power(z,4)))))/
    Power(1 + 2.479516081819192*p + 1.537*Power(p,2),3) - (3.074*(0.018957187845257784*Power(p,2) + 0.33738687*Power(p,3) + (146*Power(q,2))/2025. + 
        0.11020071474751965*Power(z,2) - (73*q*sqrt(Power(p,2)/2. + (9*Power(z,2))/50.))/405. + 
        p*(0.12345679012345678 + (1.59096*Power(z,2))/(1 + 2*Power(z,2) + Power(z,4)))))/Power(1 + 2.479516081819192*p + 1.537*Power(p,2),2);
 d2xdpdz=((73*p*q*z)/(4500.*Power(Power(p,2)/2. + (9*Power(z,2))/50.,1.5)) - (1.59096*Power(z,2)*(4*z + 4*Power(z,3)))/Power(1 + 2*Power(z,2) + Power(z,4),2) + 
      (3.18192*z)/(1 + 2*Power(z,2) + Power(z,4)))/(1 + 2.479516081819192*p + 1.537*Power(p,2)) - 
   ((2.479516081819192 + 3.074*p)*(0.2204014294950393*z - (73*q*z)/(2250.*sqrt(Power(p,2)/2. + (9*Power(z,2))/50.)) + 
        p*((-1.59096*Power(z,2)*(4*z + 4*Power(z,3)))/Power(1 + 2*Power(z,2) + Power(z,4),2) + (3.18192*z)/(1 + 2*Power(z,2) + Power(z,4)))))/
    Power(1 + 2.479516081819192*p + 1.537*Power(p,2),2);
 d2xdpdq=(-73*p)/(810.*(1 + 2.479516081819192*p + 1.537*Power(p,2))*sqrt(Power(p,2)/2. + (9*Power(z,2))/50.)) - 
   ((2.479516081819192 + 3.074*p)*((292*q)/2025. - (73*sqrt(Power(p,2)/2. + (9*Power(z,2))/50.))/405.))/Power(1 + 2.479516081819192*p + 1.537*Power(p,2),2);
 d2xd2z=(0.2204014294950393 + (73*q*Power(z,2))/(12500.*Power(Power(p,2)/2. + (9*Power(z,2))/50.,1.5)) - (73*q)/(2250.*sqrt(Power(p,2)/2. + (9*Power(z,2))/50.)) + 
     p*((3.18192*Power(z,2)*Power(4*z + 4*Power(z,3),2))/Power(1 + 2*Power(z,2) + Power(z,4),3) - 
        (1.59096*Power(z,2)*(4 + 12*Power(z,2)))/Power(1 + 2*Power(z,2) + Power(z,4),2) - (6.36384*z*(4*z + 4*Power(z,3)))/Power(1 + 2*Power(z,2) + Power(z,4),2) + 
        3.18192/(1 + 2*Power(z,2) + Power(z,4))))/(1 + 2.479516081819192*p + 1.537*Power(p,2));
 d2xdzdq=(-73*z)/(2250.*(1 + 2.479516081819192*p + 1.537*Power(p,2))*sqrt(Power(p,2)/2. + (9*Power(z,2))/50.));
 d2xd2q=292/(2025.*(1 + 2.479516081819192*p + 1.537*Power(p,2)));
 dqdalpha=(-9*(0.4*(-1 + alpha) + 0.4*alpha)*(-1 + alpha))/(40.*Power(1 + 0.4*(-1 + alpha)*alpha,1.5)) + 9/(20.*sqrt(1 + 0.4*(-1 + alpha)*alpha));
 dqdp=0.6666666666666666;
 d2qd2alpha=(27*Power(0.4*(-1 + alpha) + 0.4*alpha,2)*(-1 + alpha))/(80.*Power(1 + 0.4*(-1 + alpha)*alpha,2.5)) - 
   (9*(0.4*(-1 + alpha) + 0.4*alpha))/(20.*Power(1 + 0.4*(-1 + alpha)*alpha,1.5)) - (0.18000000000000002*(-1 + alpha))/Power(1 + 0.4*(-1 + alpha)*alpha,1.5);
 dalphadn=(5* *g )/(12.*Power(3,0.6666666666666666)*Power( *n ,3.6666666666666665)*Power(Pi,1.3333333333333333)) - 
   (50*(-1. * *g /(8.* *n ) +  *t ))/(9.*Power(3,0.6666666666666666)*Power( *n ,2.6666666666666665)*Power(Pi,1.3333333333333333));
 dalphadg=-5/(12.*Power(3,0.6666666666666666)*Power( *n ,2.6666666666666665)*Power(Pi,1.3333333333333333));
 dalphadt=10/(3.*Power(3,0.6666666666666666)*Power( *n ,1.6666666666666667)*Power(Pi,1.3333333333333333));
 d2alphadndg=10/(9.*Power(3,0.6666666666666666)*Power( *n ,3.6666666666666665)*Power(Pi,1.3333333333333333));
 dpdn=(-2* *g )/(3.*Power(3,0.6666666666666666)*Power( *n ,3.6666666666666665)*Power(Pi,1.3333333333333333));
 dpdg=1/(4.*Power(3,0.6666666666666666)*Power( *n ,2.6666666666666665)*Power(Pi,1.3333333333333333));
 d2pdndg=-2/(3.*Power(3,0.6666666666666666)*Power( *n ,3.6666666666666665)*Power(Pi,1.3333333333333333));
 dzdn=-( *g /(Power( *n ,2)* *t *8.));
 dzdg=1/(*n * *t *8.);
 dzdt=-( *g /(8.* *n *Power( *t ,2)));
 d2zdndg =-(1/(Power( *n ,2)* *t *8.));
 d2zdgdt=-(1/(8.* *n *Power( *t ,2)));

 d2alphadndt=-50/(9.*Power(3,0.6666666666666666)*Power(*n,2.6666666666666665)*Power(Pi,1.3333333333333333));
 d2zdndt = *g / (8.*Power(*n,2)*Power(*t,2));
 d2zd2t = *g/(4. * *n *Power(*t,3));
/* Total derivatives */

 *dfxudn = dexunifdn*dfxunpdexunif + dfxunpdn + dFxdx*dfxunpdFx*(dpdn*dxdp + (dalphadn*dqdalpha + dpdn*dqdp)*dxdq + dxdz*dzdn);
 *dfxudg = dFxdx*dfxunpdFx*(dpdg*dxdp + (dalphadg*dqdalpha + dpdg*dqdp)*dxdq + dxdz*dzdg);
 *dfxudtau = dFxdx*dfxunpdFx*(dalphadt*dqdalpha*dxdq + dxdz*dzdt);
 *d2fxudndg = d2fxunpdndFx*dFxdx*(dpdg*dxdp + (dalphadg*dqdalpha + dpdg*dqdp)*dxdq + dxdz*dzdg) + d2fxunpdexunifdFx*dexunifdn*dFxdx*(dpdg*dxdp + (dalphadg*dqdalpha + dpdg*dqdp)*dxdq + dxdz*dzdg) + dfxunpdFx*(d2Fxd2x*(dpdg*dxdp + (dalphadg*dqdalpha + dpdg*dqdp)*dxdq + dxdz*dzdg)*(dpdn*dxdp + (dalphadn*dqdalpha + dpdn*dqdp)*dxdq + dxdz*dzdn) + dFxdx*(d2xd2p*dpdg*dpdn + d2xdpdq*dpdn*(dalphadg*dqdalpha + dpdg*dqdp) + d2xdpdq*dpdg*(dalphadn*dqdalpha + dpdn*dqdp) + d2xd2q*(dalphadg*dqdalpha + dpdg*dqdp)*(dalphadn*dqdalpha + dpdn*dqdp) + d2pdndg*dxdp + (d2qd2alpha*dalphadg*dalphadn + d2alphadndg*dqdalpha + d2pdndg*dqdp)*dxdq + d2zdndg*dxdz + d2xdpdz*dpdn*dzdg + d2xdzdq*(dalphadn*dqdalpha + dpdn*dqdp)*dzdg + d2xdpdz*dpdg*dzdn + d2xdzdq*(dalphadg*dqdalpha + dpdg*dqdp)*dzdn + d2xd2z*dzdg*dzdn));
 *d2fxudg2 = dfxunpdFx*(d2Fxd2x*Power(dpdg*dxdp + (dalphadg*dqdalpha + dpdg*dqdp)*dxdq + dxdz*dzdg,2) + dFxdx*(d2xd2p*Power(dpdg,2) + 2*d2xdpdq*dpdg*(dalphadg*dqdalpha + dpdg*dqdp) + d2xd2q*Power(dalphadg*dqdalpha + dpdg*dqdp,2) + d2qd2alpha*Power(dalphadg,2)*dxdq + 2*d2xdpdz*dpdg*dzdg + 2*d2xdzdq*(dalphadg*dqdalpha + dpdg*dqdp)*dzdg + d2xd2z*Power(dzdg,2)));
 *d2fxudtaudg = dfxunpdFx*(d2Fxd2x*(dpdg*dxdp + (dalphadg*dqdalpha + dpdg*dqdp)*dxdq + dxdz*dzdg)*(dalphadt*dqdalpha*dxdq + dxdz*dzdt) + dFxdx*(d2xdpdq*dalphadt*dpdg*dqdalpha + d2xd2q*dalphadt*dqdalpha*(dalphadg*dqdalpha + dpdg*dqdp) + d2qd2alpha*dalphadg*dalphadt*dxdq + d2zdgdt*dxdz + d2xdzdq*dalphadt*dqdalpha*dzdg + d2xdpdz*dpdg*dzdt + d2xdzdq*(dalphadg*dqdalpha + dpdg*dqdp)*dzdt + d2xd2z*dzdg*dzdt));

 *d2fxudndt = d2fxunpdndFx*dFxdx*(dalphadt*dqdalpha*dxdq + dxdz*dzdt) + d2fxunpdexunifdFx*dexunifdn*dFxdx*(dalphadt*dqdalpha*dxdq + dxdz*dzdt) + dfxunpdFx*(d2Fxd2x*(dpdn*dxdp + (dalphadn*dqdalpha + dpdn*dqdp)*dxdq + dxdz*dzdn)*(dalphadt*dqdalpha*dxdq + dxdz*dzdt) + 
      dFxdx*(d2xdpdq*dalphadt*dpdn*dqdalpha + d2xd2q*dalphadt*dqdalpha*(dalphadn*dqdalpha + dpdn*dqdp) + 
         (d2qd2alpha*dalphadn*dalphadt + d2alphadndt*dqdalpha)*dxdq + d2zdndt*dxdz + d2xdzdq*dalphadt*dqdalpha*dzdn + d2xdpdz*dpdn*dzdt + 
	     d2xdzdq*(dalphadn*dqdalpha + dpdn*dqdp)*dzdt + d2xd2z*dzdn*dzdt));

 *d2fxud2t = dfxunpdFx*(d2Fxd2x*Power(dalphadt*dqdalpha*dxdq + dxdz*dzdt,2) + dFxdx* (d2xd2q*Power(dalphadt,2)*Power(dqdalpha,2) + d2qd2alpha*Power(dalphadt,2)*dxdq + d2zd2t*dxdz + 2*d2xdzdq*dalphadt*dqdalpha*dzdt + d2xd2z*Power(dzdt,2)));






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

/* $Log: tpss_ex.c,v $
 * Revision 1.1  2006/11/30 14:37:59  miwalter
 * Initial revision
 *
 */
