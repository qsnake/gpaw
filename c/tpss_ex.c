#define Pi 3.1415926535897932384626433832795
#define Power(a,b) ((b==2)?(a)*(a) : ((b==3)?(a)*(a)*(a):pow(a,b)))
#include <stdio.h>
#include <math.h>

void tpssfx(double *n,    // I: density
	      double *g,  // I: g=gradient squared=gamma 
	      double *t,  // I: tau
	      double *fxu,   // O: local Ex
	      double *dfxudn,   // O: local derivative after n
	      double *dfxudg, // O: local derivative after g
	      double *dfxudtau, // O: local derivative after tau
	      int iexc        //I:exchange functional, if 7 then alpha=1-z
	      ) 
{
  double exunif,p,z,alpha,q,x,Fx;
  double dfxunpdn,dfxunpdexunif,dfxunpdFx,dexunifdn,dFxdx; 
  double dxdp,dxdz,dxdq;
  double dqdp,dqdalpha,dalphadn,dalphadg,dalphadt;
  double dpdn,dpdg,dzdn,dzdg,dzdt;

/* Exchange Functional Ex[n,p,t]= Int fxunp[n,g,t] dr*/
  exunif = -3. * Power(3. * Pi * Pi* *n,0.3333333333333333)/(4.*Pi);
  z = *g / (8. * *n * *t);
  alpha = (10 * (-1 * *g / (8. * *n) + *t))/(3. * Power(3,0.6666666666666666) * Power(*n,1.6666666666666667) * Power(Pi,1.3333333333333333));
  p = *g / (4. * Power(3,0.6666666666666666) * Power(*n,2.6666666666666665) * Power(Pi,1.3333333333333333));
  //  if (alpha<0) {alpha=0.;z=1.;} /*Treatment in Tao is alpha=fabs(alpha)*/
  if (alpha<0) {alpha=fabs(alpha);}
  if (iexc==7) alpha=1-z;
  q = (9. * (-1. + alpha)) / (20. * sqrt(1 + 0.4 * (-1. + alpha) * alpha)) + (2. * p)/3.;     
  x = (0.018957187845257784 * Power(p,2) + 0.33738687 * Power(p,3) + (146. * Power(q,2))/2025. + 0.11020071474751965 * Power(z,2) - (73. * q * sqrt(Power(p,2) / 2. + (9. * Power(z,2)) / 50.)) / 405. + p * (0.12345679012345678 + (1.59096 * Power(z,2))/(1 + 2 * Power(z,2) + Power(z,4))))/(1 + 2.479516081819192 * p + 1.537 * Power(p,2));
  Fx = 1.804 - 0.804 / (1 + 1.243781094527363 * x);
  *fxu = exunif * Fx * *n;	
  
/* Internal variables */

  dfxunpdn=exunif*Fx; /*partial derivative*/
  dfxunpdexunif=Fx* *n ;
  dfxunpdFx=exunif* *n ;
  dexunifdn=-Power(3/Pi,0.3333333333333333)/(4.*Power( *n ,0.6666666666666666));
  dFxdx=1./Power(1 + 1.243781094527363*x,2);
  dxdp=(0.12345679012345678 + 0.03791437569051557*p + 1.01216061*Power(p,2) - (73*p*q)/(810.*sqrt(Power(p,2)/2. + (9*Power(z,2))/50.)) + 
      (1.59096*Power(z,2))/(1 + 2*Power(z,2) + Power(z,4)))/(1 + 2.479516081819192*p + 1.537*Power(p,2)) - 
   ((2.479516081819192 + 3.074*p)*(0.018957187845257784*Power(p,2) + 0.33738687*Power(p,3) + (146*Power(q,2))/2025. + 0.11020071474751965*Power(z,2) - 
        (73*q*sqrt(Power(p,2)/2. + (9*Power(z,2))/50.))/405. + p*(0.12345679012345678 + (1.59096*Power(z,2))/(1 + 2*Power(z,2) + Power(z,4)))))/
    Power(1 + 2.479516081819192*p + 1.537*Power(p,2),2);
  dxdz=(0.2204014294950393*z - (73*q*z)/(2250.*sqrt(Power(p,2)/2. + (9*Power(z,2))/50.)) + 
     p*((-1.59096*Power(z,2)*(4*z + 4*Power(z,3)))/Power(1 + 2*Power(z,2) + Power(z,4),2) + (3.18192*z)/(1 + 2*Power(z,2) + Power(z,4))))/
   (1 + 2.479516081819192*p + 1.537*Power(p,2));
 dxdq=((292*q)/2025. - (73*sqrt(Power(p,2)/2. + (9*Power(z,2))/50.))/405.)/(1 + 2.479516081819192*p + 1.537*Power(p,2));
 dqdalpha=(-9*(0.4*(-1 + alpha) + 0.4*alpha)*(-1 + alpha))/(40.*Power(1 + 0.4*(-1 + alpha)*alpha,1.5)) + 9/(20.*sqrt(1 + 0.4*(-1 + alpha)*alpha));
 dqdp=0.6666666666666666;
 dqdalpha=0;
 dalphadn=(5* *g )/(12.*Power(3,0.6666666666666666)*Power( *n ,3.6666666666666665)*Power(Pi,1.3333333333333333)) - 
   (50*(-1. * *g /(8.* *n ) +  *t ))/(9.*Power(3,0.6666666666666666)*Power( *n ,2.6666666666666665)*Power(Pi,1.3333333333333333));
 dalphadg=-5/(12.*Power(3,0.6666666666666666)*Power( *n ,2.6666666666666665)*Power(Pi,1.3333333333333333));
 dalphadt=10/(3.*Power(3,0.6666666666666666)*Power( *n ,1.6666666666666667)*Power(Pi,1.3333333333333333));
 dpdn=(-2* *g )/(3.*Power(3,0.6666666666666666)*Power( *n ,3.6666666666666665)*Power(Pi,1.3333333333333333));
 dpdg=1/(4.*Power(3,0.6666666666666666)*Power( *n ,2.6666666666666665)*Power(Pi,1.3333333333333333));
 dzdn=-( *g /(Power( *n ,2)* *t *8.));
 dzdg=1/(*n * *t *8.);
 dzdt=-( *g /(8.* *n *Power( *t ,2)));
 if (iexc==7)
   {dalphadn=-dzdn;dalphadg=-dzdg;dalphadt=-dzdt;}

/* Total derivatives */

 *dfxudn = dexunifdn*dfxunpdexunif + dfxunpdn + dFxdx*dfxunpdFx*(dpdn*dxdp + (dalphadn*dqdalpha + dpdn*dqdp)*dxdq + dxdz*dzdn);
 *dfxudg = dFxdx*dfxunpdFx*(dpdg*dxdp + (dalphadg*dqdalpha + dpdg*dqdp)*dxdq + dxdz*dzdg);
 *dfxudtau = dFxdx*dfxunpdFx*(dalphadt*dqdalpha*dxdq + dxdz*dzdt);

 return ;
}
