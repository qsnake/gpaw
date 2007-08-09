#define Pi 3.1415926535897932384626433832795
#define E 2.7182818284590452353602874713527
#define p3_2o3 2.08008382305190411453005682436
#define p3_1o3 1.44224957030740838232163831078
#define pio3_1o6 1.007715881368979550746625621861
#define pio3_1o3 1.01549129756325926723938600149
#define p2_2o3 1.58740105196819947475170563927
#define p2_1o3 1.25992104989487316476721060728
#define pi_2o3 2.14502939711102560007744410094
#define r1log2 0.306852819440054690582767878542
#define pi2 9.86960440108935861883449099988
#define r3pi_1o3 0.984745021842696541178973376908
#define pio6_1o3 0.805995977008234820358483423320
#define pi_4o3 4.60115111447048996098369283016
#define Power(a,b) ((b==2)?(a)*(a) : ((b==3)?(a)*(a)*(a):( (b==4)?(a)*(a)*(a)*(a) : ( pow(a,b) ))) )
#define Sqrt(a) sqrt(fabs(a))
#define Log(a) log(a)
#include <stdio.h>
#include <math.h>

void tpssfc(double *nu, double *nd,    // I: density
	      double *guu, double *gdd,double *gud,  // I: g=gradient squared=gamma 
	      double *tu, double *td,  // I: tau
	      double *fc,   // O: local Ex
 	      double *dtotfcdnu,   // O: local derivative after n up */
 	      double *dtotfcdnd,   // O: local derivative after n down*/
 	      double *dtotfcdguu, // O: local derivative after g */
 	      double *dtotfcdgdd, // O: local derivative after g */
 	      double *dtotfcdgud, // O: local derivative after g */
 	      double *dtotfcdtu, // O: local derivative after tau */
 	      double *dtotfcdtd // O: local derivative after tau */
	      ) 
{

  double n,rtau,zeta,xi,c,rs,g1,g2,g3,fz,EcU,phi,nut,tH,AH,H,EcG,rsup,g1up,g3up;
  double EcUup,nutup,tHup,AHup,ectildeup,rsdn,g1dn,g3dn,EcUdn,nutdn,tHdn,AHdn,ectildedn;
  double Hup,Hdn,EcR,ecterm;
  double cs,EcUs,phis,tHs,EcUups,EcUdns,tHups,tHdns;
  double dHdnu,dHdphis,dphisdnu,dHdtHs,dtHsdnu,dHdAH,dAHdEcUs,dEcUsdnu, dAHdphis;
  double dHdnd,dphisdnd,dtHsdnd,dEcUsdnd;
  double dHdguu,dtHsdguu,dHdgdd,dtHsdgdd,dHdgud,dtHsdgud;
  double dectildeupdnu,dEcUupsdnu,dHupdtHups,dtHupsdnu,dHupdAHup,dAHupdEcUups;
  double dectildedndnd,dEcUdnsdnd,dHdndtHdns,dtHdnsdnd,dHdndAHdn,dAHdndEcUdns;
  double dectildeupdguu,dtHupsdguu;
  double dectildedndgdd,dtHdnsdgdd;
  double drtaudtu,drtaudtd;
  double dfcdnu,dfcdrtau,drtaudnu,dfcdEcR,dEcRdrtau,dEcRdcs,dcsdnu,dEcRdEcG;
  double dEcRdecterm,dectermdnu,dfcdnd,drtaudnd,dcsdnd,dectermdnd,drtaudguu;
  double drtaudgud;
  double dectermdectildeup,drtaudgdd,dectermdectildedn;
  double dcsdguu,dcsdgdd,dcsdgud;
  double dEcUdzeta,dEcUdg1,dEcUdg2,dEcUdg3,dEcUdfz;
  double dzetadnu,dzetadnd,dg1dn,dg2dn,dg3dn,dfzdz;
  
  double in,in1,in2,in3,in4, in5,in6,in7,sqrtin;
  double inu1,inu2,inu3,inu4, inu5,inu6,sqrtinu;
  double ind1,ind2,ind3,ind4, ind5,ind6,sqrtind;
  double n1;
  double zet1,zet2,zet3,zet4;
/* Correlation Functional Ec[n,p,t]= Int fc[nu,nd,...] dr*/

  n =  *nd  +  *nu  ;
  rtau = ( *gdd  + 2* *gud  +  *guu )/(8.*n*( *td  +  *tu )) ;
  zeta = (- *nd  +  *nu )/( *nd  +  *nu ) ;
  
  in=1./n;
  in1=Power(in,1.1666666666666667);
  in2=Power(in,1.3333333333333333);
  in3=Power(in,1.6666666666666667);
  in4=Power(in,0.16666666666666666);
  in5=Power(in,0.6666666666666666);
  in6=Power(in,0.3333333333333333);
  sqrtin=Sqrt(in);
  in7=Power(in,1.5);
  inu1=Power(1./ *nu,1.1666666666666667);
  inu2=Power(1./ *nu,1.3333333333333333);
  inu3=Power(1./ *nu,1.6666666666666667);
  inu4=Power(1./ *nu,0.16666666666666666);
  inu5=Power(1./ *nu,0.6666666666666666);
  inu6=Power(1./ *nu,0.3333333333333333);
       
  sqrtinu=Sqrt(1./ *nu);
  ind1=Power(1./ *nd,1.1666666666666667);
  ind2=Power(1./ *nd,1.3333333333333333);
  ind3=Power(1./ *nd,1.6666666666666667);
  ind4=Power(1./ *nd,0.16666666666666666);
  ind5=Power(1./ *nd,0.6666666666666666);
  ind6=Power(1./ *nd,0.3333333333333333);
  sqrtind=Sqrt(1./ *nd);

  n1=Power( *nd +  *nu,4.666666666666667);
  zet1=Power(1 - zeta,-1.3333333333333333);
  zet2=Power(1 + zeta,-1.3333333333333333);
  zet3=Power(1 - zeta,0.6666666666666666);
  zet4=Power(1 + zeta,0.6666666666666666);
      
  xi = Sqrt( *guu * Power(*nd,2) - 2.* *gud * *nd * *nu + *gdd*Power(*nu,2))/
   (p3_1o3*Power(n,2.3333333333333335)*
    pi_2o3);
  c = (0.53 + 0.87*Power(zeta,2) + 0.5*Power(zeta,4) + 2.26*Power(zeta,6))/
   Power(1 + (Power(xi,2)*(zet1 + zet2))/2.,4) ;
  rs =(in6*r3pi_1o3)/p2_2o3  ; 
  g1 = -0.0621814*(1 + 0.2137*rs)*Log(1 +  16.081979498692537/(7.5957*Sqrt(rs) + 3.5876*rs + 1.6382*Power(rs,1.5) + 0.49294*Power(rs,2)));
  g2 = 0.0337738*(1 + 0.11125*rs)*Log(1 +   29.608749977793437/ (10.357*Sqrt(rs) + 3.6231*rs + 0.88026*Power(rs,1.5) +  0.49671*Power(rs,2))) ; 
  g3 =0.0621814*(1 + 0.2137*rs)*Log(1 +   16.081979498692537/ (7.5957*Sqrt(rs) + 3.5876*rs + 1.6382*Power(rs,1.5) +  0.49294*Power(rs,2))) -  0.0310907*(1 + 0.20548*rs)*Log(1 +  32.16395899738507/   (14.1189*Sqrt(rs) + 6.1977*rs + 3.3662*Power(rs,1.5) +   0.62517*Power(rs,2))) ; 
  fz = (-2 + 1./zet1 + 1./zet2)/(-2 + 2*p2_1o3) ;
  EcU = g1 + fz*g3*Power(zeta,4) + 0.5848223397455204*fz*g2*(1 - Power(zeta,4)) ;
  phi = (zet3 + zet4)/2. ;
  nut = Sqrt( *gdd  + 2* *gud  +  *guu )/n ; 
  tH = ( nut*pio3_1o3*Sqrt(rs))/(2.*p2_2o3*phi) ; 
  AH =2.1461263399673642/(-1 + Power(E,-(EcU*pi2)/(Power(phi,3)*r1log2))) ;
  H = (Power(phi,3)*r1log2*Log(1 + 
       (2.1461263399673642*Power(tH,2)*(1 + AH*Power(tH,2)))/
        (1 + AH*Power(tH,2) + Power(AH,2)*Power(tH,4))))/pi2 ; 
  EcG =EcU + H  ;
  rsup =  (inu6*r3pi_1o3)/p2_2o3; 
  g1up = -0.0621814*(1 + 0.2137*rsup)*Log(1 + 16.081979498692537/
      (7.5957*Sqrt(rsup) + 3.5876*rsup + 1.6382*Power(rsup,1.5) + 0.49294*Power(rsup,2))) ; 
  g3up = 0.0621814*(1 + 0.2137*rsup)*Log(1 + 16.081979498692537/
       (7.5957*Sqrt(rsup) + 3.5876*rsup + 1.6382*Power(rsup,1.5) + 0.49294*Power(rsup,2))) - 
   0.0310907*(1 + 0.20548*rsup)*Log(1 + 32.16395899738507/
       (14.1189*Sqrt(rsup) + 6.1977*rsup + 3.3662*Power(rsup,1.5) + 0.62517*Power(rsup,2))) ;
  EcUup =g1up + g3up  ;
  nutup =Sqrt( *guu )/ *nu   ; 
  tHup = (nutup*pio6_1o3*Sqrt(rsup))/2. ; 
  AHup = 2.1461263399673642/(-1 + Power(E,(-2*EcUup*pi2)/r1log2)) ; 
  Hup =  (r1log2*Log(1 + (2.1461263399673642*Power(tHup,2)*(1 + AHup*Power(tHup,2)))/(1 + AHup*Power(tHup,2) + Power(AHup,2)*Power(tHup,4))))/
   (2.*pi2) ;
  ectildeup = (EcG > (EcUup + Hup)? EcG : EcUup + Hup)  ;
  rsdn = (ind6*r3pi_1o3)/p2_2o3 ; 
  g1dn = -0.0621814*(1 + 0.2137*rsdn)*Log(1 + 16.081979498692537/
      (7.5957*Sqrt(rsdn) + 3.5876*rsdn + 1.6382*Power(rsdn,1.5) + 0.49294*Power(rsdn,2))) ;
  g3dn = 0.0621814*(1 + 0.2137*rsdn)*Log(1 + 16.081979498692537/
       (7.5957*Sqrt(rsdn) + 3.5876*rsdn + 1.6382*Power(rsdn,1.5) + 0.49294*Power(rsdn,2))) - 0.0310907*(1 + 0.20548*rsdn)*Log(1 + 32.16395899738507/  (14.1189*Sqrt(rsdn) + 6.1977*rsdn + 3.3662*Power(rsdn,1.5) + 0.62517*Power(rsdn,2))) ; 
  EcUdn = g1dn + g3dn ; 
  nutdn =Sqrt( *gdd )/ *nd   ; 
  tHdn = (nutdn*pio6_1o3*Sqrt(rsdn))/2. ; 
  AHdn =2.1461263399673642/(-1 + Power(E,(-2*EcUdn*pi2)/r1log2))  ;

  Hdn = (r1log2*Log(1 + (2.1461263399673642*Power(tHdn,2)*(1 + AHdn*Power(tHdn,2)))/(1 + AHdn*Power(tHdn,2) + Power(AHdn,2)*Power(tHdn,4))))/
   (2.*pi2) ;
  ectildedn = ( EcG > (EcUdn + Hdn)? EcG : EcUdn+Hdn);
  ecterm = (ectildedn* *nd + ectildeup* *nu)/n ;
  EcR = -((1 + c)*ecterm*Power(rtau,2)) + EcG*(1 + c*Power(rtau,2));
  *fc = EcR*n*(1 + 2.8*EcR*Power(rtau,3));

/* Internal variables */

cs=c;
EcUs=EcU;
phis=phi;
tHs=tH;
EcUups=EcUup;
EcUdns=EcUdn;
 tHups=tHup;
 tHdns=tHdn;
dHdphis=(3*Power(phis,2)*r1log2*Log(1 + 
       (2.1461263399673642*Power(tHs,2)*(1 + AH*Power(tHs,2)))/
					  (1 + AH*Power(tHs,2) + Power(AH,2)*Power(tHs,4))))/pi2;
dphisdnu=((2*(zeta/n - in))/
      (3.*Power(1 - zeta,0.3333333333333333)) + 
     (2*(-(zeta/n) + in))/
	  (3.*Power(1 + zeta,0.3333333333333333)))/2.;
dHdtHs=(Power(phis,3)*((-2.1461263399673642*Power(tHs,2)*(1 + AH*Power(tHs,2))*
          (2*AH*tHs + 4*Power(AH,2)*Power(tHs,3)))/
        Power(1 + AH*Power(tHs,2) + Power(AH,2)*Power(tHs,4),2) + 
       (4.2922526799347285*AH*Power(tHs,3))/
        (1 + AH*Power(tHs,2) + Power(AH,2)*Power(tHs,4)) + 
       (4.2922526799347285*tHs*(1 + AH*Power(tHs,2)))/
        (1 + AH*Power(tHs,2) + Power(AH,2)*Power(tHs,4)))*r1log2)/
   (pi2*(1 + (2.1461263399673642*Power(tHs,2)*(1 + AH*Power(tHs,2)))/
		 (1 + AH*Power(tHs,2) + Power(AH,2)*Power(tHs,4))));
dtHsdnu=-(Sqrt(*gdd + 2* *gud + *guu)*in1*
       ((2*(zeta/n - in))/
          (3.*Power(1 - zeta,0.3333333333333333)) + 
         (2*(-(zeta/n) + in))/
          (3.*Power(1 + zeta,0.3333333333333333)))*
       pio3_1o6)/
    (2.*Power(zet3 + 
        zet4,2)) - 
   (7*Sqrt(*gdd + 2* *gud + *guu)*Power(in,2.1666666666666665)*
      pio3_1o6)/
    (12.*(zet3 + 
	  zet4));
dHdAH=(Power(phis,3)*((-2.1461263399673642*Power(tHs,2)*(1 + AH*Power(tHs,2))*
          (Power(tHs,2) + 2*AH*Power(tHs,4)))/
        Power(1 + AH*Power(tHs,2) + Power(AH,2)*Power(tHs,4),2) + 
       (2.1461263399673642*Power(tHs,4))/
        (1 + AH*Power(tHs,2) + Power(AH,2)*Power(tHs,4)))*r1log2)/
   (pi2*(1 + (2.1461263399673642*Power(tHs,2)*(1 + AH*Power(tHs,2)))/
		 (1 + AH*Power(tHs,2) + Power(AH,2)*Power(tHs,4))));
dAHdEcUs=69.02793987321857/
   (Power(E,(EcUs*pi2)/(Power(phis,3)*r1log2))*
     Power(-1 + Power(E,-(EcUs*pi2)/(Power(phis,3)*r1log2)),2)*
    Power(phis,3));

 dEcUdzeta=-2.339289358982082*fz*g2*Power(zeta,3) + 4*fz*g3*Power(zeta,3);
 dEcUdg2=0.5848223397455204*fz*(1 - Power(zeta,4));
 dEcUdg3=fz*Power(zeta,4);
 dEcUdfz=g3*Power(zeta,4) + 0.5848223397455204*g2*(1 - Power(zeta,4));
 dzetadnu=-(zeta/n) + in;
 dzetadnd=-(zeta/n) - in;
dg1dn=(1.*(1 + 0.13256889990520182*in6)*
      (-0.9970917392951799*in1 - 
        0.741856473716896*in2 - 
        0.4002143174996817*in7 - 
        0.12646695504983724*in3))/
    ((1 + 16.081979498692537/
         (5.9825504357710795*in4 + 
           2.2255694211506882*in6 + 
           0.8004286349993635*sqrtin + 
           0.18970043257475588*in5))*
      Power(5.9825504357710795*in4 + 
        2.2255694211506882*in6 + 
        0.8004286349993635*sqrtin + 
        0.518970043257475588*in5,2)) + 
   0.0027477732641884383*in2*
    Log(1 + 16.081979498692537/
       (5.9825504357710795*in4 + 
         2.2255694211506882*in6 + 
         0.8004286349993635*sqrtin + 
	0.18970043257475588*in5));
dg2dn=(-1.*(1 + 0.06901399211255828*in6)*
      (-1.35956911724794*in1 - 
        0.7491972878592055*in2 - 
        0.21504862356383217*in7 - 
        0.1274341730084892*in3))/
    ((1 + 29.608749977793437/
         (8.15741470348764*in4 + 
           2.2475918635776164*in6 + 
           0.43009724712766434*sqrtin + 
           0.1911512595127338*in5))*
      Power(8.15741470348764*in4 + 
        2.2475918635776164*in6 + 
        0.43009724712766434*sqrtin + 
        0.1911512595127338*in5,2)) - 
   0.0007769549222703736*in2*
    Log(1 + 29.608749977793437/
       (8.15741470348764*in4 + 
         2.2475918635776164*in6 + 
         0.43009724712766434*sqrtin + 
	0.1911512595127338*in5));
dg3dn=(1.*(1 + 0.12746961887000874*in6)*
      (-1.8533958105157808*in1 - 
        1.2815820791490706*in2 - 
        0.8223668877838045*in7 - 
        0.16039141941921278*in3))/
    ((1 + 32.16395899738507/
         (11.120374863094685*in4 + 
           3.8447462374472123*in6 + 
           1.644733775567609*sqrtin + 
           0.2405871291288192*in5))*
      Power(11.120374863094685*in4 + 
        3.8447462374472123*in6 + 
        1.644733775567609*sqrtin + 
        0.2405871291288192*in5,2)) - 
   (0.9999999999999999*(1 + 0.13256889990520182*in6)*
      (-0.9970917392951799*in1 - 
        0.741856473716896*in2 - 
        0.4002143174996817*in7 - 
        0.12646695504983724*in3))/
    ((1 + 16.081824322151103/
         (5.9825504357710795*in4 + 
           2.2255694211506882*in6 + 
           0.8004286349993635*sqrtin + 
           0.18970043257475588*in5))*
      Power(5.9825504357710795*in4 + 
        2.2255694211506882*in6 + 
        0.8004286349993635*sqrtin + 
        0.18970043257475588*in5,2)) - 
   0.0027477997779684197*in2*
    Log(1 + 16.081824322151103/
       (5.9825504357710795*in4 + 
         2.2255694211506882*in6 + 
         0.8004286349993635*sqrtin + 
         0.18970043257475588*in5)) + 
   0.0013210398931339266*in2*
    Log(1 + 32.16395899738507/
       (11.120374863094685*in4 + 
         3.8447462374472123*in6 + 
         1.644733775567609*sqrtin + 
	0.2405871291288192*in5));
dfzdz=((-4*Power(1 - zeta,0.3333333333333333))/3. + 
       (4*Power(1 + zeta,0.3333333333333333))/3.)/(-2 + 2*p2_1o3);

dEcUsdnu= dEcUdzeta*dzetadnu + dg1dn + dEcUdg2*dg2dn + dEcUdg3*dg3dn + 
      dEcUdfz*dfzdz*dzetadnu;
dEcUsdnd=dEcUdzeta*dzetadnd + dg1dn + dEcUdg2*dg2dn + dEcUdg3*dg3dn + 
    dEcUdfz*dfzdz*dzetadnd;

dAHdphis=(-207.08381961965574*EcUs)/
   (Power(E,(EcUs*pi2)/(Power(phis,3)*r1log2))*
     Power(-1 + Power(E,-(EcUs*pi2)/(Power(phis,3)*r1log2)),2)*
    Power(phis,4));
dphisdnd=((2*(zeta/n + in))/
      (3.*Power(1 - zeta,0.3333333333333333)) + 
     (2*(-(zeta/n) - in))/
	  (3.*Power(1 + zeta,0.3333333333333333)))/2.;
dtHsdnd=-(Sqrt(*gdd + 2* *gud + *guu)*in1*
       ((2*(zeta/n + in))/
          (3.*Power(1 - zeta,0.3333333333333333)) + 
         (2*(-(zeta/n) - in))/
          (3.*Power(1 + zeta,0.3333333333333333)))*
       pio3_1o6)/
    (2.*Power(zet3 + 
        zet4,2)) - 
   (7*Sqrt(*gdd + 2* *gud + *guu)*Power(in,2.1666666666666665)*
      pio3_1o6)/
    (12.*(zet3 + 
	  zet4));

dHdtHs=(Power(phis,3)*((-2.1461263399673642*Power(tHs,2)*(1 + AH*Power(tHs,2))*
          (2*AH*tHs + 4*Power(AH,2)*Power(tHs,3)))/
        Power(1 + AH*Power(tHs,2) + Power(AH,2)*Power(tHs,4),2) + 
       (4.2922526799347285*AH*Power(tHs,3))/
        (1 + AH*Power(tHs,2) + Power(AH,2)*Power(tHs,4)) + 
       (4.2922526799347285*tHs*(1 + AH*Power(tHs,2)))/
        (1 + AH*Power(tHs,2) + Power(AH,2)*Power(tHs,4)))*r1log2)/
   (pi2*(1 + (2.1461263399673642*Power(tHs,2)*(1 + AH*Power(tHs,2)))/
		 (1 + AH*Power(tHs,2) + Power(AH,2)*Power(tHs,4))));
dtHsdguu=(in1*pio3_1o6)/
   (4.*Sqrt(*gdd + 2* *gud + *guu)*(zet3 + zet4));
dtHsdgdd=(in1*pio3_1o6)/
   (4.*Sqrt(*gdd + 2* *gud + *guu)*(zet3 + zet4));
dtHsdgud=(in1*pio3_1o6)/
   (2.*Sqrt(*gdd + 2* *gud + *guu)*(zet3 + zet4));
dEcUupsdnu = (1.*(1 + 0.12746961887000874*inu6)*
  (-1.8533958105157808*inu1 - 
        1.2815820791490706*inu2 - 
        0.8223668877838045*Power(1/ *nu,1.5) - 
        0.16039141941921278*inu3))/
    ((1 + 32.16395899738507/
         (11.120374863094685*inu4 + 
           3.8447462374472123*inu6 + 
           1.644733775567609*sqrtinu + 
           0.2405871291288192*inu5))*
      Power(11.120374863094685*inu4 + 
        3.8447462374472123*inu6 + 
        1.644733775567609*sqrtinu + 
        0.2405871291288192*inu5,2)) + 
   (0.*(1 + 0.13256889990520182*inu6)*
      (-0.9970917392951799*inu1 - 
        0.741856473716896*inu2 - 
        0.4002143174996817*Power(1/ *nu,1.5) - 
        0.12646695504983724*inu3))/
    ((1 + 16.081979498692537/
         (5.9825504357710795*inu4 + 
           2.2255694211506882*inu6 + 
           0.8004286349993635*sqrtinu + 
           0.18970043257475588*inu5))*
      Power(5.9825504357710795*inu4 + 
        2.2255694211506882*inu6 + 
        0.8004286349993635*sqrtinu + 
        0.18970043257475588*inu5,2)) + 
   0.*inu2*
    Log(1 + 16.081979498692537/
       (5.9825504357710795*inu4 + 
         2.2255694211506882*inu6 + 
         0.8004286349993635*sqrtinu + 
         0.18970043257475588*inu5)) + 
   0.0013210398931339266*inu2*
    Log(1 + 32.16395899738507/
       (11.120374863094685*inu4 + 
         3.8447462374472123*inu6 + 
         1.644733775567609*sqrtinu + 
	0.2405871291288192*inu5));
dHupdtHups=(((-2.1461263399673642*Power(tHups,2)*(1 + AHup*Power(tHups,2))*
          (2*AHup*tHups + 4*Power(AHup,2)*Power(tHups,3)))/
        Power(1 + AHup*Power(tHups,2) + Power(AHup,2)*Power(tHups,4),2) + 
       (4.2922526799347285*AHup*Power(tHups,3))/
        (1 + AHup*Power(tHups,2) + Power(AHup,2)*Power(tHups,4)) + 
       (4.2922526799347285*tHups*(1 + AHup*Power(tHups,2)))/
        (1 + AHup*Power(tHups,2) + Power(AHup,2)*Power(tHups,4)))*r1log2)/
   (2.*pi2*(1 + (2.1461263399673642*Power(tHups,2)*
          (1 + AHup*Power(tHups,2)))/
		    (1 + AHup*Power(tHups,2) + Power(AHup,2)*Power(tHups,4))));
dtHupsdnu=(-7*Sqrt(*guu)*Power(1/ *nu,2.1666666666666665)*pio3_1o6)/
  (12.*p2_2o3);
dHupdAHup=(((-2.1461263399673642*Power(tHups,2)*(1 + AHup*Power(tHups,2))*
          (Power(tHups,2) + 2*AHup*Power(tHups,4)))/
        Power(1 + AHup*Power(tHups,2) + Power(AHup,2)*Power(tHups,4),2) + 
       (2.1461263399673642*Power(tHups,4))/
        (1 + AHup*Power(tHups,2) + Power(AHup,2)*Power(tHups,4)))*r1log2)/
   (2.*pi2*(1 + (2.1461263399673642*Power(tHups,2)*
          (1 + AHup*Power(tHups,2)))/
		    (1 + AHup*Power(tHups,2) + Power(AHup,2)*Power(tHups,4))));
dAHupdEcUups=138.05587974643714/
   (Power(E,(2*EcUups*pi2)/r1log2)*
    Power(-1 + Power(E,(-2*EcUups*pi2)/r1log2),2));

dEcUdnsdnd=(1.*(1 + 0.12746961887000874*ind6)*
      (-1.8533958105157808*ind1 - 
        1.2815820791490706*ind2 - 
        0.8223668877838045*Power(1/ *nd,1.5) - 
        0.16039141941921278*ind3))/
    ((1 + 32.16395899738507/
         (11.120374863094685*ind4 + 
           3.8447462374472123*ind6 + 
           1.644733775567609*sqrtind + 
           0.2405871291288192*ind5))*
      Power(11.120374863094685*ind4 + 
        3.8447462374472123*ind6 + 
        1.644733775567609*sqrtind + 
        0.2405871291288192*ind5,2)) + 
   (0.*(1 + 0.13256889990520182*ind6)*
      (-0.9970917392951799*ind1 - 
        0.741856473716896*ind2 - 
        0.4002143174996817*Power(1/ *nd,1.5) - 
        0.12646695504983724*ind3))/
    ((1 + 16.081979498692537/
         (5.9825504357710795*ind4 + 
           2.2255694211506882*ind6 + 
           0.8004286349993635*sqrtind + 
           0.18970043257475588*ind5))*
      Power(5.9825504357710795*ind4 + 
        2.2255694211506882*ind6 + 
        0.8004286349993635*sqrtind + 
        0.18970043257475588*ind5,2)) + 
   0.*ind2*
    Log(1 + 16.081979498692537/
       (5.9825504357710795*ind4 + 
         2.2255694211506882*ind6 + 
         0.8004286349993635*sqrtind + 
         0.18970043257475588*ind5)) + 
   0.0013210398931339266*ind2*
    Log(1 + 32.16395899738507/
       (11.120374863094685*ind4 + 
         3.8447462374472123*ind6 + 
         1.644733775567609*sqrtind + 
	0.2405871291288192*ind5));
dHdndtHdns=(((-2.1461263399673642*Power(tHdns,2)*(1 + AHdn*Power(tHdns,2))*
          (2*AHdn*tHdns + 4*Power(AHdn,2)*Power(tHdns,3)))/
        Power(1 + AHdn*Power(tHdns,2) + Power(AHdn,2)*Power(tHdns,4),2) + 
       (4.2922526799347285*AHdn*Power(tHdns,3))/
        (1 + AHdn*Power(tHdns,2) + Power(AHdn,2)*Power(tHdns,4)) + 
       (4.2922526799347285*tHdns*(1 + AHdn*Power(tHdns,2)))/
        (1 + AHdn*Power(tHdns,2) + Power(AHdn,2)*Power(tHdns,4)))*r1log2)/
   (2.*pi2*(1 + (2.1461263399673642*Power(tHdns,2)*
          (1 + AHdn*Power(tHdns,2)))/
		    (1 + AHdn*Power(tHdns,2) + Power(AHdn,2)*Power(tHdns,4))));
dtHdnsdnd=(-7*Sqrt(*gdd)*Power(1/ *nd,2.1666666666666665)*pio3_1o6)/
  (12.*p2_2o3);
dHdndAHdn=(((-2.1461263399673642*Power(tHdns,2)*(1 + AHdn*Power(tHdns,2))*
          (Power(tHdns,2) + 2*AHdn*Power(tHdns,4)))/
        Power(1 + AHdn*Power(tHdns,2) + Power(AHdn,2)*Power(tHdns,4),2) + 
       (2.1461263399673642*Power(tHdns,4))/
        (1 + AHdn*Power(tHdns,2) + Power(AHdn,2)*Power(tHdns,4)))*r1log2)/
   (2.*pi2*(1 + (2.1461263399673642*Power(tHdns,2)*
          (1 + AHdn*Power(tHdns,2)))/
		    (1 + AHdn*Power(tHdns,2) + Power(AHdn,2)*Power(tHdns,4))));
dAHdndEcUdns=138.05587974643714/
   (Power(E,(2*EcUdns*pi2)/r1log2)*
    Power(-1 + Power(E,(-2*EcUdns*pi2)/r1log2),2));

dtHupsdguu=(inu1*pio3_1o6)/
  (4.*p2_2o3*Sqrt(*guu));
dtHdnsdgdd=(ind1*pio3_1o6)/
  (4.*p2_2o3*Sqrt(*gdd));
 drtaudtu=-(*gdd + 2* *gud + *guu)/(8.*n*Power(*td + *tu,2));
 drtaudtd=-(*gdd + 2* *gud + *guu)/(8.*n*Power(*td + *tu,2));

 dHdnu = dHdphis*dphisdnu + dHdtHs*dtHsdnu + 
      dHdAH*(dAHdEcUs*dEcUsdnu + dAHdphis*dphisdnu);
 dHdnd = dHdphis*dphisdnd + dHdtHs*dtHsdnd + 
      dHdAH*(dAHdEcUs*dEcUsdnd + dAHdphis*dphisdnd);
 dHdguu = dHdtHs*dtHsdguu;
 dHdgdd = dHdtHs*dtHsdgdd;
 dHdgud = dHdtHs*dtHsdgud;
 dectildeupdnu = 
    dEcUupsdnu + (dHupdtHups*dtHupsdnu + dHupdAHup*dAHupdEcUups*dEcUupsdnu);
   dectildeupdguu = dHupdtHups*dtHupsdguu;
 dectildedndnd = 
    dEcUdnsdnd + (dHdndtHdns*dtHdnsdnd + dHdndAHdn*dAHdndEcUdns*dEcUdnsdnd);

 dectildedndgdd = dHdndtHdns*dtHdnsdgdd;

 dfcdnu=EcR*(1 + 2.8*EcR*Power(rtau,3)) ;
 dfcdrtau= 8.399999999999999*Power(EcR,2)*n*Power(rtau,2);
 drtaudnu= -(*gdd + 2.* *gud + *guu)/(8.*Power(n,2)*(*td + *tu));
 dfcdEcR= 2.8*EcR*n*Power(rtau,3) + n*(1 + 2.8*EcR*Power(rtau,3));
 dEcRdrtau=2*cs*EcG*rtau - 2*(1 + cs)*ecterm*rtau ;
 dEcRdcs=EcG*Power(rtau,2) - ecterm*Power(rtau,2) ;
dcsdnu= (-4*(0.53 + (2.26*Power(- *nd +  *nu,6))/Power( *nd +  *nu,6) + 
        (0.5*Power(- *nd +  *nu,4))/Power( *nd +  *nu,4) + 
        (0.87*Power(- *nd +  *nu,2))/Power( *nd +  *nu,2))*
      ((( *guu*Power( *nd,2) - 2* *gud* *nd* *nu + *gdd *Power( *nu,2))*
           ((-4*((- *nd +  *nu)/Power( *nd +  *nu,2) - 1/n))/
              (3.*Power(1 - zeta,2.3333333333333335)) - 
             (4*(-((- *nd +  *nu)/Power( *nd +  *nu,2)) + 1/n))/
              (3.*Power(1 + zeta,2.3333333333333335))))/
         (2.*p3_2o3*n1*
           pi_4o3) + 
        ((-2* *gud * *nd + 2* *gdd * *nu)*(zet1 + 
             zet2))/
         (2.*p3_2o3*n1*
           pi_4o3) - 
        (7*( *guu*Power( *nd,2) - 2* *gud * *nd * *nu + *gdd * Power( *nu,2))*
           (zet1 + 
             zet2))/
         (3.*p3_2o3*Power( *nd +  *nu,5.666666666666667)*
           pi_4o3)))/
    Power(1 + (( *guu * Power( *nd,2) - 2 * *gud * *nd * *nu + *gdd * Power( *nu,2))*
         (zet1 + 
           zet2))/
       (2.*p3_2o3*n1*
         pi_4o3),5) + 
   ((-13.559999999999999*Power(- *nd +  *nu,6))/Power( *nd +  *nu,7) + 
      (13.559999999999999*Power(- *nd +  *nu,5))/Power( *nd +  *nu,6) - 
      (2.*Power(- *nd +  *nu,4))/Power( *nd +  *nu,5) + 
      (2.*Power(- *nd +  *nu,3))/Power( *nd +  *nu,4) - 
      (1.74*Power(- *nd +  *nu,2))/Power( *nd +  *nu,3) + (1.74*(- *nd +  *nu))/Power( *nd +  *nu,2)
      )/Power(1 + (( *guu * Power( *nd,2) - 2 * *gud * *nd * *nu + *gdd * Power( *nu,2))*
         (zet1 + 
           zet2))/
       (2.*p3_2o3*n1*
	pi_4o3),4);
 dEcRdEcG= 1 + cs*Power(rtau,2);
 dEcRdecterm= -((1 + cs)*Power(rtau,2)) ;
 dectermdnu= ectildeup/n - (ectildedn* *nd + ectildeup* *nu)/Power( *nd +  *nu,2);
 dfcdnd= EcR*(1 + 2.8*EcR*Power(rtau,3));
 drtaudnd= -(*gdd + 2* *gud +  *guu)/(8.*Power( *nd +  *nu,2)*( *td +  *tu));
dcsdnd= (-4*(0.53 + (2.26*Power(- *nd +  *nu,6))/Power( *nd +  *nu,6) + 
        (0.5*Power(- *nd +  *nu,4))/Power( *nd +  *nu,4) + 
        (0.87*Power(- *nd +  *nu,2))/Power( *nd +  *nu,2))*
      ((( *guu * Power( *nd,2) - 2* *gud* *nd * *nu + *gdd * Power( *nu,2))*
           ((-4*((- *nd +  *nu)/Power( *nd +  *nu,2) + 1/n))/
              (3.*Power(1 - zeta,2.3333333333333335)) - 
             (4*(-((- *nd +  *nu)/Power( *nd +  *nu,2)) - 1/n))/
              (3.*Power(1 + zeta,2.3333333333333335))))/
         (2.*p3_2o3*n1*
           pi_4o3) +  ((2* *guu* *nd - 2* *gud* *nu)*(zet1 + zet2))/
         (2.*p3_2o3*n1 * pi_4o3) - 
        (7*( *guu * Power( *nd,2) - 2* *gud * *nd * *nu + *gdd * Power( *nu,2))*
           (zet1 + zet2))/
         (3.*p3_2o3*Power( *nd +  *nu,5.666666666666667)*
           pi_4o3)))/
    Power(1 + (( *guu * Power( *nd,2) - 2* *gud* *nd * *nu + *gdd * Power( *nu,2))*
         (zet1 + 
           zet2))/
       (2.*p3_2o3*n1*
         pi_4o3),5) + 
   ((-13.559999999999999*Power(- *nd +  *nu,6))/Power( *nd +  *nu,7) - 
      (13.559999999999999*Power(- *nd +  *nu,5))/Power( *nd +  *nu,6) - 
      (2.*Power(- *nd +  *nu,4))/Power( *nd +  *nu,5) - 
      (2.*Power(- *nd +  *nu,3))/Power( *nd +  *nu,4) - 
      (1.74*Power(- *nd +  *nu,2))/Power( *nd +  *nu,3) - (1.74*(- *nd +  *nu))/Power( *nd +  *nu,2)
      )/Power(1 + (( *guu * Power( *nd,2) - 2* *gud * *nd* *nu + *gdd * Power( *nu,2))*
         (zet1 + 
           zet2))/
       (2.*p3_2o3*n1*
	pi_4o3),4);
 dectermdnd= ectildedn/n - (ectildedn* *nd + ectildeup* *nu)/Power( *nd +  *nu,2);
 drtaudguu=1/(8.*n*( *td +  *tu));
 drtaudgud=1/(4.*n*(*td + *tu));
 dectermdectildeup=  *nu/n;
 drtaudgdd= 1/(8.*n*( *td +  *tu));
 dectermdectildedn= *nd/n;
 dcsdguu=(-2*Power( *nd,2)*(0.53 + (2.26*Power(- *nd +  *nu,6))/Power( *nd +  *nu,6) + 
       (0.5*Power(- *nd +  *nu,4))/Power( *nd +  *nu,4) + 
       (0.87*Power(- *nd +  *nu,2))/Power( *nd +  *nu,2))*
     (zet1 + 
       zet2))/
   (p3_2o3*n1*
     Power(1 + ((*guu * Power( *nd,2) - 2* *gud * *nd* *nu + *gdd * Power( *nu,2))*
          (zet1 + 
            zet2))/
        (2.*p3_2o3*n1*
	 pi_4o3),5)*pi_4o3);
 dcsdgdd=(-2*Power( *nu,2)*(0.53 + (2.26*Power(- *nd +  *nu,6))/Power( *nd +  *nu,6) + 
       (0.5*Power(- *nd +  *nu,4))/Power( *nd +  *nu,4) + 
       (0.87*Power(- *nd +  *nu,2))/Power( *nd +  *nu,2))*
     (zet1 + 
       zet2))/
   (p3_2o3*n1*
     Power(1 + ((*guu * Power( *nd,2) - 2* *gud* *nd* *nu + *gdd * Power( *nu,2))*
          (zet1 + 
            zet2))/
        (2.*p3_2o3*n1*
	 pi_4o3),5)*pi_4o3);
 dcsdgud=(4* *nd* *nu*(0.53 + (2.26*Power(- *nd +  *nu,6))/Power( *nd +  *nu,6) + 
       (0.5*Power(- *nd +  *nu,4))/Power( *nd +  *nu,4) + 
       (0.87*Power(- *nd +  *nu,2))/Power( *nd +  *nu,2))*
     (zet1 + 
       zet2))/
   (p3_2o3*n1*
     Power(1 + ((*guu*Power( *nd,2) - 2* *gud* *nd* *nu + *gdd * Power( *nu,2))*
          (zet1 + 
            zet2))/
        (2.*p3_2o3*n1*
	 pi_4o3),5)*pi_4o3);
 
/* Total derivatives */
 *dtotfcdnu = dfcdnu + dfcdrtau*drtaudnu + dfcdEcR*(dEcRdrtau*drtaudnu + dEcRdcs*dcsdnu + dEcRdEcG*(dEcUsdnu + dHdnu) +  dEcRdecterm*(dectermdnu + dectermdectildeup*dectildeupdnu));
  
 *dtotfcdnd =  dfcdnd + dfcdrtau*drtaudnd + dfcdEcR*(dEcRdrtau*drtaudnd + dEcRdcs*dcsdnd + dEcRdEcG*(dEcUsdnd + dHdnd) +   dEcRdecterm*(dectermdnd + dectermdectildedn*dectildedndnd));

  *dtotfcdguu = dfcdrtau*drtaudguu  + dfcdEcR*(dEcRdrtau*drtaudguu + dEcRdcs*dcsdguu+dEcRdEcG*dHdtHs*dtHsdguu +  dEcRdecterm*dectermdectildeup*dectildeupdguu);
 
 *dtotfcdgdd = dfcdrtau*drtaudgdd + dfcdEcR*(dEcRdrtau*drtaudgdd + dEcRdcs*dcsdgdd+dEcRdEcG*dHdtHs*dtHsdgdd + dEcRdecterm*dectermdectildedn*dectildedndgdd);

 *dtotfcdgud = dfcdrtau*drtaudgud+dfcdEcR*(dEcRdrtau*drtaudgud+dEcRdcs*dcsdgud+dEcRdEcG*dHdtHs*dtHsdgud);

 *dtotfcdtu = dfcdrtau*drtaudtu + dfcdEcR*dEcRdrtau*drtaudtu;

 *dtotfcdtd = dfcdrtau*drtaudtd + dfcdEcR*dEcRdrtau*drtaudtd;

 if ( EcG > (EcUup + Hup)){
   *dtotfcdnu = dfcdnu + dfcdrtau*drtaudnu + dfcdEcR*(dEcRdrtau*drtaudnu + dEcRdcs*dcsdnu + dEcRdEcG*(dEcUsdnu + dHdnu) +  dEcRdecterm*(dectermdnu + dectermdectildeup*(dEcUsdnu + dHdnu)));
   *dtotfcdguu = dfcdrtau*drtaudguu  + dfcdEcR*(dEcRdrtau*drtaudguu + dEcRdcs*dcsdguu+dEcRdEcG*dHdtHs*dtHsdguu +  dEcRdecterm*dectermdectildeup*dHdtHs*dtHsdguu);
   *dtotfcdgud = *dtotfcdgud+dfcdEcR*(dEcRdecterm*dectermdectildeup*dHdtHs*dtHsdgud); 
     };

 if ( EcG > (EcUdn + Hdn)){
   *dtotfcdnd =  dfcdnd + dfcdrtau*drtaudnd + dfcdEcR*(dEcRdrtau*drtaudnd + dEcRdcs*dcsdnd + dEcRdEcG*(dEcUsdnd + dHdnd) +   dEcRdecterm*(dectermdnd + dectermdectildedn*(dEcUsdnd + dHdnd)));
   *dtotfcdgdd = dfcdrtau*drtaudgdd + dfcdEcR*(dEcRdrtau*drtaudgdd + dEcRdcs*dcsdgdd+dEcRdEcG*dHdtHs*dtHsdgdd + dEcRdecterm*dectermdectildedn*dHdtHs*dtHsdgdd);
   *dtotfcdgud = *dtotfcdgud+dfcdEcR*(dEcRdecterm*dectermdectildedn*dHdtHs*dtHsdgud); 
     };

 return ;
}
