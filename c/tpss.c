#include <Python.h>
#define NO_IMPORT_ARRAY
#include <Numeric/arrayobject.h>
#include "xc.h"
#include "extensions.h"

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
	     );

typedef struct 
{
  PyObject_HEAD
  double (*exchange)(const xc_parameters* par,
		     double n, double rs, double a2,
		     double* dedrs, double* deda2);
  double (*correlation)(double n, double rs, double zeta, double a2, 
			bool gga, bool spinpol,
			double* dedrs, double* dedzeta, double* deda2);
  xc_parameters par;
} MGGAFunctionalObject;

static void MGGAFunctional_dealloc(MGGAFunctionalObject *self)
{
  PyObject_DEL(self);
}

static PyObject* 
MGGAFunctional_CalculateSpinPaired(MGGAFunctionalObject *self, PyObject *args)
{
  PyArrayObject* e_array;
  PyArrayObject* n_array;
  PyArrayObject* v_array;
  PyArrayObject* a2_array = 0;
  PyArrayObject* deda2_array = 0;
  PyArrayObject* tau_array;
  printf("<MGGAFunctional_CalculateSpinPaired>\n");
  if (!PyArg_ParseTuple(args, "OOOOOO", &e_array, &n_array, &v_array,
			&a2_array, &deda2_array, &tau_array))
    return NULL;

  int ng = e_array->dimensions[0];
  const xc_parameters* par = &self->par;

  double* e_g = DOUBLEP(e_array);
  const double* n_g = DOUBLEP(n_array);
  const double* a2_g = DOUBLEP(a2_array); 
  const double* tau_g = DOUBLEP(n_array);
  double* v_g = DOUBLEP(v_array);

  double* deda2_g = 0;
/*   if (par->gga) */
/*     { */
/*       a2_g = DOUBLEP(a2_array); */
/*       deda2_g = DOUBLEP(deda2_array); */
/*     } */
  double fxu,dfxudn,dfxudg,dfxudtau,d2fxudndg,
    d2fxudg2,d2fxudtaudg,d2fxudndt,d2fxud2t;

  for (int i = 0; i < ng; i++)
    {
/*       printf("<MGGAFunctional_CalculateSpinPaired> i=%d\n",i); */
      double n = n_g[i];
      if (n < NMIN) n = 0;
      double g = a2_g[i];
      if (g < NMIN) g = 0;
      double tau = tau_g[i];
      if (tau < NMIN) tau = 0;

      /* exchange */
      tpssfxu(&n,&g,&tau,&fxu,&dfxudn,&dfxudg,&dfxudtau,&d2fxudndg,
	      &d2fxudg2,&d2fxudtaudg,&d2fxudndt,&d2fxud2t);
      e_g[i] = fxu;
      
      /* correlation */

    }
  Py_RETURN_NONE;
}

static PyObject* 
MGGAFunctional_CalculateSpinPolarized(MGGAFunctionalObject *self, 
				      PyObject *args)
{
  PyArrayObject* e;
  PyArrayObject* na;
  PyArrayObject* va;
  PyArrayObject* nb;
  PyArrayObject* vb;
  PyArrayObject* a2 = 0;
  PyArrayObject* aa2 = 0;
  PyArrayObject* ab2 = 0;
  PyArrayObject* deda2 = 0;
  PyArrayObject* dedaa2 = 0;
  PyArrayObject* dedab2 = 0;
  if (!PyArg_ParseTuple(args, "OOOOO|OOOOOO", &e, &na, &va, &nb, &vb,
                        &a2, &aa2, &ab2, &deda2, &dedaa2, &dedab2))
    return NULL;

  int ng = e->dimensions[0];
  printf("<MGGAFunctional_CalculateSpinPolarized> ng=%d\n",ng);
  double* e_g = DOUBLEP(e);
  const double* na_g = DOUBLEP(na);
  double* va_g = DOUBLEP(va);
  const double* nb_g = DOUBLEP(nb);
  double* vb_g = DOUBLEP(vb);

  const double* a2_g = 0;
  const double* aa2_g = 0;
  const double* ab2_g = 0;
  double* deda2_g = 0;
  double* dedaa2_g = 0;
  double* dedab2_g = 0;
  const xc_parameters* par = &self->par;
  if (par->gga)
    {
      a2_g = DOUBLEP(a2);
      aa2_g = DOUBLEP(aa2);
      ab2_g = DOUBLEP(ab2);
      deda2_g = DOUBLEP(deda2);
      dedaa2_g = DOUBLEP(dedaa2);
      dedab2_g = DOUBLEP(dedab2);
    }

  for (int g = 0; g < ng; g++)
    {
      double na = 2.0 * na_g[g];
      if (na < NMIN)
        na = NMIN;
      double rsa = pow(C0I / na, THIRD);
      double nb = 2.0 * nb_g[g];
      if (nb < NMIN)
        nb = NMIN;
      double rsb = pow(C0I / nb, THIRD);
      double n = 0.5 * (na + nb);
      double rs = pow(C0I / n, THIRD);
      double zeta = 0.5 * (na - nb) / n;
      double dexadrs;
      double dexada2;
      double exa;
      double dexbdrs;
      double dexbda2;
      double exb;
      double decdrs;
      double decdzeta;
      double decda2;
      double ec;
      if (par->gga)
        {
          exa = self->exchange(par, na, rsa, 4.0 * aa2_g[g],
			       &dexadrs, &dexada2);
          exb = self->exchange(par, nb, rsb, 4.0 * ab2_g[g],
			       &dexbdrs, &dexbda2);
          ec = self->correlation(n, rs, zeta, a2_g[g], 1, 1, 
				 &decdrs, &decdzeta, &decda2);
          dedaa2_g[g] = na * dexada2;
          dedab2_g[g] = nb * dexbda2;
          deda2_g[g] = n * decda2;
        }
      else
        {
          exa = self->exchange(par, na, rsa, 0.0, &dexadrs, 0);
          exb = self->exchange(par, nb, rsb, 0.0, &dexbdrs, 0);
          ec = self->correlation(n, rs, zeta, 0.0, 0, 1, 
				 &decdrs, &decdzeta, 0);
        }
      e_g[g] = 0.5 * (na * exa + nb * exb) + n * ec;
      va_g[g] += (exa + ec - (rsa * dexadrs + rs * decdrs) / 3.0 -
                  (zeta - 1.0) * decdzeta);
      vb_g[g] += (exb + ec - (rsb * dexbdrs + rs * decdrs) / 3.0 -
                  (zeta + 1.0) * decdzeta);
    }
  Py_RETURN_NONE;
}

static PyObject* 
MGGAFunctional_exchange(MGGAFunctionalObject *self, PyObject *args)
{
  double rs;
  double a2;
  if (!PyArg_ParseTuple(args, "dd", &rs, &a2))
    return NULL;

  double dedrs;
  double deda2;
  double n = 1.0 / (C0 * rs * rs * rs);
  double ex = self->exchange(&self->par, n, rs, a2, &dedrs, &deda2);
  return Py_BuildValue("ddd", ex, dedrs, deda2); 
}

static PyObject* 
MGGAFunctional_correlation(MGGAFunctionalObject *self, PyObject *args)
{
  double rs;
  double zeta;
  double a2;
  if (!PyArg_ParseTuple(args, "ddd", &rs, &zeta, &a2)) 
    return NULL;

  double dedrs;
  double dedzeta;
  double deda2;
  double n = 1.0 / (C0 * rs * rs * rs);
  double ec = self->correlation(n, rs, zeta, a2, self->par.gga, 1,
				&dedrs, &dedzeta, &deda2);
  return Py_BuildValue("dddd", ec, dedrs, dedzeta, deda2); 
}

static PyMethodDef MGGAFunctional_Methods[] = {
    {"calculate_spinpaired", 
     (PyCFunction)MGGAFunctional_CalculateSpinPaired, METH_VARARGS, 0},
    {"calculate_spinpolarized", 
     (PyCFunction)MGGAFunctional_CalculateSpinPolarized, METH_VARARGS, 0},
    {"exchange", (PyCFunction)MGGAFunctional_exchange, METH_VARARGS, 0},
    {"correlation", (PyCFunction)MGGAFunctional_correlation, METH_VARARGS, 0},
    {NULL, NULL, 0, NULL}
};

static PyObject* MGGAFunctional_getattr(PyObject *obj, char *name)
{
    return Py_FindMethod(MGGAFunctional_Methods, obj, name);
}

static PyTypeObject MGGAFunctionalType = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,
  "MGGAFunctional",
  sizeof(MGGAFunctionalObject),
  0,
  (destructor)MGGAFunctional_dealloc,
  0,
  MGGAFunctional_getattr
};

PyObject * NewMGGAFunctionalObject(PyObject *obj, PyObject *args)
{
  int type;
  if (!PyArg_ParseTuple(args, "i", &type))
    return NULL;
  printf("<NewMGGAFunctionalObject> type=%d\n", type);

  MGGAFunctionalObject *self = PyObject_NEW(MGGAFunctionalObject,
					    &MGGAFunctionalType);

  if (self == NULL)
    return NULL;

  return (PyObject*)self;
}


/* ******************************************************************
   original code by Tao translated via f2c follows
   ****************************************************************** */

/* tpss_org.f -- translated by f2c (version 20000817).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include <math.h>
#include "f2c.h"

/* Table of constant values */

static doublereal c_b2 = 29.608813203268074;
static doublereal c_b3 = .66666666666666663;
static doublereal c_b4 = 1.6666666666666665;
static doublereal c_b7 = 2.6666666666666665;
static doublereal c_b8 = 2.;
static doublereal c_b14 = 3.;
static doublereal c_b18 = 4.;
static doublereal c_b19 = 6.;
static doublereal c_b22 = .33333333333333333333333333333;
static doublereal c_b32 = 1.3333333333333333;
static doublereal c_b40 = 1.8505508252042546;
static doublereal c_b41 = .33333333333333331;
static doublereal c_b44 = .0310907;
static doublereal c_b45 = .2137;
static doublereal c_b46 = 7.5957;
static doublereal c_b47 = 3.5876;
static doublereal c_b48 = 1.6382;
static doublereal c_b49 = .49294;
static doublereal c_b50 = 1.;
static doublereal c_b51 = .01554535;
static doublereal c_b52 = .20548;
static doublereal c_b53 = 14.1189;
static doublereal c_b54 = 6.1977;
static doublereal c_b55 = 3.3662;
static doublereal c_b56 = .62517;
static doublereal c_b58 = .0168869;
static doublereal c_b59 = .11125;
static doublereal c_b60 = 10.357;
static doublereal c_b61 = 3.6231;
static doublereal c_b62 = .88026;
static doublereal c_b63 = .49671;

/*<       subroutine FX(rho,gr,tau,FXTPSS) >*/
/* Subroutine */ int fx_(rho, gr, tau, fxtpss)
doublereal *rho, *gr, *tau, *fxtpss;
{
    /* System generated locals */
    doublereal d__1;

    /* Builtin functions */
    double pow_dd(), sqrt();

    /* Local variables */
    static doublereal ptil, qtil, tauw, xtil, effec, p, pbemu, d1, d2, hh, xx,
	     xx2, alp, dmu, alpterm, tau0;

/*<       implicit double precision (a-h,o-z) >*/
/*<       parameter ( pi = 3.1415926535897932384626433832795d0 ) >*/
/*<       parameter ( THRD = 0.33333333333333333333333333333333D0 ) >*/
/*<       parameter ( ee = 1.537d0 ) >*/
/*<       parameter ( cc = 0.39774d0 ) >*/
/*<       parameter ( dk = 0.804d0 ) >*/
/*<       parameter ( bb = 0.40d0 ) >*/
/* gr = the magnitude of the gradient of the electron density */
/* tau = the kinetic energy density */
/* FXTPSS = the enhancement factor of the spin-unpolarized density */
/*<       tau0 = 0.3d0*(3.d0*pi*pi)**(2.d0*THRD)*rho**(5.d0*THRD) >*/
    tau0 = pow_dd(&c_b2, &c_b3) * .3 * pow_dd(rho, &c_b4);
/*<       tauw = 0.125d0*gr*gr/rho >*/
    tauw = *gr * .125 * *gr / *rho;
/*<       p = gr*gr/( 4.d0*(3.d0*pi*pi)**(2.d0/3.d0)*rho**(8.d0/3.d0)) >*/
    p = *gr * *gr / (pow_dd(&c_b2, &c_b3) * 4. * pow_dd(rho, &c_b7));
/*<       alp = (tau - tauw)/tau0 >*/
    alp = (*tau - tauw) / tau0;
/*<       alp = dabs(alp) >*/
    alp = fabs(alp);
/*<    >*/
    alpterm = (alp - 1.) * .45000000000000001 / sqrt(alp * .4 * (alp - 1.) + 
	    1.);
/*<       qtil = alpterm + 2.d0*p/3.d0 >*/
    qtil = alpterm + p * 2. / 3.;
/*<       xx = tauw/tau >*/
    xx = tauw / *tau;
/*<       xx2 = xx*xx >*/
    xx2 = xx * xx;
/*<       dmu = 10.d0/81.d0 >*/
    dmu = .12345679012345678;
/*<       pbemu = 0.21951d0 >*/
    pbemu = .21951;
/*<       d1 = 146.d0/2025.d0 >*/
    d1 = .072098765432098769;
/*<       d2 = -73.d0/405.d0 >*/
    d2 = -.18024691358024691;
/*<       effec = cc*4.d0*xx2/(1.d0 + xx2)**2.d0 >*/
    d__1 = xx2 + 1.;
    effec = xx2 * 1.5909599999999999 / pow_dd(&d__1, &c_b8);
/*<       hh = 2.d0*dsqrt(ee)*dmu*(3.d0*xx/5.d0)**2.d0 >*/
    d__1 = xx * 3. / 5.;
    hh = sqrt(1.537) * 2. * dmu * pow_dd(&d__1, &c_b8);
/*<       ptil = dsqrt(0.5*(0.6d0*xx)**2.d0 + 0.5d0*p**2.d0) >*/
    d__1 = xx * .6;
    ptil = sqrt(pow_dd(&d__1, &c_b8) * (float).5 + pow_dd(&p, &c_b8) * .5);
/*<    >*/
    d__1 = sqrt(1.537) * p + 1.;
    xtil = ((dmu + effec) * p + d1 * qtil * qtil + d2 * qtil * ptil + pow_dd(&
	    dmu, &c_b8) / .804 * pow_dd(&p, &c_b8) + hh + pbemu * 1.537 * 
	    pow_dd(&p, &c_b14)) / pow_dd(&d__1, &c_b8);
/*<       FXTPSS = 1.d0 + dk - dk/(1.d0 + xtil/dk) >*/
    *fxtpss = 1.804 - .804 / (xtil / .804 + 1.);
/*<       return >*/
    return 0;
/*<       end >*/
} /* fx_ */

/* ******************************************************** */
/*<       subroutine sicpbe(up,dn,grup,grdn,gr,tauup,taudn,grupdn,ectpss) >*/
/* Subroutine */ int sicpbe_(up, dn, grup, grdn, gr, tauup, taudn, grupdn, 
	ectpss)
doublereal *up, *dn, *grup, *grdn, *gr, *tauup, *taudn, *grupdn, *ectpss;
{
    /* System generated locals */
    doublereal d__1, d__2;

    /* Builtin functions */
    double pow_dd(), sqrt();

    /* Local variables */
    static doublereal delc, tfac, dens, zeta, tauw, sumz, term1, term2, ecpbe,
	     pbedn, pbeup, aa, bb, cc, dd, szeta2, gg, hh, en, sk, rs, tt;
    extern /* Subroutine */ int corpbe_();
    static doublereal revsic, twoksg, xx2, xx3, ccc, ecc, pbe, fer, tau, grz, 
	    yyz, ter1, ter2, ter3;

/*<       implicit real*8 (a-h,o-z) >*/
/* grupdn = del n_up dot del n_down */
/* ectpss = TPSS meta-GGA correlation energy per electron */
/* grup = sqrt( grada(1)**2 + grada(2)**2 + grada(3)**2 ) */
/* grdn = sqrt( gradb(1)**2 + gradb(2)**2 + gradb(3)**2 ) */
/* gr   = sqrt( grad(1)**2 + grad(2)**2 + grad(3)**2 ) */
/* grupdn = grada(1)*gradb(1) + grada(2)*gradb(2) + grada(3)*gradb(3) */
/*<       parameter ( PI = 3.1415926535897932384626433832795d0 ) >*/
/*<       parameter ( THRD = 0.33333333333333333333333333333d0 ) >*/
/*<       parameter ( TWOTHRD = 2.0d0*thrd) >*/
/*<       parameter ( FAC = 3.09366772628013593097d0 ) >*/
/*      FAC = (3*PI**2)**(1/3) */
/*<       parameter ( ALPHA = 1.91915829267751300662482032624669d0 ) >*/
/*<       parameter ( GGPOL = 0.793700526d0) >*/
/*     GGPOL = 0.5*(2**TWOTHRD) */
/*<       aa = 0.87d0 >*/
    aa = .87;
/*<       bb = 0.5d0 >*/
    bb = .5;
/*<       cc = 2.26d0 >*/
    cc = 2.26;
/*<       dd = 2.8d0 >*/
    dd = 2.8;
/*<       dens = up + dn >*/
    dens = *up + *dn;
/*<       if (dens.gt.1.0d-18) then >*/
    if (dens > 1e-18) {
/*<          zeta = (up - dn)/dens >*/
	zeta = (*up - *dn) / dens;
/*<          CCC = 0.53d0 + aa*zeta**2.d0 + bb*zeta**4.d0 + cc*zeta**6.d0 >*/
	ccc = aa * pow_dd(&zeta, &c_b8) + .53 + bb * pow_dd(&zeta, &c_b18) + 
		cc * pow_dd(&zeta, &c_b19);
/*<          gg = 0.5d0*((1.0d0+zeta)**TWOTHRD + (1.0d0-zeta)**TWOTHRD) >*/
	d__1 = zeta + 1.;
	d__2 = 1. - zeta;
	gg = (pow_dd(&d__1, &c_b3) + pow_dd(&d__2, &c_b3)) * .5;
/*<          fer = FAC*dens**THRD >*/
	fer = pow_dd(&dens, &c_b22) * 3.09366772628013593097;
/*<          rs = ALPHA/fer >*/
	rs = 1.91915829267751300662482032624669 / fer;
/*<          sk = dsqrt(4.0d0*fer/PI) >*/
	sk = sqrt(fer * 4. / 3.1415926535897932384626433832795);
/*<          twoksg = 2.0d0*sk*gg >*/
	twoksg = sk * 2. * gg;
/*<          tt = gr/(twoksg*dens) >*/
	tt = *gr / (twoksg * dens);
/*<          call CORPBE(RS,ZETA,TT,ECC,HH,ECPBE) >*/
	corpbe_(&rs, &zeta, &tt, &ecc, &hh, &ecpbe);
/*<          tauw = 0.125d0*gr**2.d0/dens >*/
	tauw = pow_dd(gr, &c_b8) * .125 / dens;
/*<          tau = tauup + taudn >*/
	tau = *tauup + *taudn;
/*<          xx2 = (tauw/tau)**2.d0 >*/
	d__1 = tauw / tau;
	xx2 = pow_dd(&d__1, &c_b8);
/*<          xx3 = (tauw/tau)**3.d0 >*/
	d__1 = tauw / tau;
	xx3 = pow_dd(&d__1, &c_b14);
/* ---------------------------- */
/*<          ter1 = (1.d0 - zeta)**2.d0*grup**2.d0 >*/
	d__1 = 1. - zeta;
	ter1 = pow_dd(&d__1, &c_b8) * pow_dd(grup, &c_b8);
/*<          ter2 = (1.d0 + zeta)**2.d0*grdn**2.d0 >*/
	d__1 = zeta + 1.;
	ter2 = pow_dd(&d__1, &c_b8) * pow_dd(grdn, &c_b8);
/*<          ter3 = 2.d0*(1.d0 - zeta**2.d0)*grupdn >*/
	ter3 = (1. - pow_dd(&zeta, &c_b8)) * 2. * *grupdn;
/*<          sumz = ter1 + ter2 - ter3 >*/
	sumz = ter1 + ter2 - ter3;
/*<          sumz = dabs(sumz) >*/
	sumz = fabs(sumz);
/*<          grz = dsqrt(sumz)/dens >*/
	grz = sqrt(sumz) / dens;
/*<          szeta2 = (grz/(2.d0*fer))**2.d0 >*/
	d__1 = grz / (fer * 2.);
	szeta2 = pow_dd(&d__1, &c_b8);
/*<          if (dabs(zeta).lt.0.9999d0) then >*/
	if (fabs(zeta) < .9999) {
/*<             term1 = 1.d0/(1.d0 + zeta)**(4.d0/3.d0) >*/
	    d__1 = zeta + 1.;
	    term1 = 1. / pow_dd(&d__1, &c_b32);
/*<             term2 = 1.d0/(1.d0 - zeta)**(4.d0/3.d0)  >*/
	    d__1 = 1. - zeta;
	    term2 = 1. / pow_dd(&d__1, &c_b32);
/*<             yyz = szeta2*(term1 + term2)/2.d0 >*/
	    yyz = szeta2 * (term1 + term2) / 2.;
/*<          else >*/
	} else {
/*<             yyz = 0.d0 >*/
	    yyz = 0.;
/*<          end if >*/
	}
/*<          CCC = CCC/(1.d0 + yyz)**4.d0 >*/
	d__1 = yyz + 1.;
	ccc /= pow_dd(&d__1, &c_b18);
/*<          pbe = ecc + hh >*/
	pbe = ecc + hh;
/*<          en = pbe*(1.d0 + CCC*xx2) >*/
	en = pbe * (ccc * xx2 + 1.);
/*<          delc = 0.0d0 >*/
	delc = 0.;
/*        spin up correction */
/*<          if (up.gt.1.0d-18) then >*/
	if (*up > 1e-18) {
/*<             zeta = 1.0d0 >*/
	    zeta = 1.;
/*<             fer = FAC*up**THRD >*/
	    fer = pow_dd(up, &c_b22) * 3.09366772628013593097;
/*<             rs = ALPHA/fer >*/
	    rs = 1.91915829267751300662482032624669 / fer;
/*<             sk = dsqrt(4.0d0*fer/PI) >*/
	    sk = sqrt(fer * 4. / 3.1415926535897932384626433832795);
/*<             twoksg = 2.0d0*sk*GGPOL >*/
	    twoksg = sk * 2. * .793700526;
/*<             tt = dabs(grup)/(twoksg*up) >*/
	    tt = fabs(*grup) / (twoksg * *up);
/*<             call CORPBE(RS,ZETA,TT,ECC,HH,ECPBE) >*/
	    corpbe_(&rs, &zeta, &tt, &ecc, &hh, &ecpbe);
/*<             tfac = (tauw/tau)**2.d0 >*/
	    d__1 = tauw / tau;
	    tfac = pow_dd(&d__1, &c_b8);
/*<             pbeup = ecc + hh >*/
	    pbeup = ecc + hh;
/*<             if(pbeup.lt.pbe) then >*/
	    if (pbeup < pbe) {
/*<               delc = delc + tfac*up*pbe >*/
		delc += tfac * *up * pbe;
/*<             else >*/
	    } else {
/*<               delc = delc + tfac*up*pbeup >*/
		delc += tfac * *up * pbeup;
/*<             end if >*/
	    }
/*<          endif >*/
	}
/*        spin down correction */
/*<          if (dn.gt.1.0d-18) then >*/
	if (*dn > 1e-18) {
/*<             zeta = -1.0d0 >*/
	    zeta = -1.;
/*<             fer = FAC*dn**THRD >*/
	    fer = pow_dd(dn, &c_b22) * 3.09366772628013593097;
/*<             rs = ALPHA/fer >*/
	    rs = 1.91915829267751300662482032624669 / fer;
/*<             sk = dsqrt(4.0d0*fer/PI) >*/
	    sk = sqrt(fer * 4. / 3.1415926535897932384626433832795);
/*<             twoksg = 2.0d0*sk*GGPOL >*/
	    twoksg = sk * 2. * .793700526;
/*<             tt = dabs(grdn)/(twoksg*dn) >*/
	    tt = fabs(*grdn) / (twoksg * *dn);
/*<             call CORPBE(RS,ZETA,TT,ECC,HH,ECPBE) >*/
	    corpbe_(&rs, &zeta, &tt, &ecc, &hh, &ecpbe);
/*<             tfac = (tauw/tau)**2.d0 >*/
	    d__1 = tauw / tau;
	    tfac = pow_dd(&d__1, &c_b8);
/*<             pbedn = ecc + hh >*/
	    pbedn = ecc + hh;
/*<             if(pbedn.lt.pbe) then >*/
	    if (pbedn < pbe) {
/*<               delc = delc + tfac*dn*pbe >*/
		delc += tfac * *dn * pbe;
/*<             else >*/
	    } else {
/*<               delc = delc + tfac*dn*pbedn >*/
		delc += tfac * *dn * pbedn;
/*<             end if >*/
	    }
/*<          endif >*/
	}
/*<          delc = -(1.d0 + CCC)*delc/dens >*/
	delc = -(ccc + 1.) * delc / dens;
/*<          revsic = en + delc >*/
	revsic = en + delc;
/*<          ectpss = revsic*(1.d0 + dd*revsic*xx3) >*/
	*ectpss = revsic * (dd * revsic * xx3 + 1.);
/*<       else >*/
    } else {
/*<          ectpss = 0.0d0 >*/
	*ectpss = 0.;
/*<       endif >*/
    }
/*<       return >*/
    return 0;
/*<       end >*/
} /* sicpbe_ */

/* ******************************************** */
/*<       SUBROUTINE CORPBE(RS,ZETA,TT,ECC,HH,ECPBE) >*/
/* Subroutine */ int corpbe_(rs, zeta, tt, ecc, hh, ecpbe)
doublereal *rs, *zeta, *tt, *ecc, *hh, *ecpbe;
{
    /* System generated locals */
    doublereal d__1, d__2;

    /* Builtin functions */
    double pow_dd(), exp(), log();

    /* Local variables */
    static doublereal alfc, alfm;
    extern /* Subroutine */ int gcor_();
    static doublereal ec0rs, ec1rs, b, f, b2, q4, q5, z4, cc, gg, alfmrs, ec0,
	     ec1, gg3, rs2, rs3, tt2, tt3, tt4, pon;

/*  INPUT: RS=SEITZ RADIUS=(3/4pi rho)^(1/3) */
/*       : ZETA=RELATIVE SPIN POLARIZATION = (rhoup-rhodn)/rho */
/*       : tt=ABS(GRAD rho)/(rho*2.*KS*GG)  -- only needed for PBE */
/*       : HH,ecgga,vcgga */
/* ---------------------------------------------------------------------- */
/*<       IMPLICIT REAL*8 (A-H,O-Z) >*/
/* numbers for use in LSD energy spin-interpolation formula. */
/*      GAM= 2^(4/3)-2 */
/*      FZZ=f''(0)= 8/(9*GAM) */
/* numbers for construction of PBE */
/*      gamma=(1-log(2))/pi^2 */
/*      betamb=coefficient in gradient expansion for correlation. */
/*<       parameter( pi = 3.1415926535897932384626433832795d0 ) >*/
/*<       parameter(thrd=1.d0/3.d0,thrdm=-thrd,thrd2=2.d0*thrd) >*/
/*<       parameter(sixthm=thrdm/2.d0) >*/
/*<       parameter(thrd4=4.d0*thrd) >*/
/*<       parameter(GAM=0.5198420997897463295344212145565d0) >*/
/*<       parameter(fzz=8.d0/(9.d0*GAM)) >*/
/*<       parameter(gamma=0.03109069086965489503494086371273d0) >*/
/*<       parameter(betamb=0.06672455060314922d0,DELT=betamb/gamma) >*/
/*<       parameter ( FAC = 3.09366772628013593097d0 ) >*/
/*     FAC = (3*PI**2)**(1/3) */
/*<       parameter ( ALPHA = 1.91915829267751300662482032624669d0 ) >*/
/*<       cc = (3.0d0*pi*pi/16.0d0)**thrd >*/
    cc = pow_dd(&c_b40, &c_b41);
/* EC0=unpolarized LSD correlation energy */
/* EC0RS=dEC0/drs */
/* EC1=fully polarized LSD correlation energy */
/* EC1RS=dEC1/drs */
/* ALFM=-spin stiffness */
/* ALFMRS=-dalphac/drs */
/* F=spin-scaling factor */
/* construct LSD ec */
/*<       F = ((1.D0+ZETA)**THRD4+(1.D0-ZETA)**THRD4-2.D0)/GAM >*/
    d__1 = *zeta + 1.;
    d__2 = 1. - *zeta;
    f = (pow_dd(&d__1, &c_b32) + pow_dd(&d__2, &c_b32) - 2.) / 
	    .5198420997897463295344212145565;
/*<    >*/
    gcor_(&c_b44, &c_b45, &c_b46, &c_b47, &c_b48, &c_b49, &c_b50, rs, &ec0, &
	    ec0rs);
/*<    >*/
    gcor_(&c_b51, &c_b52, &c_b53, &c_b54, &c_b55, &c_b56, &c_b50, rs, &ec1, &
	    ec1rs);
/*<    >*/
    gcor_(&c_b58, &c_b59, &c_b60, &c_b61, &c_b62, &c_b63, &c_b50, rs, &alfm, &
	    alfmrs);
/*  ALFM IS MINUS THE SPIN STIFFNESS ALFC */
/*<       ALFC = -ALFM >*/
    alfc = -alfm;
/*<       Z4 = ZETA**4.0d0 >*/
    z4 = pow_dd(zeta, &c_b18);
/*<       ECC = EC0*(1.D0-F*Z4)+EC1*F*Z4-ALFM*F*(1.D0-Z4)/FZZ >*/
    *ecc = ec0 * (1. - f * z4) + ec1 * f * z4 - alfm * f * (1. - z4) / 
	    1.7099209341613653;
/* ---------------------------------------------------------------------- */
/* PBE correlation energy */
/* GG=phi(zeta) */
/* DELT=betamb/gamma */
/* B=A */
/*<       gg = 0.5d0*((1.0d0+zeta)**thrd2 + (1.0d0-zeta)**thrd2) >*/
    d__1 = *zeta + 1.;
    d__2 = 1. - *zeta;
    gg = (pow_dd(&d__1, &c_b3) + pow_dd(&d__2, &c_b3)) * .5;
/*<       GG3 = GG**3.0d0 >*/
    gg3 = pow_dd(&gg, &c_b14);
/*<       PON=-ECC/(GG3*gamma) >*/
    pon = -(*ecc) / (gg3 * .03109069086965489503494086371273);
/*<       B = DELT/(DEXP(PON)-1.D0) >*/
    b = 2.1461263399673647 / (exp(pon) - 1.);
/*<       B2 = B*B >*/
    b2 = b * b;
/*<       TT2 = TT*TT >*/
    tt2 = *tt * *tt;
/*<       TT3 = TT2*TT >*/
    tt3 = tt2 * *tt;
/*<       TT4 = TT2*TT2 >*/
    tt4 = tt2 * tt2;
/*<       RS2 = RS*RS >*/
    rs2 = *rs * *rs;
/*<       RS3 = RS2*RS >*/
    rs3 = rs2 * *rs;
/*<       Q4 = 1.D0+B*TT2 >*/
    q4 = b * tt2 + 1.;
/*<       Q5 = 1.D0+B*TT2+B2*TT4 >*/
    q5 = b * tt2 + 1. + b2 * tt4;
/*<       HH = GG3*gamma*DLOG(1.D0+DELT*TT2*Q4/Q5) >*/
    *hh = gg3 * .03109069086965489503494086371273 * log(tt2 * 
	    2.1461263399673647 * q4 / q5 + 1.);
/*<       ECPBE = ECC + HH >*/
    *ecpbe = *ecc + *hh;
/*<       return >*/
    return 0;
/*<       end >*/
} /* corpbe_ */

/* ------------------------------------------------------------------- */
/*<       SUBROUTINE GCOR(A,A1,B1,B2,B3,B4,P,RS,GG,GGRS) >*/
/* Subroutine */ int gcor_(a, a1, b1, b2, b3, b4, p, rs, gg, ggrs)
doublereal *a, *a1, *b1, *b2, *b3, *b4, *p, *rs, *gg, *ggrs;
{
    /* Builtin functions */
    double sqrt(), pow_dd(), log();

    /* Local variables */
    static doublereal p1, q0, q1, q2, q3, rs12, rs32, rsp;

/*<       IMPLICIT REAL*8 (A-H,O-Z) >*/
/*<       P1 = P + 1.D0 >*/
    p1 = *p + 1.;
/*<       Q0 = -2.D0*A*(1.D0+A1*RS) >*/
    q0 = *a * -2. * (*a1 * *rs + 1.);
/*<       RS12 = DSQRT(RS) >*/
    rs12 = sqrt(*rs);
/*<       RS32 = RS12**3.0d0 >*/
    rs32 = pow_dd(&rs12, &c_b14);
/*<       RSP = RS**P >*/
    rsp = pow_dd(rs, p);
/*<       Q1 = 2.D0*A*(B1*RS12+B2*RS+B3*RS32+B4*RS*RSP) >*/
    q1 = *a * 2. * (*b1 * rs12 + *b2 * *rs + *b3 * rs32 + *b4 * *rs * rsp);
/*<       Q2 = DLOG(1.D0+1.D0/Q1) >*/
    q2 = log(1. / q1 + 1.);
/*<       GG = Q0*Q2 >*/
    *gg = q0 * q2;
/*<       Q3 = A*(B1/RS12+2.D0*B2+3.D0*B3*RS12+2.D0*B4*P1*RSP) >*/
    q3 = *a * (*b1 / rs12 + *b2 * 2. + *b3 * 3. * rs12 + *b4 * 2. * p1 * rsp);
/*<       GGRS = -2.D0*A*A1*Q2-Q0*Q3/(Q1**2.0d0+Q1) >*/
    *ggrs = *a * -2. * *a1 * q2 - q0 * q3 / (pow_dd(&q1, &c_b8) + q1);
/*<       RETURN >*/
    return 0;
/*<       END >*/
} /* gcor_ */

