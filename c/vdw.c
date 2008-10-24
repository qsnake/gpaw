#include "extensions.h"

double vdwkernel(double r, double q01, double q02, int nD, int ndelta,
		 double dD, double ddelta,
		 const double (*phi)[nD])
{
  const double C = -1024.0 / 243.0 * M_PI * M_PI * M_PI * M_PI;
  double d1 = r * q01;
  double d2 = r * q02;
  double D = 0.5 * (d1 + d2);
  double xD = D / dD;
  int jD = (int)xD;
  double e12;
  if (jD >= nD - 1)
    {
      double d12 = d1 * d1;
      double d22 = d2 * d2;
      e12 = C / (d12 * d22 * (d12 + d22));
    }
  else
    {
      double xdelta = fabs(0.5 * (d1 - d2) / D) / ddelta;
      int jdelta = (int)xdelta;
      double a;
      if (jdelta >= ndelta - 1)
	{
	  jdelta = ndelta - 2;
	  a = 1.0;
	}
      else
	a = xdelta - jdelta;
      double b = xD - jD;
      e12 = ((a         * b         * phi[jdelta + 1][jD + 1] +
	      a         * (1.0 - b) * phi[jdelta + 1][jD    ] +
	      (1.0 - a) * b         * phi[jdelta    ][jD + 1] +
	      (1.0 - a) * (1.0 - b) * phi[jdelta    ][jD    ]) /
	     (D * D));
    }
  return e12;
}

PyObject * vdw(PyObject* self, PyObject *args)
{
  const PyArrayObject* n_obj;
  const PyArrayObject* q0_obj;
  const PyArrayObject* R_obj;
  const PyArrayObject* cell_obj;
  const PyArrayObject* pbc_obj;
  const PyArrayObject* repeat_obj;
  const PyArrayObject* phi_obj;
  double dD;
  double ddelta;
  int iA;
  int iB;
  PyArrayObject* histogram_obj;
  double rcut;
  if (!PyArg_ParseTuple(args, "OOOOOOOddiiOd", &n_obj, &q0_obj, &R_obj,
			&cell_obj, &pbc_obj, &repeat_obj,
			&phi_obj, &dD, &ddelta, &iA, &iB,
			&histogram_obj, &rcut))
    return NULL;

  int nD = phi_obj->dimensions[1];
  int ndelta = phi_obj->dimensions[1];
  const double* n = (const double*)DOUBLEP(n_obj);
  const double* q0 = (const double*)DOUBLEP(q0_obj);
  const double (*R)[3] = (const double (*)[3])DOUBLEP(R_obj);
  const double* cell = (const double*)DOUBLEP(cell_obj);
  const char* pbc = (const char*)(cell_obj->data);
  const long* repeat = (const long*)(repeat_obj->data);
  const double (*phi)[nD] = (const double (*)[nD])DOUBLEP(phi_obj);
  double* histogram = (double*)DOUBLEP(histogram_obj);

  double dr = rcut / (histogram_obj->dimensions[0] - 1);
  double rcut2 = rcut * rcut;

  double energy = 0.0;
  if (repeat[0] == 0 && repeat[1] == 0 && repeat[2] == 0)
    for (int i1 = iA; i1 < iB; i1++)
      {
	const double* R1 = R[i1];
	double q01 = q0[i1];
	for (int i2 = 0; i2 < i1; i2++)
	  {
	    double rr = 0.0;
	    for (int c = 0; c < 3; c++)
	      {
		double f = R[i2][c] - R1[c];
		if (pbc[c])
		  f = fmod(f + 1.5 * cell[c], cell[c]) - 0.5 * cell[c];
		rr += f * f;
	      }
	    if (rr < rcut2)
	      {
		double r = sqrt(rr);
		double e12 = (vdwkernel(r, q01, q0[i2],
					nD, ndelta, dD, ddelta, phi) *
			      n[i1] * n[i2]);
		histogram[(int)(r / dr)] += e12; 
		energy += e12;
	      }
	  }
      }
  else
    for (int i1 = iA; i1 < iB; i1++)
      {
	const double* R1 = R[i1];
	double q01 = q0[i1];
	for (int a1 = -repeat[0]; a1 <= repeat[0]; a1++)
	  for (int a2 = -repeat[1]; a2 <= repeat[1]; a2++)
	    for (int a3 = -repeat[2]; a3 <= repeat[2]; a3++)
	      {
		//int i2max = ni;
		double x = 0.5;
		int i2max = iB;
		if (a1 == 0 && a2 == 0 && a3 == 0)
		  {
		    i2max = i1;
		    x = 1.0;
		  }
		double R1a[3] = {R1[0] + a1 * cell[0],
				 R1[1] + a2 * cell[1],
				 R1[2] + a3 * cell[2]};
		for (int i2 = 0; i2 < i2max; i2++)
		  {
		    double rr = 0.0;
		    for (int c = 0; c < 3; c++)
		      {
			double f = R[i2][c] - R1a[c];
			rr += f * f;
		      }
		    if (rr < rcut2)
		      {
			double r = sqrt(rr);
			double e12 = (vdwkernel(r, q01, q0[i2],
						nD, ndelta, dD, ddelta, phi) *
				      n[i1] * n[i2] * x);
			energy += e12;
			histogram[(int)(r / dr)] += e12; 
		      }
		  }
	      }
      }
  return PyFloat_FromDouble(0.25 * energy / M_PI);
}

void vdwkernel2(double r, double q01, double q02, int nD, int ndelta,
		double dD, double ddelta,
		const double (*phi)[nD],
		double* Phi)
{
  const double C = -1024.0 / 243.0 * M_PI * M_PI * M_PI * M_PI;
  double d1 = r * q01;
  double d2 = r * q02;
  double D = 0.5 * (d1 + d2);
  double xD = D / dD;
  int jD = (int)xD;
  if (jD >= nD - 1)
    {
      double d12 = d1 * d1;
      double d22 = d2 * d2;
      double d13 = d12*d1;
      Phi[3] = C / (d12 * d22 * (d12 + d22));
      Phi[0] = d1*(2*C/(d13*d22*(d12+d22))-2*C/(d1*d22*(d12*d22)*(d12*d22)));
      Phi[1] = d12 * (6 * C / (d12 * d12 * d22 * (d12 * d22) + 
			       6 * C / (d12 * d22 *
					(d12 + d22) * (d12 * d22))
			       + 8 * C / (d22 *
					  (d12 + d22) * (d12 + d22)
					  * (d12 + d22))));
      Phi[2] = 0.0;
     }
  else
    {
      double xdelta = fabs(0.5 * (d1 - d2) / D) / ddelta;//OBS!! new interpolation??????
      int jdelta = (int)xdelta;
      double a;
      if (jdelta >= ndelta - 1)
	{
	  jdelta = ndelta - 2;
	  a = 1.0;
	}
      else
	a = xdelta - jdelta;
      double b = xD - jD;
      //original phi interpolation
      Phi[3] = ((a         * b         * phi[jdelta + 1][jD + 1] +
		 a         * (1.0 - b) * phi[jdelta + 1][jD    ] +
		 (1.0 - a) * b         * phi[jdelta    ][jD + 1] +
		 (1.0 - a) * (1.0 - b) * phi[jdelta    ][jD    ]) /
		(D * D));
      //first derivative of phi with D 
      double dphidD = ((          a         * phi[jdelta + 1][jD + 1] -
				   a *         phi[jdelta + 1][jD    ] +
				  (1.0 - a)   * phi[jdelta    ][jD + 1] -
				  (1.0 - a) * phi[jdelta    ][jD    ]) /
			(D * D))-2/(D * D)*Phi[3];

      double dphiddelta = ((          b         * phi[jdelta + 1][jD + 1] +
				  (1.0 - b) * phi[jdelta + 1][jD    ] -
				  b         * phi[jdelta    ][jD + 1] -
				  (1.0 - b) * phi[jdelta    ][jD    ]) /
	     (D * D));
      Phi[0] = d1 * (0.5* dphiddelta+0.5*dphidD);

      

      //getting the interpolation of second derivative
      /*
      double phidd1=phi[jdelta-1][jD]-2*phi[jdelta][jd]+phi[jdelta+1][jD];
      double phidd2=phi[jdelta][jD]-2*phi[jdelta+1][jd]+phi[jdelta+2][jD];
      double phidd3=phi[jdelta-1][jD+1]-2*phi[jdelta][jd+1]+phi[jdelta+1][jD+1];
      double phidd4=phi[jdelta][jD+1]-2*phi[jdelta+1][jd+1]+phi[jdelta+2][jD+1];
      //second derivative of phi with d 
      e14= ((a         * b          * phidd4+
	      a         * (1.0 - b) * phidd2+
	      (1.0 - a) * b         * phidd3+
	     (1.0 - a) * (1.0 - b) * phidd1)/
	    (D * D));
      */
      
      double d2phidddd =d1*d1*(((         1         *  phi[jdelta + 1][jD + 1] -
				   1  *         phi[jdelta + 1][jD    ] -
				   1  *         phi[jdelta    ][jD + 1] +
				   1  *         phi[jdelta    ][jD    ]) /
				(D * D))-2/(D * D)*dphiddelta);

      Phi[1] = d1 * d1 * d2phidddd;
      double d2phiddddp = 0;
      Phi[2] = (0.5* dphiddelta+0.5*dphidD) + d1 * d2phidddd + d2 * d2phiddddp;
    }
}

PyObject * vdw2(PyObject* self, PyObject *args)
{
  
  const PyArrayObject* n_obj;
  const PyArrayObject* q0_obj;
  const PyArrayObject* R_obj;
  const PyArrayObject* cell_obj;
  const PyArrayObject* pbc_obj;
  const PyArrayObject* phi_obj;
  double dD;
  double ddelta;
  int iA;
  int iB;
  const PyArrayObject* a1_obj;
  const PyArrayObject* a2_obj;
  const PyArrayObject* s_obj;
  PyArrayObject* v_obj;
  if (!PyArg_ParseTuple(args, "OOOOOOddiiOOOO",
			&n_obj, &q0_obj, &R_obj,
			&cell_obj, &pbc_obj,
			&phi_obj, &dD, &ddelta, &iA, &iB,
			&a1_obj, &a2_obj, &s_obj, &v_obj))
    return NULL;

  const double Zabover9 = -0.8491 / 9.0;
  int nD = phi_obj->dimensions[1];
  int ndelta = phi_obj->dimensions[1];
  const double* n = (const double*)DOUBLEP(n_obj);
  const double* q0 = (const double*)DOUBLEP(q0_obj);
  const double (*R)[3] = (const double (*)[3])DOUBLEP(R_obj);
  const double* cell = (const double*)DOUBLEP(cell_obj);
  const char* pbc = (const char*)(cell_obj->data);
  const double (*phi)[nD] = (const double (*)[nD])DOUBLEP(phi_obj);
  const double* a1 = (const double*)DOUBLEP(a1_obj);
  const double* a2 = (const double*)DOUBLEP(a2_obj);
  const double (*s)[3] = (const double (*)[3])DOUBLEP(s_obj);

  
  double* potential = (double*)DOUBLEP(v_obj);
  double energy = 0.0;
  for (int i1 = iA; i1 < iB; i1++)
    {
      const double* R1 = R[i1];
      double q01 = q0[i1];
      double e1 = 0.0;
      double e2 = 0.0;
      double e3 = 0.0;
      double e4 = 0.0;
      double A1 = a1[i1];
      double A2 = a2[i1];
      const double* S = s[i1];
      double ie[4]; 
      for (int i2 = 0; i2 < i1; i2++)
	{ 
	  double Rrr[3];
	  double r2 = 0.0;
	  for (int c = 0; c < 3; c++)
	    {
	      double f = R[i2][c] - R1[c];
	      if (pbc[c])
		f = fmod(f + 1.5 * cell[c], cell[c]) - 0.5 * cell[c];
	      r2 += f * f;
	      Rrr[c] = f * f;
	    }
	  double r = sqrt(r2);
	  vdwkernel2(r, q01, q0[i2],
		     nD, ndelta, dD, ddelta, phi, ie);
	  double A3 = Rrr[0] * S[0] + Rrr[1] * S[1] + Rrr[2] * S[2];
	  A3 *= Zabover9 / r;
	  e1 += n[i2] * ie[0];
	  e2 += n[i2] * ie[1] * A1;
	  e3 += n[i2] * ie[2] * A2;
	  e4 += n[i2] * ie[3] * A3;
	}
      potential[i1] += e1 + e2 + e3 + e4;
      
      
      energy += n[i1] * e1;
    }
  return PyFloat_FromDouble(0.25 * energy / M_PI);
}
