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
  if (!PyArg_ParseTuple(args, "OOOOOOOddii", &n_obj, &q0_obj, &R_obj,
			&cell_obj, &pbc_obj, &repeat_obj,
			&phi_obj, &dD, &ddelta, &iA, &iB))
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

  double energy = 0.0;
  if (repeat[0] == 0 && repeat[1] == 0 && repeat[2] == 0)
    for (int i1 = iA; i1 < iB; i1++)
      {
	const double* R1 = R[i1];
	double q01 = q0[i1];
	double e1 = 0.0;
	for (int i2 = 0; i2 < i1; i2++)
	  {
	    double r2 = 0.0;
	    for (int c = 0; c < 3; c++)
	      {
		double f = R[i2][c] - R1[c];
		if (pbc[c])
		  f = fmod(f + 1.5 * cell[c], cell[c]) - 0.5 * cell[c];
		r2 += f * f;
	      }
	    double e12 = vdwkernel(sqrt(r2), q01, q0[i2],
				   nD, ndelta, dD, ddelta, phi);
	    e1 += n[i2] * e12;
	  }
	energy += n[i1] * e1;
      }
  else
    for (int i1 = iA; i1 < iB; i1++)
      {
	const double* R1 = R[i1];
	double q01 = q0[i1];
	double e1 = 0.0;
	for (int a1 = -repeat[0]; a1 <= repeat[0]; a1++)
	  for (int a2 = -repeat[1]; a2 <= repeat[1]; a2++)
	    for (int a3 = -repeat[2]; a3 <= repeat[2]; a3++)
	      {
		//int i2max = ni;
		int i2max = iB;
		if (a1 == 0 && a2 == 0 && a3 == 0)
		  i2max = i1;
		double R1a[3] = {R1[0] + a1 * cell[0],
				 R1[1] + a2 * cell[1],
				 R1[2] + a3 * cell[2]};
		for (int i2 = 0; i2 < i2max; i2++)
		  {
		    double r2 = 0.0;
		    for (int c = 0; c < 3; c++)
		      {
			double f = R[i2][c] - R1a[c];
			r2 += f * f;
		      }
		    double e12 = vdwkernel(sqrt(r2), q01, q0[i2],
					   nD, ndelta, dD, ddelta, phi);
		    e1 += n[i2] * e12;
		  }
	      }
	energy += n[i1] * e1;
      }
  return PyFloat_FromDouble(0.25 * energy / M_PI);
}
