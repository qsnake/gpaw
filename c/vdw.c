#include "extensions.h"

PyObject * vdw(PyObject* self, PyObject *args)
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
  if (!PyArg_ParseTuple(args, "OOOOOOddii", &n_obj, &q0_obj, &R_obj,
			&cell_obj, &pbc_obj, &phi_obj, &dD, &ddelta, &iA, &iB))
    return NULL;

  int nD = phi_obj->dimensions[1];
  int ndelta = phi_obj->dimensions[1];
  const double* n = (const double*)DOUBLEP(n_obj);
  const double* q0 = (const double*)DOUBLEP(q0_obj);
  const double (*R)[3] = (const double (*)[3])DOUBLEP(R_obj);
  const double* cell = (const double*)DOUBLEP(cell_obj);
  const char* pbc = (const char*)(cell_obj->data);
  const double (*phi)[nD] = (const double (*)[nD])DOUBLEP(phi_obj);

  const double C = -1024.0 / 243.0 * M_PI * M_PI * M_PI * M_PI;
  double energy = 0.0;
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
	  double r = sqrt(r2);
	  double d1 = r * q01;
	  double d2 = r * q0[i2];
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
	  e1 += n[i2] * e12;
	}
      energy += n[i1] * e1;
    }
  return PyFloat_FromDouble(0.25 * energy / M_PI);
}
