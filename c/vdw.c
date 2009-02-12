#include "extensions.h"

double vdwkernel(double D, double d1, double d2, int nD, int ndelta,
		 double dD, double ddelta,
		 const double (*phi)[nD])
{
  if (D < 1e-10)
    return phi[0][0];

  double y = D / dD;
  int j = (int)y;
  double e12;
  if (j >= nD - 1)
    {
      double d12 = d1 * d1;
      double d22 = d2 * d2;
      const double C = -1024.0 / 243.0 * M_PI * M_PI * M_PI * M_PI;
      e12 = C / (d12 * d22 * (d12 + d22));
    }
  else
    {
      double x = fabs(0.5 * (d1 - d2) / D) / ddelta;
      int i = (int)x;
      if (i >= ndelta - 1)
	{
	  i = ndelta - 2;
	  x = 1.0;
	}
      else
	x -= i;
      y -= j;
      e12 = ((x         * y         * phi[i + 1][j + 1] +
	      x         * (1.0 - y) * phi[i + 1][j    ] +
	      (1.0 - x) * y         * phi[i    ][j + 1] +
	      (1.0 - x) * (1.0 - y) * phi[i    ][j    ]));
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
  double ddelta;
  double dD;
  int iA;
  int iB;
  PyArrayObject* rhistogram_obj;
  double drhist;
  PyArrayObject* Dhistogram_obj;
  double dDhist;
  if (!PyArg_ParseTuple(args, "OOOOOOOddiiOdOd", &n_obj, &q0_obj, &R_obj,
			&cell_obj, &pbc_obj, &repeat_obj,
			&phi_obj, &ddelta, &dD, &iA, &iB,
			&rhistogram_obj, &drhist,
			&Dhistogram_obj, &dDhist))
    return NULL;

  int ndelta = phi_obj->dimensions[0];
  int nD = phi_obj->dimensions[1];
  const double* n = (const double*)DOUBLEP(n_obj);
  const double* q0 = (const double*)DOUBLEP(q0_obj);
  const double (*R)[3] = (const double (*)[3])DOUBLEP(R_obj);
  const double* cell = (const double*)DOUBLEP(cell_obj);
  const char* pbc = (const char*)(cell_obj->data);
  const long* repeat = (const long*)(repeat_obj->data);
  const double (*phi)[nD] = (const double (*)[nD])DOUBLEP(phi_obj);
  double* rhistogram = (double*)DOUBLEP(rhistogram_obj);
  double* Dhistogram = (double*)DOUBLEP(Dhistogram_obj);

  int nbinsr = rhistogram_obj->dimensions[0];
  int nbinsD = Dhistogram_obj->dimensions[0];

  double energy = 0.0;
  if (repeat[0] == 0 && repeat[1] == 0 && repeat[2] == 0)
    for (int i1 = iA; i1 < iB; i1++)
      {
	const double* R1 = R[i1];
	double q01 = q0[i1];
	for (int i2 = 0; i2 <= i1; i2++)
	  {
	    double rr = 0.0;
	    for (int c = 0; c < 3; c++)
	      {
		double f = R[i2][c] - R1[c];
		if (pbc[c])
		  f = fmod(f + 1.5 * cell[c], cell[c]) - 0.5 * cell[c];
		rr += f * f;
	      }
            double r = sqrt(rr);
            double d1 = r * q01;
            double d2 = r * q0[i2];
            double D = 0.5 * (d1 + d2);
            double e12 = (vdwkernel(D, d1, d2, nD, ndelta, dD, ddelta, phi) *
                          n[i1] * n[i2]);
            if (i1 == i2)
              e12 /= 2.0;
            int bin = (int)(r / drhist);
            if (bin < nbinsr)
              rhistogram[bin] += e12; 
            bin = (int)(D / dDhist);
            if (bin < nbinsD)
              Dhistogram[bin] += e12; 
            energy += e12;
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
		for (int i2 = 0; i2 <= i2max; i2++)
		  {
		    double rr = 0.0;
		    for (int c = 0; c < 3; c++)
		      {
			double f = R[i2][c] - R1a[c];
			rr += f * f;
		      }
                    double r = sqrt(rr);
                    double d1 = r * q01;
                    double d2 = r * q0[i2];
                    double D = 0.5 * (d1 + d2);
                    double e12 = (vdwkernel(D, d1, d2,
                                            nD, ndelta, dD, ddelta, phi) *
                                  n[i1] * n[i2] * x);
                    int bin = (int)(r / drhist);
                    if (bin < nbinsr)
                      rhistogram[bin] += e12; 
                    bin = (int)(D / dDhist);
                    if (bin < nbinsD)
                      Dhistogram[bin] += e12; 
                    energy += e12;
		  }
	      }
      }
  return PyFloat_FromDouble(energy);
}

PyObject * vdw2(PyObject* self, PyObject *args)
{
  const PyArrayObject* phi_jp_obj;
  const PyArrayObject* j_k_obj;
  const PyArrayObject* dk_k_obj;
  const PyArrayObject* theta_k_obj;
  const PyArrayObject* F_k_obj;
  if (!PyArg_ParseTuple(args, "OOOOO", &phi_jp_obj, &j_k_obj, &dk_k_obj,
			&theta_k_obj, &F_k_obj))
    return NULL;

  const double* phi_jp = (const double*)phi_jp_obj->data;
  const long* j_k = (const long*)j_k_obj->data;
  const double* dk_k = (const double*)dk_k_obj->data;
  const complex double* theta_k = (const complex double*)theta_k_obj->data;
  complex double* F_k = (complex double*)F_k_obj->data;

  int nk = PyArray_SIZE(j_k_obj);
  for (int k = 0; k < nk; k++)
    {
      const double* phi_p = phi_jp + 4 * j_k[k];
      double a = phi_p[0];
      double b = phi_p[1];
      double c = phi_p[2];
      double d = phi_p[3];
      double x = dk_k[k];
      F_k[k] += theta_k[k] * (a + x * (b + x * (c + x * d)));
    }
  Py_RETURN_NONE;
}
