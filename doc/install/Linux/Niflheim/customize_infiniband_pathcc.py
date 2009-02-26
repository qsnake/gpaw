scalapack = True

libraries = [
  'pathfortran',
  'mpiblacsCinit',
  'acml',
  'mpiblacs',
  'scalapack'
  ]

library_dirs = [
  '/opt/pathscale/lib/2.5',
  '/opt/acml-4.0.1/pathscale64/lib',
  '/usr/local/blacs-1.1-24.6.infiniband/lib64',
  '/usr/local/scalapack-1.8.0-1.infiniband/lib64',
  '/usr/local/infinipath-2.0/lib64'
  ]

extra_link_args += [
  '-Wl,-rpath=/opt/pathscale/lib/2.5',
  '-Wl,-rpath=/opt/acml-4.0.1/pathscale64/lib',
  '-Wl,-rpath=/usr/local/blacs-1.1-24.6.infiniband/lib64',
  '-Wl,-rpath=/usr/local/scalapack-1.8.0-1.infiniband/lib64',
  '-Wl,-rpath=/usr/local/infinipath-2.0/lib64'
]

define_macros += [
  ('GPAW_MKL', '1'),
  ('SL_SECOND_UNDERSCORE', '1')
]

mpicompiler = '/usr/local/infinipath-2.0/bin/mpicc -Ofast'
mpilinker = mpicompiler
