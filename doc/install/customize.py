#User provided customizations for the gpaw setup

#Here, one can override the default arguments, or append own 
#arguments to default ones
#To override use the form
#     libraries = ['somelib',otherlib']
#To append use the form
#     libraries += ['somelib',otherlib']

extra_compile_args = ['-fastsse']
mpicompiler = 'cc'
numpybase = '/fs/proj1/mika/jussie/numpy-1.0.4/'
numpylibs = [numpybase + 'build/lib.linux-x86_64-2.5/numpy/core/multiarray.a']
numpylibs.append(numpybase + 'build/lib.linux-x86_64-2.5/numpy/core/umath.a')
numpylibs.append(numpybase + 'build/lib.linux-x86_64-2.5/numpy/core/_sort.a')
numpylibs.append(numpybase + 'build/lib.linux-x86_64-2.5/numpy/core/scalarmath.a')
numpylibs.append(numpybase + 'build/lib.linux-x86_64-2.5/numpy/lib/_compiled_base.a')
numpylibs.append(numpybase + 'build/lib.linux-x86_64-2.5/numpy/numarray/_capi.a')
numpylibs.append(numpybase + 'build/lib.linux-x86_64-2.5/numpy/fft/fftpack_lite.a')
numpylibs.append(numpybase + 'build/lib.linux-x86_64-2.5/numpy/linalg/lapack_lite.a')
numpylibs.append(numpybase + 'build/lib.linux-x86_64-2.5/numpy/random/mtrand.a')
extra_link_args += ['-L/fs/proj1/mika/jussie/GotoBLAS -lgoto']
extra_link_args += numpylibs
extra_link_args += ['-L/fs/proj1/mika/jussie/lib/expat/lib',
                    '-L/fs/proj1/mika/jussie/zlib-1.2.3']
extra_link_args += ['-lz', '-lexpat']

mpi_define_macros = [('NO_SOCKET','1')]
