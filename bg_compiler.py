#!/usr/bin/python
"""gcc2xlc.py is a wrapper for the xlc compiler, 
   converting/removing incompatible gcc args.   """


import sys
from subprocess import call


args2change = {"-fno-strict-aliasing":"",
               "-fmessage-length=0":"",
               "-Wall":"",
               "-std=c99":"-qlanglvl=extc99",
               "-fPIC":"",
               "-g":"",
               "-D_FORTIFY_SOURCE=2":"",
               "-DNDEBUG":"",
               "-UNDEBUG":"",
               "-pthread":"",
               "-shared":"-qmkshrobj",
               "-Xlinker":"",
               "-export-dynamic":"",
               "-Wstrict-prototypes":"",
               "-dynamic":"",
               "-O3":"",
               "-O2":"",
               "-O1":""}

cmd = "mpixlc_r -O3 -qhot=nosimd -qlanglvl=extc99 -qnostaticlink -qsmp -qarch=450d -qtune=450 -qflag=e:e"

for arg in sys.argv[1:]:
	cmd += " "
	t = arg.strip()
	if t in args2change:
		cmd += args2change[t]
	else:
		cmd += arg

#print "\nexecmd: %s\n"%cmd
call(cmd, shell=True)
