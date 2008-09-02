#!/usr/bin/python
"""bg_compiler.py is a wrapper for the xlc compiler, 
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

fragile_files = ["test.c"]

cmd = ""
fragile = False
for arg in sys.argv[1:]:
    cmd += " "
    t = arg.strip()
    if t in fragile_files:
        fragile = True
    if t in args2change:
        cmd += args2change[t]
    else:
        cmd += arg
if fragile:
    flags = "-O3 -qhot=nosimd -qlanglvl=extc99 -qnostaticlink -qsmp -qarch=450d -qtune=450 -qflag=e:e"
else:
    flags = "-O5 -qhot=nosimd -qlanglvl=extc99 -qnostaticlink -qsmp -qarch=450d -qtune=450 -qflag=e:e"
cmd = "mpixlc_r %s %s"%(flags, cmd)

#print "\nexecmd: %s\n"%cmd
call(cmd, shell=True)
