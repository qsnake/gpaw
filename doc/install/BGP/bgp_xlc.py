#!/usr/bin/python
"""bgp_xlc.py is a wrapper for the BGP xlc compiler,
   converting/removing incompatible gcc args.   """

import sys
from subprocess import call
from glob import glob

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
non_c99files = glob('c/libxc/src/*.c')

cmd = ""
opt = 1

for arg in sys.argv[1:]:
    cmd += " "
    t = arg.strip()
    if t in fragile_files:
        opt += 1
    if t in non_c99files:
        opt += 2
    if t in args2change:
        cmd += args2change[t]
    else:
        cmd += arg

flags_list = {1: "-O3 -qlanglvl=extc99 -qnostaticlink -qflag=e:e",
              2: "-O3 -qstrict -qlanglvl=extc99 -qnostaticlink -qflag=e:e",
              3: "-O3 -qlanglvl=extc99 -qnostaticlink -qflag=e:e",
              4: "-O3 -qstrict -qlanglvl=extc99 -qnostaticlink -qflag=e:e",
              }

flags = flags_list[opt]  
cmd = "mpixlc_r %s %s"%(flags, cmd)

print "\nexecmd: %s\n"%cmd
call(cmd, shell=True)
