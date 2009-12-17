import sys
import datetime
import numpy as np
d = []
for x in sys.argv[1:]:
    execfile(x)
    d.append((date, results))
d.sort()
d2 = {}
i = 0
for date, results in d:
    for name in results:
        t, status = results[name]
        if status == 'done':
            if name in d2:
                d2[name].append((i, t))
            else:
                d2[name] = [(i, t)]
    i += 1
N = i
d3 = []
for name, xy in d2.items():
    x, y = zip(*xy)
    y = np.array(y)
    t = y.min()
    y *= 100 / t
    y -= 100
    d3.append((t, name, x, y))
d3.sort()
for t, name, x, y in d3:
    print '%-30s %7.1f ' % (name, t),
    xy = dict(zip(x, y))
    for i in range(N):
        if i in xy:
            print '%6.1f' % xy[i],
        else:
            print '      ',
    print
