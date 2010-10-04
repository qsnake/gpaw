# creates: publications.png
import datetime
months = [datetime.date(2000, m, 1).strftime('%B').lower()
          for m in range(1, 13)]
publications = []
for line in open('literature.rst'):
    words = line.split()
    if len(words) == 4 and words[0] == '..' and words[1][0] != '_':
        day = datetime.date(int(words[3]),
                            months.index(words[2].lower()) + 1,
                            int(words[1]))
        publications.append(day)

#publications.sort()
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
n = 2
x = publications[n:]
y = range(n + 1, len(publications) + 1)
z = [d.toordinal() for d in x]
p = np.polyfit(z, np.log(y), 1)

plt.figure(figsize=(10, 5))
plt.semilogy(x, np.exp(np.polyval(p, z)), lw=3,
             label='Doubling time: %d days' % round(np.log(2) / p[0]))
plt.semilogy(x, y, 'ro', mew=2)
plt.title('Number of publications')
plt.legend(loc='lower right')
plt.plot()
plt.savefig('publications.png')
#plt.savefig('publications.eps')
#plt.show()
