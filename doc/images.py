# creates: logo-csc.gif  logo-dtu.gif  logo-gpaw.png  logo-hut.png  logo-jyu.png  logo-tree.png  logo-tut.png logo-fmf.png

from urllib import urlretrieve

names = """logo-csc.gif  logo-dtu.gif  logo-gpaw.png  logo-hut.png  logo-jyu.png  logo-tree.png  logo-tut.png logo-fmf.png
""".split()

basepath = 'http://dcwww.camd.dtu.dk/~s021864/temp-gpaw-wiki-files/img/'

for name in names:
    urlretrieve(basepath + name, name)
