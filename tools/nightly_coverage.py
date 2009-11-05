#!/usr/bin/env python

import os
import sys
import re
import time
import tempfile
import numpy as np
import subprocess as subproc

from glob import glob
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

from gpaw.version import version
from gpaw.svnrevision import svnrevision
from gpaw.utilities import devnull


def CoverageFilter(url, coverage, filtering, output):
    """Filter a coverage file through 'grep -n >>>>>>' and an inlined
    stream editor script in order to convert to reStructuredText."""
    sedname = tempfile.mktemp(prefix='gpaw-sed4rst-')
    htmlfilt = ['s%([\*<._|]{1})%\\\\\\1%g']
    #htmlfilt = ['s%\&%\&amp;%g', 's%>%\&gt;%g', 's%<%\&lt;%g']
    linefilt = ['s%^--$%| --%g',
        's%^([0-9]+)[:-]{1}[ \t]*$%| `\\1 <' + url + '#L\\1>`__:  %g',
        's%^([0-9]+)[:-]{1}([ \t]*)([0-9]+):%| `\\1 <' + url + '#L\\1>`__:  \\2*\\3* %g',
        's%^([0-9]+)[:-]{1}([ \t]*)([>]{6})%| `\\1 <' + url + '#L\\1>`__:  \\2**\\3**%g',
        's%^([0-9]+)[:-]{1}%| `\\1 <' + url + '#L\\1>`__:  %g']
    open(sedname, 'w').write('; '.join(htmlfilt + linefilt))
    la = lb = filtering
    if not isinstance(output, str):
        output.flush() # preserve temporal ordering by flushing pipe
        poi = subproc.Popen('grep -A%d -B%d -n ">>>>>>" "%s" | sed -rf "%s"' \
            % (la,lb,coverage,sedname), shell=True, stdout=output)
        assert poi.wait() == 0
    elif os.system('grep -A%d -B%d -n ">>>>>>" "%s" | sed -rf "%s" > "%s"' \
        % (la,lb,coverage,sedname,output)) != 0:
        raise RuntimeError('Could not parse coverage file "%s".' % coverage)
    os.system('rm -f "%s"' % sedname)


class CoverageParser:
    """Parses any coverage file, optionally filtered through 'grep -n >>>>>>',
    and convert to reStructuredText while maintaining line-by-line counters."""
    def __init__(self, url, coverage, filtering=None):
        self.url = url
        if isinstance(coverage, str):
            self.wait, self.pin = self.create_pipe(coverage, filtering)
            self.direct = (filtering is None)
        else:
            self.pin = coverage
            self.direct = False #assumed to be numbered

        self.nol = self.nos = self.nom = self.noc = 0
        self.buf = None

    def create_pipe(self, covername, filtering=None):
        if filtering is None:
            return lambda: 0, open(covername, 'r')
        else:
            la = lb = filtering
            poi = subproc.Popen('grep -A%d -B%d -n ">>>>>>" "%s"' \
                % (la,lb,covername), shell=True, stdout=subproc.PIPE)
            return poi.wait, poi.stdout

    def digest(self, pattern, replace, add=False):
        assert pattern.startswith('^')
        m = re.match(pattern, self.buf)
        if m is None:
            return False
        args = (m.group(1),self.url,) + m.groups()
        self.buf = replace % args + self.buf[m.end():]
        if add:
            self.noc += int(m.group(3))
        return True

    def parse(self, line):
        for c in '\*<._|':
            line = line.replace(c, '\\'+c)
        self.buf = line
        if self.digest('^([0-9]+)[:-]{1}[ \t]*$', \
                       '| `%s <%s#L%s>`__:  '):
            pass # empty line
        elif self.digest('^([0-9]+)[:-]{1}([ \t]*)([>]{6})', \
                       '| `%s <%s#L%s>`__:  %s**%s**'):
            # statement with zero count
            self.nos += 1
            self.nom += 1
        elif self.digest('^([0-9]+)[:-]{1}([ \t]*)([0-9]+):', \
                         '| `%s <%s#L%s>`__:  %s*%s* ', add=True):
            # statement with non-zero count
            self.nos += 1
        elif self.digest('^([0-9]+)[:-]{1}', '| `%s <%s#L%s>`__:  '):
            pass # no countable statement
        else:
            raise IOError('Could not parse "'+self.buf.strip('\n')+'"')
        self.nol += 1
        return self.buf

    def __iter__(self):
        if self.direct:
            # Reading directly from coverage file so prefix line numbers
            # return ('%d:%s' % (l+1,s) for l,s in enumerate(self.pin))
            for line in self.pin:
                yield self.parse('%d:%s' % (self.nol+1, line))
        else:
            # Ignore grep contingency marks if encountered
            for line in self.pin:
                if line.strip('\n') == '--':
                    yield '| --\n'
                else:
                    yield self.parse(line)
        assert self.wait() == 0
        raise StopIteration

    def write_to_stream(self, pout):
        for line in iter(self):
            pout.write(line)

    def write_to_file(self, filename):
        f = open(filename, 'w')
        self.write_to_stream(f)
        f.close()


class TableIO: # we can't subclass cStringIO.StringIO
    """Formats input into fixed-width text tables for reStructuredText."""
    def __init__(self, widths, simple=False, pipe=None):
        self.widths = widths
        if simple:
            self.seps = {'=':['',' ','\n'], 's':['',' ','\n'], '':['','','']}
            self.ws = ['=', 's', '=', '', '=']
        else:
            self.seps = {'=':['+=', '=+=', '=+\n'], '-':['+-', '-+-', '-+\n'],
                         's':['| ', ' | ', ' |\n'], '':['', '', '']}
            self.ws = ['-', 's', '=', '-', '']
        if pipe is None:
            pipe = StringIO()
        self.pipe = pipe
        self.set_formats(formats=('%-*s',)*len(self.widths), sizes=self.widths)

    def put(self, w, entries, func=None):
        if func is None:
            func = lambda l: w*l
        sep = self.seps[w]
        self.pipe.write(sep[0]+sep[1].join(map(func, entries))+sep[2])

    def set_formats(self, formats, sizes):
        self.formats, self.sizes = formats, sizes

    def add_section(self, w, title):
        self.pipe.write('\n\n%s\n%s\n%s\n\n' % (w*len(title),title,w*len(title)))

    def add_subtitle(self, w, title):
        self.pipe.write('\n%s\n%s\n\n' % (title,w*len(title)))

    def add_heading(self, labels=None):
        if labels is not None:
            self.put(self.ws[0], self.widths)
            self.put(self.ws[1], zip(self.widths,labels), lambda ls: '%-*s' % ls)
        self.put(self.ws[2], self.widths)

    def add_row(self, *args):
        formats, sizes = self.formats, self.sizes
        self.put(self.ws[1], zip(formats,sizes,args), lambda (f,s,a): f % (s,a))
        self.put(self.ws[3], self.widths)

    def write_to_stream(self, pout=None):
        self.put(self.ws[4], self.widths)
        if pout is not None:
            pout.write(self.pipe.getvalue())
            self.pipe.close()

# -------------------------------------------------------------------

#XXX
# gpaw-test --debug --coverage counts.pickle
# mpirun -np 2 gpaw-python gpaw-test --debug --coverage counts.pickle
# mpirun -np 4 gpaw-python gpaw-test --debug --coverage counts.pickle
# mpirun -np 8 gpaw-python gpaw-test --debug --coverage counts.pickle

#if os.path.isfile('counts.pickle'):
#    os.path.remove('counts.pickle')

for ncores in [1,2,4,8]:
    # os.system('mpirun -np %d gpaw-python gpaw-test --debug --coverage counts.pickle' % ncores)

	if not os.path.isfile('counts.pickle'):
    	raise IOError('ERROR: No coverage file generated!')

# python -m trace --report --file counts.pickle --coverdir coverage --missing

# -------------------------------------------------------------------

svnbase = 'https://svn.fysik.dtu.dk/projects/gpaw/trunk'
urlbase = 'https://trac.fysik.dtu.dk/projects/gpaw/browser/trunk'
indexname = 'coverage/index.rst'
tablename = 'coverage/summary.rst'
tod = time.strftime('%d/%m-%Y')

# Initialize pipes as tables for various categories
limits = np.array([0, 0.5, 0.9, 1.0, np.inf])
categories = ['Poor','Mediocre','Good', 'Complete']
pipes = []
l_filename, l_nol, l_nos, l_nom, l_cov, l_noc, l_avg = 42, 4, 4, 4, 8, 10, 10
l_filelink = len(':ref:` <>`') + 2*l_filename-4
for category in categories:
    pipe = TableIO((l_filelink, l_nol, l_nos, l_nom, l_cov, l_noc, l_avg))
    pipe.set_formats( \
        formats=('%-*s', '%*d', '%*d', '%*d', '%*.2f %%', '%*d', '%*.2f'),
        sizes=(l_filelink, l_nol, l_nos, l_nom, l_cov-2, l_noc, l_avg))
    pipe.add_subtitle('-', 'Files with %s coverage' % category.lower())
    pipe.add_heading(('Filename', 'NOL', 'NOS', 'NOM', 'Coverage', \
                     'Executions', 'Average'))
    pipes.append(pipe)

# Generate a toctree with an alphabetical list of files
f = open(indexname, 'w')
f.write("""

-----------------------------------
List of files with missing coverage
-----------------------------------

.. toctree::
   :maxdepth: 1

""")

for covername in sorted(glob('coverage/gpaw.test.*.cover')):
    rstname = covername[:-len('.cover')]+'.rst'
    refname = rstname[len('coverage/'):-len('.rst')]
    print 'covername:', covername, 'rstname:', rstname, 'refname:', refname

    filename = covername[len('coverage/'):-len('.cover')].replace('.','/')+'.py'
    fileurl = '%s/%s?rev=%s' % (urlbase,filename,svnrevision)
    filelink = (':ref:`%s <%s>`' % (filename,refname)).ljust(l_filelink)

    # Tally up how many developers have contributed to the file and by how much
    patn = '^[ \t*]*[0-9]+[ \t]+([^ \t]+)[ \t]+.*$'
    poi = subproc.Popen('svn praise --revision %s "%s/%s" | sed -r ' \
        '"/%s/!d; s/%s/\\1/g"' % (svnrevision,svnbase,filename,patn,patn), \
        shell=True, stdout=subproc.PIPE)
    owners = np.array(poi.stdout.readlines())
    assert poi.wait() == 0
    devels = np.unique(owners)
    totals = np.array([np.sum(owners==devel) for devel in devels])

    p = CoverageParser(fileurl, covername)
    p.write_to_stream(devnull)

    # Skip files which are fully covered, i.e. have no ">>>>>>"
    #if os.system('grep -q ">>>>>>" "%s"' % covername):
    if p.nos == 0:
        continue
    elif p.nom == 0:
        args = (filename, p.nol, p.nos, p.nom, 100.0, p.noc, p.noc/float(p.nos))
        pipes[-1].add_row(*args)
        continue

    f.write('   %s\n' % refname) # add to toctree in index file

    ratio = 1-p.nom/float(p.nos)
    c = np.argwhere((limits[:-1]<=ratio) & (ratio<limits[1:])).item()
    args = (filelink, p.nol, p.nos, p.nom, ratio*100, p.noc, p.noc/float(p.nos))
    pipes[c].add_row(*args)

    g = open(rstname, 'w')
    g.write('\n\n.. _%s:\n' % refname)

    header = TableIO((27, 10), simple=True, pipe=g)
    header.add_section('=', 'Coverage of %s' % filename)
    header.add_heading()
    header.add_row('GPAW version:', version)
    header.add_row('Date of compilation:', tod)
    header.add_row('Coverage category:', categories[c])
    header.add_row('Number of lines (NOL):', p.nol)
    header.add_row('Number of statements (NOS):', p.nos)
    header.add_row('Number of misses (NOM):', p.nom)
    header.write_to_stream()

    l_devel = devels.dtype.itemsize-1 #trailing newline stripped
    svninfo = TableIO((l_devel, l_cov), simple=True, pipe=g)
    svninfo.set_formats(formats=('%-*s', '%*.2f %%'), sizes=(l_devel,l_cov-2))
    svninfo.add_subtitle('-', 'List of developers by contribution')
    svninfo.add_heading()
    for i in np.argsort(totals)[::-1]:
        svninfo.add_row(devels[i].strip(), totals[i]*100/float(len(owners)))
    svninfo.write_to_stream()

    g.write("""
-------------------
Test suite coverage
-------------------
""")
    #CoverageParser(fileurl, covername, 2).write_to_stream(g)
    CoverageFilter(fileurl, covername, 2, g)
    g.close()

# Include the summary in toctree
f.write('   summary')
f.close()

# Build summary with tables for the various categories
h = open(tablename, 'w')
h.write("""
-------
Summary
-------

The following tables summarize the coverage of Python files in GPAW
by the test suite. Files are divided into the following %d categories
based on the amount of test suite coverage:

""" % len(categories))
end = 'of the executable statements were covered.\n'
h.write('- %s coverage: Less than %.0f %% %s' % (categories[0],100*limits[1],end))
for c,category in tuple(enumerate(categories))[1:-1]:
    h.write('- %s coverage: %.0f %% - %.0f %% %s' \
         % (category,100*limits[c],100*limits[c+1],end))
h.write('- %s coverage: %.0f %% %s' % (categories[-1],100*limits[-2],end))

for pipe in pipes:
    pipe.write_to_stream(h)
h.close()
