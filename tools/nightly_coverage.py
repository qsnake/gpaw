#!/usr/bin/env python

import os
import sys
import re
import time
import glob
import tempfile
import numpy as np

from subprocess import Popen, PIPE
from trace import pickle
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

def CoverageFilter(url, coverage, filtering, output):
    """Filter a coverage file through 'grep -n >>>>>>' and an inlined
    stream editor script in order to convert to reStructuredText."""
    sedname = tempfile.mktemp(prefix='gpaw-sed4rst-')
    htmlfilt = [r's%([\*<._|]{1})%\\\1%g']
    #htmlfilt = ['s%\&%\&amp;%g', 's%>%\&gt;%g', 's%<%\&lt;%g']
    linefilt = ['s%^--$%| --%g',
        r's%^([0-9]+)[:-]{1}[ \t]*$%| `\1 <' + url + r'#L\1>`__:  %g',
        r's%^([0-9]+)[:-]{1}([ \t]*)([0-9]+):%| `\1 <' + url + r'#L\1>`__:  \2*\3* %g',
        r's%^([0-9]+)[:-]{1}([ \t]*)([>]{6})%| `\1 <' + url + r'#L\1>`__:  \2**\3**%g',
        r's%^([0-9]+)[:-]{1}%| `\1 <' + url + r'#L\1>`__:  %g']
    open(sedname, 'w').write('; '.join(htmlfilt + linefilt))
    la = lb = filtering
    if not isinstance(output, str):
        output.flush() # preserve temporal ordering by flushing pipe
        poi = Popen('grep -A%d -B%d -n ">>>>>>" "%s" | sed -rf "%s"' \
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
            poi = Popen('grep -A%d -B%d -n ">>>>>>" "%s"' \
                % (la,lb,covername), shell=True, stdout=PIPE)
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
        if self.digest(r'^([0-9]+)[:-]{1}[ \t]*$', \
                       r'| `%s <%s#L%s>`__:  '):
            pass # empty line
        elif self.digest(r'^([0-9]+)[:-]{1}([ \t]*)([>]{6})', \
                         r'| `%s <%s#L%s>`__:  %s**%s**'):
            # statement with zero count
            self.nos += 1
            self.nom += 1
        elif self.digest(r'^([0-9]+)[:-]{1}([ \t]*)([0-9]+):', \
                         r'| `%s <%s#L%s>`__:  %s*%s* ', add=True):
            # statement with non-zero count
            self.nos += 1
        elif self.digest(r'^([0-9]+)[:-]{1}', r'| `%s <%s#L%s>`__:  '):
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

def fail(subject, filename='/dev/null'):
    assert os.system('mail -s "%s" s032082@fysik.dtu.dk < %s' %
                     (subject, filename)) == 0
    raise SystemExit

def svnexport(url, path):
    """Exports a clean directory tree from the repository specified by `url`
    into `path` and return the revision number (of the unversioned copy)."""
    poi = Popen('svn export "%s" %s' % (url,path), shell=True, stdout=PIPE)
    if poi.wait() != 0:
        fail('Checkout of %s failed!' % path.upper())
    lastline = poi.stdout.readlines()[-1]
    return re.match('^Exported revision ([0-9]+).$', lastline).group(1)

svnbase = 'https://svn.fysik.dtu.dk/projects/gpaw/trunk'
tmpdir = tempfile.mkdtemp(prefix='gpaw-coverage-')
hostname = os.getenv('HOSTNAME')

if '--rebuild' in sys.argv[1:]:
    # Build .rst files from .cover files without installing or running tests.
    assert os.path.isfile('counts.out')
    rvs = dict([entry.strip('$ \n').split('=',1) for entry in 
                open('counts.out','r').readline().split(';')])
    os.system('cp counts.out "%s"' % tmpdir)
    assert os.path.isdir('coverage')
    os.system('cp -r coverage "%s/coverage"' % tmpdir)
    os.chdir(tmpdir)
else:
    # Get SVN revision numbers, checkout a fresh version and install:
    rvs = {}
    os.chdir(tmpdir)
    rvs['gpaw'] = svnexport(svnbase, 'gpaw')
    rvs['ase'] = svnexport('https://svn.fysik.dtu.dk/projects/ase/trunk', 'ase')
    os.chdir('gpaw')
    open('counts.out','w').write('$ '+';'.join(map('='.join, rvs.items()))+'\n')

    # Temporary installations of GPAW/ASE revisions and latest setups
    if hostname == 'thul.fysik.dtu.dk':
        customload = 'source /home/camp/modulefiles.sh; ' \
                     'module load NUMPY; ' \
                     'module load openmpi/1.3.3-1.el5.fys.gfortran43.4.3.2; '
        flags = '--remove-default-flags ' \
                '--customize=doc/install/Linux/Niflheim/customize-thul-acml.py'
        pydir = 'lib64/python'
    elif hostname.endswith('.fysik.dtu.dk'):
        customload = ''
        flags = ''
        pydir = 'lib/python'
    else:
        raise EnvironmentError('Unknown host. Automatic installation failed.')

    if os.system(customload.replace(';', '&&') +
                 'python setup.py %s install --home=%s ' % (flags,tmpdir) +
                 '2>&1 | grep -v "c/libxc/src"') != 0 \
        or os.system('mv ../ase/ase ../%s' % pydir) != 0:
        fail('Installation failed!')

    os.system('wget --no-check-certificate --quiet ' +
              'http://wiki.fysik.dtu.dk/stuff/gpaw-setups-latest.tar.gz')
    os.system('tar xvzf gpaw-setups-latest.tar.gz')
    setups = tmpdir + '/gpaw/' + glob.glob('gpaw-setups-[0-9]*')[0]

    # Repeatedly run test-suite in code coverage mode:
    args = '--debug --coverage counts.pickle'
    for cpus in [1,2,4,8]:
        tod = time.strftime('%d/%m-%Y %H:%M:%S')
        open('counts.out', 'a').write('\n\n%s - %d thread(s).\n' % (tod,cpus))
        if os.system(customload +
                     'export PYTHONPATH=%s/%s:$PYTHONPATH; ' % (tmpdir,pydir) +
                     'export GPAW_SETUP_PATH=%s; ' % setups +
                     'export IGNOREPATHS=%s; ' % os.getenv('HOME') +
                     'mpiexec -np %d ' % cpus +
                     tmpdir + '/bin/gpaw-python ' +
                     'tools/gpaw-test %s >>counts.out 2>&1' % args) != 0 \
            or not os.path.isfile('counts.pickle'):
            fail('Test coverage failed!', 'counts.out')

    # Convert pickled coverage information to .cover files with clear text
    if os.system('export PYTHONPATH=%s/%s:$PYTHONPATH; %s -m trace --report ' \
        '--missing --file counts.pickle --coverdir coverage >>counts.out 2>&1' \
        % (tmpdir,pydir,sys.executable)) != 0:
        fail('Coverage conversion failed!', 'counts.out')

    if os.system('tar cvzf gpaw-counts-%s.tar.gz counts.* ' \
                 'coverage/gpaw.*.cover' % rvs['gpaw']) == 0:
        home = os.getenv('HOME')
        try:
            os.mkdir(home + '/sphinx')
        except OSError:
            pass
        os.system('cp --backup=existing gpaw-counts-%s.tar.gz ' \
        '"%s/sphinx/gpaw-counts-latest.tar.gz"' % (rvs['gpaw'],home))

# -------------------------------------------------------------------

# Parse output to test suite logfile and generate reStructuredText
f = open('testsuite.rst', 'w')
f.write('.. _testsuite:\n')
loginfo = TableIO((15, max(10,len(hostname))), simple=True, pipe=f)
loginfo.add_section('=', 'Test suite')
loginfo.add_heading()
for k,v in rvs.items():
    loginfo.add_row('%s revision:' % k.upper(), v)
loginfo.add_row('Build date:', time.strftime('%d/%m-%Y'))
loginfo.add_row('Ran on host:', hostname)
loginfo.write_to_stream()
assert os.system('tail -n+2 counts.out >testsuite.log') == 0
f.write('\n\n.. literalinclude:: testsuite.log\n\n')
f.close()

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


urlbase = 'https://trac.fysik.dtu.dk/projects/gpaw/browser/trunk'
indexname = 'coverage/index.rst'
tablename = 'coverage/summary.rst'
devnull = open('/dev/null', 'w', buffering=0)

# Generate a toctree with an alphabetical list of files
f = open(indexname, 'w')
f.write("""

-----------------------------------
List of files with missing coverage
-----------------------------------

.. toctree::
   :maxdepth: 1

""")

namefilt = re.compile('coverage/(gpaw\..*)\.cover')
for covername in sorted(glob.glob('coverage/gpaw.*.cover')):
    refname = namefilt.match(covername).group(1)
    rstname = 'coverage/' + refname + '.rst'
    print 'cover:', covername, '->', rstname, 'as :ref:`%s`' % refname
    filename = refname.replace('.','/')+'.py' # unmangle paths
    fileurl = '%s/%s?rev=%s' % (urlbase,filename,rvs['gpaw'])
    filelink = (':ref:`%s <%s>`' % (filename,refname)).ljust(l_filelink)

    # Tally up how many developers have contributed to the file and by how much
    patn = r'^[ \t*]*[0-9]+[ \t]+([^ \t]+)[ \t]+.*$'
    poi = Popen('svn praise -r %s "%s/%s" | sed -r "/%s/!d; s/%s/\\1/g"' \
        % (rvs['gpaw'],svnbase,filename,patn,patn), shell=True, stdout=PIPE)
    owners = np.array(poi.stdout.readlines())
    assert poi.wait() == 0
    devels = np.unique(owners)
    totals = np.array([np.sum(owners==devel) for devel in devels])

    p = CoverageParser(fileurl, covername)
    p.write_to_stream(devnull)

    # Skip files which are fully covered, i.e. have no ">>>>>>"
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
    for k,v in rvs.items():
        header.add_row('%s revision:' % k.upper(), v)
    header.add_row('Date of compilation:', time.strftime('%d/%m-%Y'))
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
f.write('   summary\n')
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

if os.system('tar cvzf gpaw-coverage-%s.tar.gz testsuite.* ' \
             'coverage/*.rst' % rvs['gpaw']) == 0:
    home = os.getenv('HOME')
    try:
        os.mkdir(home + '/sphinx')
    except OSError:
        pass
    assert os.system('cp --backup=existing gpaw-coverage-%s.tar.gz ' \
        '"%s/sphinx/gpaw-coverage-latest.tar.gz"' % (rvs['gpaw'],home)) == 0

os.system('cd; rm -r ' + tmpdir)
