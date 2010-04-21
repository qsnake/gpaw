import os
import sys
import time
import random


class AGTSJob:
    def __init__(self, dir, script,
                 ncpus=1, walltime=10 * 60, deps=None, creates=None,
                 show=None):
        """Advaced GPAW test system job.

        Example:

        >>> job = AGTSJob('doc/tutorial/xas', 'run.py --setups=.')
        >>> job.dir
        'doc/tutorial/xas'
        >>> job.script
        'run.py'
        >>> job.args
        '--setups=.'
        >>> job.name
        'run.py_--setups=.'
        >>> job.absname
        'doc/tutorial/xas/run.py_--setups=.'
        """
        
        if ' ' in script:
            script, self.args = script.split(' ', 1)
        else:
            self.args = ''
        pathname = os.path.normpath(os.path.join(dir, script))
        self.dir, self.script = os.path.split(pathname)
        if self.dir == '':
            self.dir = '.'
        self.absname = pathname
        if self.args:
            self.absname += '_' + self.args.replace(' ', '_')
        dir, self.name = os.path.split(self.absname)
        self.ncpus = ncpus
        self.walltime = walltime
        if deps:
            self.deps = deps
        else:
            self.deps = []
        self.creates = creates

        # Filenames to use for pylab.savefig() replacement of pylab.show():
        if not show:
            show = []
        self.show = show
        
        self.status = 'waiting'
        self.tstart = None
        self.tstop = None
        self.exitcode = None


class Cluster:
    def check_status(self, job):
        name = job.absname
        if job.status == 'running':
            if time.time() - job.tstart > job.walltime:
                job.status = 'TIMEOUT'
                return 'TIMEOUT'
            if os.path.exists('%s.done' % name):
                job.tstop = os.stat('%s.done' % name).st_mtime
                job.exitcode = int(open('%s.done' % name).readlines()[-1])
                if job.exitcode:
                    job.status = 'FAILED'
                else:
                    job.status = 'success'
                return job.status

        elif job.status == 'submitted' and os.path.exists('%s.start' % name):
            job.tstart = os.stat('%s.start' % name).st_mtime
            job.status = 'running'
            return 'running'

        # Nothing happened:
        return None

    def write_pylab_wrapper(self, job):
        """Use Agg backend and prevent windows from popping up."""
        fd = open(job.script + '.py', 'w')
        fd.write('from gpaw.test import wrap_pylab\n')
        fd.write('wrap_pylab(%s)\n' % job.show)
        fd.write('execfile(%s)\n' % job.script)
        fd.close()

    def clean(self, job):
        try:
            os.remove('%s.start' % job.absname)
        except OSError:
            pass
        try:
            os.remove('%s.done' % job.absname)
        except OSError:
            pass


class TestCluster(Cluster):
    def submit(self, job):
        if random.random() < 0.3:
            # randomly fail some of the jobs
            exitcode = 1
        else:
            exitcode = 0
        
        wait = random.randint(1, 12)
        cmd = 'sleep %s; touch %s.start; ' % (wait, job.absname)

        if random.random() < 0.3:
            # randomly time out some of the jobs
            pass
        else:
            duration = random.randint(4, 12)
            cmd += 'sleep %s; ' % duration
            if exitcode == 0 and job.creates:
                for filename in job.creates:
                    cmd += 'echo 42 > %s; ' % os.path.join(job.dir, filename)
            cmd += 'echo %d > %s.done' % (exitcode, job.absname)
        os.system('(%s)&' % cmd)


class AGTSQueue:
    def __init__(self, sleeptime=60, log=sys.stdout):
        self.sleeptime = sleeptime
        self.jobs = []

        if isinstance(log, str):
            self.fd = open(log, 'w')
        else:
            self.fd = log

        # used by add() method:
        self._dir = None

    def log(self, job):
        N = dict(waiting=0, submitted=0, running=0,
                 success=0, FAILED=0, disabled=0, TIMEOUT=0)
        for j in self.jobs:
            N[j.status] += 1
        self.fd.write('%s %2d %2d %2d %2d %2d %2d %2d %-49s %s\n' %
                      (time.strftime('%H:%M:%S'),
                       N['waiting'], N['submitted'], N['running'],
                       N['success'], N['FAILED'], N['disabled'], N['TIMEOUT'],
                       job.absname, job.status))
        self.fd.flush()
 
    def add(self, script, dir=None, ncpus=1, walltime=15,
            deps=None, creates=None, show=None):
        """Add job.

        XXX move docs from doc/devel/testing to here and use Sphinx autodoc."""
        
        if dir is None:
            dir = self._dir
        job = AGTSJob(dir, script, ncpus, walltime * 60, deps, creates, show)
        self.jobs.append(job)
        return job

    def locate_tests(self, dir):
        for root, dirs, files in os.walk(dir):
            for fname in files:
                if fname.endswith('.agts.py'):
                    yield root, fname

    def collect(self, dir='.'):
        """Find agts.py files and collect jobs."""
        for dir, agtsfile in self.locate_tests(dir):
            _global = {}
            execfile(os.path.join(dir, agtsfile), _global)
            self._dir = dir
            _global['agts'](self)
        self.normalize()

    def normalize(self):
        """Convert string dependencies to actual job objects."""
        for job in self.jobs:
            for i, dep in enumerate(job.deps):
                if not isinstance(dep, AGTSJob):
                    absname = os.path.normpath(os.path.join(job.dir, dip))
                    job.deps[i] = self.find(absname)

    def find(self, absname):
        """Find job with a particular name."""
        for job in self.jobs:
            if job.absname == absname:
                return job

    def run(self, cluster):
        """Run jobs and return the number of unsuccessful jobs."""
        self.clean(cluster)
        self.status()
        self.fd.write('time      W  S  R  +  -  .  T job\n')
        jobs = self.jobs
        while True:
            done = True
            for job in jobs:
                if job.status == 'waiting':
                    done = False
                    ready = True
                    for dep in job.deps:
                        if dep.status != 'success':
                            ready = False
                            break
                    if ready:
                        cluster.submit(job)
                        job.status = 'submitted'
                        self.log(job)
                elif job.status in ['running', 'submitted']:
                    done = False

            if done:
                break

            time.sleep(self.sleeptime)

            for job in jobs:
                newstatus = cluster.check_status(job)
                if newstatus:
                    self.log(job)
                    if newstatus in ['TIMEOUT', 'FAILED']:
                        self.fail(job)

        self.status()

        return len([None for job in self.jobs if job != 'success'])
    
    def status(self):
        self.fd.write('job                                                ' +
                      'status      time  tmax ncpus  deps\n')
        for job in self.jobs:
            if job.tstop is not None:
                t = '%5d' % round(job.tstop - job.tstart)
            else:
                t = '     '
            self.fd.write('%-50s %-10s %s %5d %5d %5d\n' %
                          (job.absname, job.status, t, job.walltime,
                           job.ncpus, len(job.deps)))

    def fail(self, dep):
        """Recursively disable jobs depending on failed job."""
        for job in self.jobs:
            if dep in job.deps:
                job.status = 'disabled'
                self.log(job)
                self.fail(job)

    def clean(self, cluster):
        for job in self.jobs:
            cluster.clean(job)

    def copy_created_files(self, dir):
        for job in self.jobs:
            if job.creates:
                for filename in job.creates:
                    path = os.path.join(job.dir, filename)
                    if os.path.isfile(path):
                        os.system('cp %s %s' %
                                  (path, os.path.join(dir, filename)))


if __name__ == '__main__':
    # Quick test using dummy cluster and timeout after only 10 seconds:
    c = TestCluster()
    queue = AGTSQueue(sleeptime=2)
    queue.collect()
    for job in queue.jobs:
        job.walltime = 10

    queue.run(c)
    queue.copy_created_files('.')

    # Analysis:
    from gpaw.test.big.analysis import analyse
    mailto = None # None => print to stdout, or email address
    analyse(queue,
            os.getenv('HOME') + '/analyse.pickle',  # file keeping history
            os.getenv('HOME') + '/tmp/out',         # Where to dump figures!
            rev=None,                               # gpaw revision
            mailto=mailto)
