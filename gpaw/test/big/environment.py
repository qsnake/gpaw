import os
import sys
import time
import random


class AGTSJob:
    def __init__(self, dir, script, args=None,
                 ncpus=1, walltime=10 * 60,
                 dependencies=None, creates=None, agtsfile=None):
        pathname = os.path.normpath(os.path.join(dir, script))
        self.dir, self.script = os.path.split(pathname)
        if self.dir == '':
            self.dir = '.'
        if args is None:
            if ' ' in script:
                script, self.args = script.split(' ', 1)
            else:
                self.args = ''
        elif isinstance(args, str):
            self.args = ' ' + args
        else:
            self.args = ' ' + ' '.join(args)
        self.absname = pathname
        if self.args:
            self.absname += '_' + self.args.replace(' ', '_')
        dir, self.name = os.path.split(self.absname)
        self.ncpus = ncpus
        self.walltime = walltime
        if dependencies:
            self.dependencies = dependencies
        else:
            self.dependencies = []
        self.creates = creates
        self.agtsfile = agtsfile

        self.status = 'waiting'
        self.tstart = None
        self.tstop = None
        self.exitcode = None


class Cluster:
    def check_status(self, job):
        name = job.absname
        if job.status == 'running':
            if time.time() - job.tstart > job.walltime:
                job.status = 'timeout'
                return 'timeout'
            if os.path.exists('%s.done' % name):
                job.tstop = os.stat('%s.done' % name).st_mtime
                job.exitcode = int(open('%s.done' % name).readlines()[-1])
                if job.exitcode:
                    job.status = 'failed'
                else:
                    job.status = 'succes'
                return job.status

        elif job.status == 'submitted' and os.path.exists('%s.start' % name):
            job.tstart = os.stat('%s.start' % name).st_mtime
            job.status = 'running'
            return 'running'

        # Nothing happened:
        return None


class TestCluster(Cluster):
    def submit(self, job):
        wait = random.randint(1, 12)
        duration = random.randint(4, 12)
        if random.random() < 0.3:
            # randomly fail some of the jobs
            exitcode = 1
        else:
            exitcode = 0
        
        if random.random() < 0.3:
            # randomly time out some of the jobs
            os.system('(sleep %s; touch %s.start) &' %
                      (wait, job.absname))
        else:
            os.system('(sleep %s; touch %s.start; sleep %s; echo %d > %s.done)&'
                      % (wait, job.absname, duration, exitcode, job.absname))

    def clean(self, job):
        try:
            os.remove('%s.start' % job.absname)
        except OSError:
            pass
        try:
            os.remove('%s.done' % job.absname)
        except OSError:
            pass


class AGTSJobs:
    def __init__(self, sleeptime=60, log=sys.stdout):
        self.sleeptime = sleeptime
        self.jobs = []

        if isinstance(log, str):
            self.fd = open(log, 'w')
        else:
            self.fd = log

        # used by add() method:
        self._dir = None
        self._agtsfile = None

    def log(self, job):
        N = dict(waiting=0, submitted=0, running=0,
                 succes=0, failed=0, disabled=0, timeout=0)
        for j in self.jobs:
            N[j.status] += 1
        self.fd.write('%s %2d %2d %2d %2d %2d %2d %2d %-39s %s\n' %
                      (time.strftime('%H:%M:%S'),
                       N['waiting'], N['submitted'], N['running'],
                       N['succes'], N['failed'], N['disabled'], N['timeout'],
                       job.absname, job.status))
        self.fd.flush()
 
    def add(self, script, dir=None, args=None, ncpus=1, walltime=15,
            depends=None, creates=None):
        if dir is None:
            dir = self._dir
        print walltime;walltime = 10
        job = AGTSJob(dir, script, args, ncpus, walltime * 60,
                      depends, creates, self._agtsfile)
        self.jobs.append(job)
        return job

    def locate_tests(self):
        for root, dirs, files in os.walk('.'):
            for fname in files:
                if fname.endswith('.agts.py'):
                    yield root, fname

    def collect(self):
        for dir, agtsfile in self.locate_tests():
            print dir, agtsfile
            _global = {}
            execfile(os.path.join(dir, agtsfile), _global)
            self._dir = dir
            self.agtsfile = agtsfile
            _global['agtsmain'](self)
        self.normalize()

    def normalize(self):
        for job in self.jobs:
            for i, dep in enumerate(job.dependencies):
                if not isinstance(dep, AGTSJob):
                    absname = os.path.normpath(os.path.join(job.dir, dip))
                    job.dependencies[i] = self.find(absname)

    def find(self, absname):
        for job in self.jobs:
            if job.absname == absname:
                return job

    def run(self, cluster):
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
                    for dep in job.dependencies:
                        if dep.status != 'succes':
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
                    if newstatus in ['timeout', 'failed']:
                        self.fail(job)

        self.status()
    
    def status(self):
        self.fd.write('job                                      ' +
                      'status      time  tmax ncpus  deps\n')
        for job in self.jobs:
            if job.tstop is not None:
                t = '%5d' % round(job.tstop - job.tstart)
            else:
                t = '     '
            self.fd.write('%-40s %-10s %s %5d %5d %5d\n' %
                          (job.absname, job.status, t, job.walltime,
                           job.ncpus, len(job.dependencies)))

    def fail(self, dep):
        """Recursively disable jobs depending on failed job."""
        for job in self.jobs:
            if dep in job.dependencies:
                job.status = 'disabled'
                self.log(job)
                self.fail(job)

    def clean(self, cluster):
        for job in self.jobs:
            cluster.clean(job)


if __name__ == '__main__':
    jobs = AGTSJobs(sleeptime=2)
    jobs.collect()
    for job in jobs.jobs:
        job.walltime = 10
    c = TestCluster()
    j.run(c)
