import os
import time

jobs = {
    '../doc/exercises/runall': [],
    'vdw/interaction': ['vdw/dimers', 'vdw/benzene']}

class Jobs:
    def __init__(self, jobs):
        for name, requirements in jobs.items():
            for r in requirements:
                if r not in jobs:
                    jobs[r] = []
        self.jobs = jobs
        self.status = dict([(name, 'queued') for name in jobs])
        self.timer = {}

    def run(self):
        status = self.status
        while True:
            done = True
            for name in self.jobs:
                if status[name] == 'queued':
                    done = False
                    if not [None for r in self.jobs[name]
                            if status[r] != 'done']:
                        print 'Starting: %s.py' % name
                        self.status[name] = 'running'
                        try:
                            os.remove(name + '.status')
                        except OSError:
                            pass
                        self.timer[name] = time.time()
                        self.start_job(name)
                elif status[name] == 'running':
                    done = False

            if done:
                self.print_timing()
                return

            time.sleep(20.0)

            for name in self.jobs:
                if (status[name] == 'running' and
                    os.path.isfile(name + '.status')):
                    self.timer[name] = time.time() - self.timer[name]
                    code = int(open(name + '.status').readline())
                    if code == 0:
                        status[name] = 'done'
                        print name + '.py done.'
                    else:
                        status[name] = 'failed'
                        print '%s.py exited with errorcode: %d' % (name, code)
                        self.error(name, code)
                        for n, requirements in self.jobs.items():
                            if name in requirements:
                                status[n] = 'disabled'
                                print 'Disabling %s.py' % n

    def start_job(self, name):
        dir = os.path.dirname(name)
        name = os.path.basename(name)
        py = ('import os;' +
              "x = os.system('python %s.py');" % name +
              "f = open('%s.status', 'w');" % name +
              "f.write('%d' % x)")
        print py
        os.system('cd %s; python -c "%s" > %s.output' % (dir, py, name))

    def error(self, name, code):
        pass

    def print_timing(self):
        print self.timer

class MyJobs(Jobs):
    def start_job(self, name):
        dir = os.path.dirname(name)
        name = os.path.basename(name)
        cmd = ('cd %s; ' % dir +
               'mpirun /home/camp/jensj/gpaw/build/bin.linux-x86_64-2.3/gpaw-python %s.py > %s.output' % (name, name))
        i = os.popen('qsub -q small -l nodes=1:ppn=4:ethernet', 'w')
        i.write('\n'.join(
            ['#!/usr/bin/env python',
             'import os',
             'x = os.system("%s")' % cmd,
             'f = open("%s/%s.status", "w")' % (dir, name),
             'f.write("%d" % x)',
             '\n']))

j = MyJobs(jobs)
j.run()
