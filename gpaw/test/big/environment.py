import os
import sys
import time

import numpy as np

import gpaw

"""
state:
  waiting
  running
  done
  failed
"""


class AGTSJob:
    def __init__(self, agtsname, script, argstring, metadata, id):
        self.metadata = metadata
        name1, agts, py = agtsname.rsplit('.', 2)
        self.identifier = '.'.join([name1, agts, str(id), py])
        self.status = 'waiting'
        dir, filename = os.path.split(self.identifier)
        self.directory = dir
        self.filename = filename
        self.dependencies = [] # XXXX
        self.argstring = argstring

    def is_submitted(self):
        return self.status != 'waiting'

    def is_running(self):
        return self.status == 'running'

    def __str__(self):
        return 'AGTSJob[id=%s, metadata=%s]' % (self.identifier, self.metadata)

    def __repr__(self):
        return str(self)


class DryProfile:
    def __init__(self):
        self.rng = np.random.RandomState(42)
    
    def gpaw_compile_cmd(self):
        return 'echo hello'

    def submit(self, job):
        duration = self.rng.randint(4, 12)
        if self.rng.rand() < 0.4:
            # randomly fail some of the jobs
            result = 'failed'
        else:
            result = 'done'
            
        template = '(touch %s.start; sleep %s; touch %s.%s) &'
        os.system(template % (job.identifier, duration, job.identifier,
                              result))

    def update_job_status(self, job):
        id = job.identifier
        if os.path.exists('%s.done' % id):
            job.status = 'done'
        elif os.path.exists('%s.start' % id):
            job.status = 'running'
        if os.path.exists('%s.failed' % id):
            job.status = 'failed'
        

class AGTSRunner:
    def __init__(self, profile):
        self.profile = profile
    
    def check(self, jobs):
        stats = dict(waiting=0,
                     running=0,
                     done=0,
                     failed=0)
        for job in jobs:
            if not job.is_running():
                ready = True
                for dep in job.dependencies:
                    if dep.failed():
                        job.status = 'failed' # what should *happen* then?
                    if not dep.done():
                        ready = False
                        break
                
                if ready:
                    self.profile.submit(job)
            
            self.profile.update_job_status(job)
            stats[job.status] += 1
        return stats

    def run(self, ensemble):
        jobs = []
        for morejobs in ensemble.values():
            jobs.extend(morejobs)

        print len(jobs)
        for job in jobs:
            print
            print job
            print
        
        while True:
            stats = self.check(jobs)
            njobs = len(jobs)
            print stats
            ndone = stats['done'] + stats['failed']
            print 'ended: %d, total: %d' % (ndone, njobs)
            if ndone == njobs:
                return
            time.sleep(2.0)


class Collector:
    def __init__(self):
        pass
    
    def get_base_path(self):
        basepath = '/'.join(gpaw.__file__.split('/')[:-2])
        return basepath

    def locate_tests(self):
        basepath = self.get_base_path()
        for root, dirs, files in os.walk(basepath):
            for fname in files:
                if fname.endswith('.agts.py'):
                    yield root + '/' + fname

    def get_ensemble(self):
        ensemble = {}
        for fname in self.locate_tests():
            print fname
            metadatalist = []
            class MetaDataCollector:
                def add(self, script, intradepends=None, **kwargs):
                    tokens = script.split()
                    script = tokens[0]
                    argstring = ' '.join(tokens[1:])
                    #script, argstring = script.split('\\s', 1)
                    metadata = dict(kwargs)
                    if not 'depends' in metadata: # more 'cleaning' required?
                        metadata['depends'] = []
                    id = len(metadatalist)
                    key = (fname, script)
                    if intradepends is not None:
                        dep_key, dep_id = intradepends
                        dependency_job = ensemble[dep_key][dep_id]
                        metadata['depends'].append(dependency_job.identifier)
                    job = AGTSJob(fname, script, argstring, metadata, id)
                    if not metadatalist:
                        ensemble[key] = metadatalist
                    metadatalist.append(job)
                    return key, id
            
            m = MetaDataCollector()
            # yuckkk
            glob = {}
            execfile(fname, glob)
            main = glob['agtsmain']
            try:
                main(m)
            except:
                print >> sys.stderr, 'Bad things happened in %s' % fname
                raise
        return ensemble


if __name__ == '__main__':
    c = Collector()
    metadata = c.get_ensemble()
    for key, val in metadata.items():
        print key
        print val
        print
