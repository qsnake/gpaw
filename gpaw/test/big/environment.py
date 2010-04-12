import os
import time

import gpaw

"""
state:
  waiting
  running
  done
  failed
"""


class AGTSJob:
    def __init__(self, agtsname, script, metadata, id):
        self.metadata = metadata
        name1, agts, py = agtsname.rsplit('.', 2)
        self.identifier = '.'.join([name1, agts, str(id), py])
        self.status = 'waiting'
        dir, filename = os.path.split(self.identifier)
        self.directory = dir
        self.filename = filename
        self.dependencies = [] # XXXX

    def is_submitted(self):
        return self.status != 'waiting'

    def is_running(self):
        return self.status == 'running'

    def __str__(self):
        return 'AGTSJob[id=%s, metadata=%s]' % (self.identifier, self.metadata)

    def __repr__(self):
        return str(self)


class DryProfile:
    def gpaw_compile_cmd(self):
        return 'echo hello'

    def submit(self, job):
        template = '(touch %s.start; sleep 10; touch %s.done) &'
        os.system(template % (job.identifier, job.identifier))

    def update_job_status(self, job):
        id = job.identifier
        if os.path.exists('%s.done' % id):
            job.status = 'done'
        elif os.path.exists('%s.start' % id):
            job.status = 'running'
        

class AGTSRunner:
    def __init__(self, profile):
        self.profile = profile
    
    def check(self, jobs):
        stats = dict(waiting=0,
                     running=0,
                     done=0,
                     failed=0)
        for job in jobs:
        #for key, (script, job) in ensemble.items():
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
            njobs = len(stats)
            print stats
            over = stats['done'] + stats['failed']
            if over == njobs:
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
                def process(self, script, metadata, depends=None):
                    metadata = dict(metadata)
                    if not 'depends' in metadata: # more 'cleaning' required?
                        metadata['depends'] = []
                    id = len(metadatalist)
                    key = (fname, script)
                    if depends is not None:
                        dep_key, dep_id = depends
                        dependency_job = ensemble[dep_key][dep_id]
                        metadata['depends'].append(dependency_job.identifier)
                    job = AGTSJob(fname, script, metadata, id)
                    if not metadatalist:
                        ensemble[key] = metadatalist
                    metadatalist.append(job)
                    return key, id
            
            m = MetaDataCollector()
            # yuckkk
            glob = {}
            execfile(fname, glob)
            main = glob['main']
            main(m)
        return ensemble


if __name__ == '__main__':
    c = Collector()
    metadata = c.get_ensemble()
    for key, val in metadata.items():
        print key
        print val
        print
