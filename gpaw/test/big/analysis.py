#!/usr/bin/env python
import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as pl
import tempfile

"""Database structure:
dict(testname: [(date, runtime, info), (date, runtime, info), ...])
    date: Time since epoch in seconds
    runtime: Run time in seconds. Negative for crashed jobs!
    info: A string describing the outcome
"""

#TODO: Date in x-axis
#    : Only send mail if result is recent 

class DatabaseHandler:
    """Database class for keeping timings and info for long tests"""
    def __init__(self, filename):
        self.filename = filename
        self.data = dict()

    def read(self):
        if os.path.isfile(self.filename):
            self.data = pickle.load(file(self.filename))
        else:
            print 'File does not exist, starting from scratch'

    def write(self, filename=None):
        if filename is None:
            filename = self.filename
        if os.path.isfile(filename):
            os.rename(filename, filename + '.old')
        pickle.dump(self.data, open(filename, 'wb'), -1)

    def add_data(self, name, date, runtime, info):
        if not self.data.has_key(name):
            self.data[name] = []
        self.data[name].append((date, runtime, info))

    def get_data(self, name):
        """Return date_array, time_array"""
        dates, runtimes = [], []
        if self.data.has_key(name):
            for datapoint in self.data[name]:
                dates.append(datapoint[0])
                runtimes.append(datapoint[1])

        return np.asarray(dates), np.asarray(runtimes)

    def update(self, queue):
        """Add all new data to database"""
        for job in queue.jobs:
            absname = job.absname

            tstart = job.tstart
            if tstart is None:
                tstart = np.nan
            tstop = job.tstop
            if tstop is None:
                tstop = np.nan

            info = job.status

            self.add_data(absname, 0, tstop - tstart, info)

class TestAnalyzer:
    def __init__(self, name, dates, runtimes):
        self.name = name
        self.dates = dates
        self.runtimes = runtimes
        self.better = []
        self.worse = []
        self.relchange = None
        self.abschange = None

    def analyze(self, reltol=0.1, abstol=5.0):
        """Analyze timings

        When looking at a point, attention is needed if it deviates more than
        10\% from the median of previous points. If such a point occurs the
        analysis is restarted.
        """
        self.better = []
        self.worse = []
        abschange = 0.0
        relchange = 0.0
        status = 0
        current_first = 0   # Point to start analysis from
        for i in range(1, len(self.runtimes)):
            tmpruntimes = self.runtimes[current_first:i]
            median = np.median(tmpruntimes[np.isfinite(tmpruntimes)])
            if np.isnan(median):
                current_first = i
            elif np.isfinite(self.runtimes[i]):
                abschange = self.runtimes[i] - median
                relchange = abschange / median
                if relchange < -reltol and abschange < -abstol:
                    # Improvement
                    current_first = i
                    self.better.append(i)
                    status = -1
                elif relchange > reltol and abschange > abstol:
                    # Regression 
                    current_first = i
                    self.worse.append(i)
                    status = 1
                else:
                    status = 0

        self.status = status
        self.abschange = abschange
        self.relchange = relchange * 100

    def plot(self, outputdir=None):
        if outputdir is None:
            return
        fig = pl.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.runtimes, 'ko-')
        ax.plot(self.better, self.runtimes[self.better], 'go', markersize=5)
        ax.plot(self.worse, self.runtimes[self.worse], 'ro', markersize=5)
        ax.set_title(self.name)
        if not outputdir.endswith('/'):
            outputdir += '/'
        figname = self.name.replace('/','_')
        fig.savefig(outputdir + figname + '.png')

class MailGenerator:
    def __init__(self):
        self.better = []
        self.worse = []

    def add_test(self, name, abschange, relchange):
        if abschange < 0.0:
            self.add_better(name, abschange, relchange)
        else:
            self.add_worse(name, abschange, relchange)

    def add_better(self, name, abschange, relchange):
        self.better.append((name, abschange, relchange))

    def add_worse(self, name, abschange, relchange):
        self.worse.append((name, abschange, relchange))

    def generate_mail(self):
        mail = 'Results from weekly tests:\n\n'
        if len(self.better):
            mail += 'The following tests improved:\n'
            for test in self.better:
                mail += '%-20s %7.2f s (%7.2f%%)\n' % test
        else:
            mail += 'No tests improved!\n'
        mail += '\n'
        if len(self.worse):
            mail += 'The following tests regressed:\n'
            for test in self.worse:
                mail += '%-20s +%6.2f s (+%6.2f%%)\n' % test
        else:
            mail += 'No tests regressed!\n'

        return mail

    def send_mail(self, address):
        fullpath = tempfile.mktemp()
        f = open(fullpath, 'w')
        f.write(self.generate_mail())
        f.close()
        os.system('mail -s "Results from weekly tests" %s < %s' % \
                  (address, fullpath))

#def csv2database(infile, outfile):
#    """Use this file once to import the old data from csv"""
#    csvdata = np.recfromcsv(infile)
#    db = DatabaseHandler(outfile)
#    for test in csvdata:
#        name = test[0]
#        for i in range(1, len(test) - 1):
#            runtime = float(test[i])
#            info = ''
#            db.add_data(name, 0, runtime, info)
#    db.write()

def analyse(queue, dbpath, outputdir=None, mailto=None):
    db = DatabaseHandler(dbpath)
    db.read()
    db.update(queue)
    db.write()
    mg = MailGenerator()
    for job in queue.jobs:
        name = job.absname
        dates, runtimes = db.get_data(name)
        ta = TestAnalyzer(name, dates, runtimes)
        ta.analyze()
        if ta.status:
            mg.add_test(name, ta.abschange, ta.relchange)
        ta.plot(outputdir)

    if mailto is not None:
        mg.send_mail(mailto)
    else:
        print mg.generate_mail()

if __name__ == "__main__":
    analyse(None)
