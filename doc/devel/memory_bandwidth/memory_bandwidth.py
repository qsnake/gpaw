#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Emacs: treat this as -*- python -*-

from optparse import OptionParser

parser = OptionParser(usage='%prog [options] element',
                      version='%prog 0.1')
parser.add_option('--dir', dest="dir",
                  default='.',
                  help='Results directory')

opt, args = parser.parse_args()

import os

import datetime

from math import sqrt

import numpy as npy

colors = [
    'black',
    'brown',
    'red',
    'orange',
    'yellow',
    'green',
    'blue',
    'violet',
    'gray',
    'gray']

from ase.data import atomic_numbers as numbers

def plot(xdata, ydata, std,
         title,
         xlabel, ylabel,
         label, color, num=1):
    import matplotlib
    matplotlib.use('Agg')
    import pylab

    # all goes to figure num
    pylab.figure(num=num, figsize=(7, 5.5))
    pylab.gca().set_position([0.10, 0.20, 0.85, 0.60])
    # let the plot have fixed y-axis scale
    miny = min(ydata)
    maxy = max(ydata)
    ywindow = maxy - miny
    pylab.gca().set_ylim(miny-ywindow/4.0, maxy+ywindow/3.0)
    #pylab.plot(xdata, ydata, 'b.', label=label, color=color)
    #pylab.plot(xdata, ydata, 'b-', label='_nolegend_', color=color)
    pylab.bar(xdata, ydata, 0.3, yerr = std, label=label, color=color)
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    #pylab.legend(loc='upper right')
    #pylab.savefig(directory_name + os.path.sep + out_prefix +'.png')

def plot_save(directory_name, out_prefix):
    from os.path import exists
    assert exists(directory_name)
    import pylab

    pylab.savefig(directory_name + os.path.sep + out_prefix +'.png')

def analyse_benchmark(ncores=8, machine='TEST'):
    #system = ['carbon_py']
    #system = ['carbon']
    #system = ['niflheim_py']
    #system = ['niflheim']
    #system = ['TEST_py']
    system = machine+'_py'

    systems_string = {
        'carbon_py' : 'gpaw 1865 on carbon',
        'carbon' : 'mkl 10.0.2.018 dsyev on carbon',
        'niflheim_py' : 'gpaw 1865 on niflheim',
        'niflheim' : 'acml 4.0.1 dsyev on niflheim',
        #'TEST_py' : 'gpaw on TEST',
        }.get(system, False)

    processes = {
        'carbon_py' : [1, 2, 4, 6, 8],
        'carbon' : [1, 2, 4, 8],
        'niflheim_py' : [1, 2, 3, 4],
        'niflheim' : [1, 2, 4],
        #'TEST_py' : [1, 2, 4, 6, 8],
        }.get(system, False)


    if not systems_string:
        systems_string = 'gpaw on '+machine
    if not processes:
        processes = [1]
        for n in range(1, ncores+1):
            if n%2==0:
                processes.append(n)

    if system.find('_py') == -1:
        timer_entries_all = [
            'run: 0',
            'run: 1',
            'run: 2',
            'run: 3',
            'run: 4',
            'run: 5',
            'run: 6',
            'run: 7',
            'run: 8',
            'run: 9'
            ]
    else:
        timer_entries_all = [
            'Run:  0',
            'Run:  1',
            'Run:  2',
            'Run:  3',
            'Run:  4',
            'Run:  5',
            'Run:  6',
            'Run:  7',
            'Run:  8',
            'Run:  9'
            ]


    import re

    # Select timer entries
    selected_entries = range(10)

    height = {}

    pre_results = {}
    results = {}

    timer_entries = []
    timer_entries_re = {}
    for entry in selected_entries:
        height[entry] = []
        timer_entries.append(timer_entries_all[entry])
        timer_entries_re[timer_entries_all[entry]] = re.compile(timer_entries_all[entry])

    # absolute path to directory
    root_abspath = os.path.abspath(opt.dir)
    # lenght of directory name
    rootlen = len(root_abspath) + 1

    ref_value = -44.85826
    tolerance = 0.0001

    ref_failed = False
    h_failed = False
    for run in [str(p)+'_01' for p in processes]:
        # extract results
        rundir = os.path.join(root_abspath, system+run)
        file = os.path.join(rundir, 'out.txt')
        try:
            f = open(file, 'r')
            #
            print 'Analysing '+file
            #
            lines = f.readlines()
        except: pass
        # search for timings
        for entry in selected_entries:
            h = []
            ref = []
            for line in lines:
                m = timer_entries_re[timer_entries_all[entry]].search(line)
                if m is not None:
                    h.append(float(line.split(':')[-1]))
                   #break # stop after the first match
            for h_entry in h:
                if float(h_entry) < 0.0:
                    h_failed = True
                    break
            height[entry].append(h)
            for line in lines:
                m = re.compile('Zero').search(line)
                if m is not None:
                    ref.append(float(line.split(':')[-1]))
                   #break # stop after the first match
            for ref_entry in ref:
                if abs(float(ref_entry)-ref_value) > tolerance:
                    ref_failed = True
                    break
    #
    if h_failed:
        print 'Panic: negative time in '+file
        assert not h_failed
    if ref_failed:
        print 'Panic: wrong Zero Kelvin: value in '+file+' - should be '+str(ref_value)+' +- '+str(tolerance)
        assert not ref_failed
    # arrange results
    for p in range(len(processes)):
        pre_results[processes[p]] = []
        for i in range(len(height)):
            pre_results[processes[p]].append(height[i][p])
    #
    # arrange results - calculate statistics
    for p in processes:
    #for p in range(len([1])):
        #print pre_results[p]
        results[p] = []
        temp = []
        for q in range(p):
            for i in range(len(pre_results[p])):
                #print pre_results[p][i][q]
                temp.append(pre_results[p][i][q])
            results[p].append((npy.average(temp), npy.std(temp)))
    #for p in processes:
    #    #N = len(pre_results[p])
    #    #avg = sum(pre_results[p])/N
    #    #q = sqrt(sum([(x-avg)**2/(N) for x in pre_results[p]]))
    #    avg.append(npy.average(pre_results[p]))
    #    q.append(npy.std(pre_results[p]))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pylab, ticker
    from twiny import twiny
    # from http://matplotlib.sourceforge.net/examples/dashtick.py
    ROTATION=75
    DASHROTATION=115
    DASHBASE=5
    DASHLEN=25
    DASHSTAGGER=3
    FONTSIZE=10
    def dashlen(step):
        return DASHBASE+(DASHLEN*(step%DASHSTAGGER))
    # print scaling results
    parameters = processes
    zero = [0.0 for i in range(len(parameters))]
    pylab.plot(parameters, zero, 'k-', label='_nolegend_')
    ay1=pylab.gca()
    ay1.xaxis.set_ticks(parameters)
    ay1.xaxis.set_ticklabels([str(x) for x in parameters])
    for p in processes:
        parameters = []
        avg = []
        std = []
        for i in range(len(results[p])):
            parameters.append(p+0.3*i)
            avg.append(results[p][i][0])
            std.append(results[p][i][1])
        # height
        #print parameters, avg, std
        print 'No. of processes '+str(int(parameters[0]))+' Runtime '+str(round(max(avg),2))+' sec'
        plot(
            parameters, avg, q,
            systems_string,
            'processes per node',
            'time [s]',
            'gpaw',
            (colors[p%10]),
            num=1)
    # from two_scales.py
    plot_save(".", 'memory_bandwidth_'+system)
    pylab.close(1)
#

if __name__ == '__main__':
    from os import environ

    NCORES = int(environ.get('NCORES', 8))
    MACHINE = environ.get('MACHINE', 'TESTA')
    assert NCORES > 1, str(NCORES)+' must be > 1'

    analyse_benchmark(NCORES, MACHINE)
