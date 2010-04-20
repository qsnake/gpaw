def agts(queue):
    calc = queue.add('graphite_EELS.py',
                     ncpus=8,
                     walltime=200)
    
    queue.add('plot_spectra.py',
              ncpus=1,
              walltime=5,
              deps=[calc])
    
    queue.add('check_spectra.py',
              ncpus=1,
              walltime=5,
              deps=[calc])
